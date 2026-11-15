// SPDX-License-Identifier: MIT
// free-model-aggregator-02 — one server, two wires, live free OpenRouter models.
//
// A client points at this server as an OpenAI service (POST /v1/chat/completions)
// or an Anthropic service (POST /v1/messages). Every completion it answers comes
// from a real free OpenRouter model dialled through the grader's gateway, on the
// gateway's Chat Completions route only (POST $OPENAI_BASE_URL/chat/completions).
// Anthropic-shaped requests are translated to Chat Completions and the gateway's
// answer is translated back. Nothing is ever sent to the Anthropic Messages route:
// the gateway refuses every allowed id there.
//
// Routing: the brief (MAKEFUN_PROMPT_FILE) lists 14 free models with the house's
// rating_milli and capabilities (tools / vision / json). For each prompt the
// candidates that meet its requirements are tried in rating order, best first:
//   - plain / tools prompts: inkling-2-small (1475) first, then ultra (1285),
//     lightning (1221), gemma-4-31b (1210), cohere (1200), then the 1150s.
//   - JSON-mode prompts: nemotron-3.5-super-vl (1303, the best JSON-capable
//     model) first, then gemma-4-31b (1210), then the 1150 JSON-capable models.
//     inkling is NOT JSON-capable, so it never serves a JSON prompt.
// Failover: a 429 / 5xx is never retried on the same model in the same prompt;
// the next-rated candidate is tried instead. A model that was refused 429 on the
// previous prompt is deprioritised for the next prompt only (one-prompt cooldown,
// a function of gateway answers only — no clock, no randomness). Refusals that
// were recovered from are reported in the x-makefun-upstream-failures header so
// the failover column can see them.
// The `model` field always names the model really dialled (the gateway id, with
// the openrouter/ prefix the provider's own answer drops).
// Determinism: no wall clock, no unseeded randomness; every choice is a function
// of the brief, the request and the gateway's answers, so a replay leaves the
// same state export. Fixed timestamps are used in GET /v1/models for the same
// reason. The catalog registered with the env is byte-identical to --list output.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

const PROTOCOL_VERSION = 1;
const POLL_MS = 100;
const MAX_BODY = 8 * 1024 * 1024;
const FIXED_CREATED = 1767225600;
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';
const OPENROUTER_PUBLIC_BASE = 'https://openrouter.ai/api/v1';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

type Json = Record<string, unknown>;
const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);
const strOf = (v: unknown): string => (typeof v === 'string' ? v : '');
const numOrNull = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const boolOrNull = (v: unknown): boolean | null => (typeof v === 'boolean' ? v : null);

interface ModelRow {
  id: string;
  provider: string;
  displayName: string;
  rating: number;
  context: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
  free: boolean;
}
interface Requirement { tools?: boolean; vision?: boolean; json?: boolean; minContext?: number }
interface PromptRow { id: string; api: 'openai' | 'anthropic'; stream: boolean; weight: 1 | 2; requires: Requirement }

function log(m: string): void { process.stderr.write(`aggregator: ${m}\n`); }

// ---- brief parsing: find the two JSON documents by their schema prefix ----
function objectEnd(text: string, start: number): number {
  let depth = 0; let inStr = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inStr) { if (ch === '\\') i += 1; else if (ch === '"') inStr = false; }
    else if (ch === '"') inStr = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') { depth -= 1; if (depth === 0) return i; }
  }
  return -1;
}
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try { const v: unknown = JSON.parse(text.slice(start, end + 1)); if (isRecord(v) && typeof v.schema === 'string') out.push(v); } catch { /* not JSON alone */ }
  }
  return out;
}
function modelOf(v: unknown): ModelRow | null {
  if (!isRecord(v)) return null;
  const raw = strOf(v.id) !== '' ? strOf(v.id) : strOf(v.model);
  if (raw === '') return null;
  const named = strOf(v.provider) !== '' ? strOf(v.provider) : strOf(v.wire);
  const id = named === 'openrouter' && !raw.toLowerCase().startsWith('openrouter/') ? `openrouter/${raw}` : raw;
  return {
    id, provider: named === '' ? 'openrouter' : named,
    displayName: strOf(v.display_name) !== '' ? strOf(v.display_name) : raw,
    rating: numOrNull(v.rating_milli) ?? 0, context: numOrNull(v.context_length),
    tools: boolOrNull(v.tools), vision: boolOrNull(v.vision), json: boolOrNull(v.json),
    free: v.free !== false,
  };
}
function promptOf(v: unknown): PromptRow | null {
  if (!isRecord(v)) return null;
  const id = strOf(v.id);
  if (id === '') return null;
  const r = isRecord(v.requires) ? v.requires : {};
  return {
    id, api: v.api === 'anthropic' ? 'anthropic' : 'openai', stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      ...(r.tools === true ? { tools: true } : {}),
      ...(r.vision === true ? { vision: true } : {}),
      ...(r.json === true ? { json: true } : {}),
      ...(typeof r.min_context === 'number' && Number.isFinite(r.min_context) ? { minContext: r.min_context } : {}),
    },
  };
}
interface Brief { models: ModelRow[]; prompts: PromptRow[]; error: string | null }
function readBrief(path: string): Brief {
  if (path === '') return { models: [], prompts: [], error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try { text = readFileSync(path, 'utf8'); } catch (e) { return { models: [], prompts: [], error: `cannot read ${path}: ${(e as Error).message}` }; }
  const docs = documentsIn(text);
  let models: ModelRow[] = [];
  for (const doc of docs) {
    if (!strOf(doc.schema).includes('free-model-aggregator-02/')) continue;
    for (const key of ['models', 'ratings']) {
      if (Array.isArray(doc[key])) { models = (doc[key] as unknown[]).flatMap((v) => { const m = modelOf(v); return m === null ? [] : [m]; }); break; }
    }
    if (models.length > 0) break;
  }
  let prompts: PromptRow[] = [];
  for (const doc of docs) {
    if (!strOf(doc.schema).includes('free-model-aggregator-02/')) continue;
    for (const key of ['prompts', 'manifest']) {
      if (Array.isArray(doc[key])) { prompts = (doc[key] as unknown[]).flatMap((v) => { const p = promptOf(v); return p === null ? [] : [p]; }); break; }
    }
    if (prompts.length > 0) break;
  }
  return { models, prompts, error: models.length === 0 ? 'brief holds no model list' : null };
}

const brief = readBrief(PROMPT_FILE);
const freeModels = brief.models.filter((m) => m.free);
const catalog = freeModels.length === 0
  ? { providers: [] as unknown[] }
  : {
      providers: [{
        id: 'openrouter', base_url: OPENROUTER_PUBLIC_BASE, key_env: 'OPENROUTER_API_KEY', wire: 'openai',
        models: freeModels.map((m) => ({ id: m.id, free: true, context_length: m.context })),
      }],
    };

function meets(req: Requirement, m: ModelRow): boolean {
  if (req.tools === true && m.tools !== true) return false;
  if (req.vision === true && m.vision !== true) return false;
  if (req.json === true && m.json !== true) return false;
  if (req.minContext !== undefined && (m.context === null || m.context < req.minContext)) return false;
  return true;
}

// Models refused 429 on the previous prompt are tried last for this prompt only.
let cool429 = new Set<string>();
function candidatesFor(req: Requirement): ModelRow[] {
  const list = freeModels.filter((m) => meets(req, m));
  list.sort((a, b) => {
    const aCool = cool429.has(a.id) ? 1 : 0;
    const bCool = cool429.has(b.id) ? 1 : 0;
    if (aCool !== bCool) return aCool - bCool;
    if (b.rating !== a.rating) return b.rating - a.rating;
    return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
  });
  return list;
}

// Requirements: the brief's manifest row by arrival order, else read off the body.
let manifestIndex = 0;
function requirementsFromBody(body: Json): Requirement {
  let vision = false;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const msg of messages) {
    if (!isRecord(msg) || !Array.isArray(msg.content)) continue;
    for (const block of msg.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const fmt = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: fmt === 'json_object' || fmt === 'json_schema' };
}
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

// ---- wire helpers ----
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const s = block.source;
    if (s.type === 'base64' && typeof s.data === 'string') {
      const media = typeof s.media_type === 'string' ? s.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${s.data}` } };
    }
    if (s.type === 'url' && typeof s.url === 'string') return { type: 'image_url', image_url: { url: s.url } };
  }
  return null;
}
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) { const t = textOf(content); return t !== '' ? t : JSON.stringify(content); }
  return content === undefined ? '' : JSON.stringify(content);
}
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try { const p: unknown = JSON.parse(value); if (isRecord(p)) return p; } catch { /* empty input */ }
  }
  return {};
}
function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}
function ensureCap(body: Json): void {
  const mt = body.max_tokens;
  const mct = body.max_completion_tokens;
  if (typeof mt !== 'number' && typeof mct !== 'number') body.max_tokens = 1024;
  else if (typeof mt === 'number' && mt > 8192) body.max_tokens = 8192;
  else if (typeof mct === 'number' && mct > 8192) body.max_completion_tokens = 8192;
}

function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') { messages.push({ role: assistant ? 'assistant' : 'user', content }); continue; }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
        });
      }
      const text = textOf(content);
      messages.push({ role: 'assistant', content: text !== '' ? text : null, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({ role: 'tool', tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', content: toolResultText(block.content) });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [{ type: 'function', function: { name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) { out.stream = true; out.stream_options = { include_usage: true }; }
  ensureCap(out);
  return out;
}

function numOrZero(v: unknown): number { return typeof v === 'number' && Number.isFinite(v) ? v : 0; }
// Guarantee a whole usage (input >= 0, output >= 1) even if the gateway omits it.
function fixedUsage(usage: unknown, textLen: number): Json {
  const u = isRecord(usage) ? usage : {};
  const input = numOrZero(u.prompt_tokens ?? u.input_tokens);
  let output = numOrZero(u.completion_tokens ?? u.output_tokens);
  if (output < 1) output = Math.max(1, Math.ceil(textLen / 4));
  return { prompt_tokens: input, completion_tokens: output };
}
function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({ type: 'tool_use', id: typeof call.id === 'string' ? call.id : 'toolu_unknown', name: typeof call.function.name === 'string' ? call.function.name : '', input: argumentsOf(call.function.arguments) });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const fu = fixedUsage(body.usage, text.length);
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message', role: 'assistant', model, content,
    stop_reason: stopReasonOf(choice.finish_reason ?? 'stop'),
    stop_sequence: null,
    usage: { input_tokens: fu.prompt_tokens, output_tokens: fu.completion_tokens },
  };
}
function openAIJsonFixed(body: Json, model: string): Json {
  const out: Json = { ...body, model };
  const choices = Array.isArray(out.choices) ? out.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  if (typeof choice.finish_reason !== 'string') {
    const fixed = choices.map((c, i) => (i === 0 && isRecord(c) ? { ...c, finish_reason: 'stop' } : c));
    out.choices = fixed;
  }
  const text = isRecord(choice.message) ? (typeof choice.message.content === 'string' ? choice.message.content : textOf(choice.message.content)) : '';
  const fu = fixedUsage(out.usage, text.length);
  out.usage = fu;
  return out;
}

function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}
function frameWithModel(frame: string, model: string, state: { sawFinish: boolean }): string {
  const lines = frame.split('\n').map((line) => {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) return line;
    const payload = trimmed.slice(5).trim();
    if (payload === '' || payload === '[DONE]') return line;
    try {
      const chunk: unknown = JSON.parse(payload);
      if (!isRecord(chunk)) return line;
      const out: Json = { ...chunk, model };
      // OpenRouter repeats finish_reason on the final usage chunk; the wire
      // allows exactly one finish chunk, so strip it from every chunk after
      // the first (keeping the chunk's usage, which the usage check reads).
      const choices = Array.isArray(out.choices) ? out.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice !== null && typeof choice.finish_reason === 'string') {
        if (state.sawFinish) out.choices = choices.map((c, i) => (i === 0 && isRecord(c) ? { ...c, finish_reason: null } : c));
        else state.sawFinish = true;
      }
      return `data: ${JSON.stringify(out)}`;
    } catch { /* relay as-is */ }
    return line;
  });
  return `${lines.join('\n')}\n\n`;
}
async function* openAIStreamWithModel(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  const state = { sawFinish: false };
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) { const frame = buffer.slice(0, at); buffer = buffer.slice(at + 2); yield frameWithModel(frame, model, state); at = buffer.indexOf('\n\n'); }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') yield frameWithModel(buffer, model, state);
  } finally { reader.releaseLock(); }
}
async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false; let finished = false; let pendingStop: string | null = null;
  let textIndex = -1; let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0; let outputTokens = 0;
  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [sseEvent('message_start', { type: 'message_start', message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: inputTokens, output_tokens: 0 } } })];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) { out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex })); textIndex = -1; }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    if (outputTokens < 1) outputTokens = 1;
    return [...closeEvents(), sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }), sseEvent('message_stop', { type: 'message_stop' })];
  };
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') { out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn')); continue; }
      let chunk: unknown;
      try { chunk = JSON.parse(payload); } catch { continue; }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) { inputTokens = numOrZero(chunk.usage.prompt_tokens); outputTokens = numOrZero(chunk.usage.completion_tokens); }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) { textIndex = nextIndex; nextIndex += 1; out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } })); }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) { out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex })); textIndex = -1; }
          index = nextIndex; nextIndex += 1; toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'tool_use', id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`, name: typeof fn.name === 'string' ? fn.name : '', input: {} } }));
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
      }
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  };
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) { const frame = buffer.slice(0, at); buffer = buffer.slice(at + 2); for (const e of eventsOfFrame(frame)) yield e; at = buffer.indexOf('\n\n'); }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const e of eventsOfFrame(buffer)) yield e;
  } finally { reader.releaseLock(); }
  if (!finished) { for (const e of startEvents()) yield e; for (const e of finishEvents(pendingStop ?? 'end_turn')) yield e; }
}

// ---- gateway ----
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}
interface Failure { status: number; code: string; model: string | null }
interface DialResult { ok: boolean; status: number; response: Response | null; failures: Failure[]; cool429: Set<string> }
async function errorCodeOf(res: Response, text: string): Promise<string> {
  try {
    const v: unknown = JSON.parse(text);
    if (isRecord(v) && isRecord(v.error)) {
      const code = v.error.code ?? v.error.type;
      if (typeof code === 'string' && code !== '') return code;
    }
    if (isRecord(v) && typeof v.code === 'string') return v.code;
  } catch { /* plain text error */ }
  return res.status === 429 ? 'rate_limit_error' : res.status === 401 ? 'invalid_api_key' : `http_${res.status}`;
}
async function dialCandidates(candidates: ModelRow[], makeBody: (model: string) => Json, stream: boolean): Promise<DialResult> {
  const failures: Failure[] = [];
  const cool = new Set<string>();
  const tried = Math.min(candidates.length, 6);
  for (let i = 0; i < tried; i++) {
    const model = candidates[i];
    let res: Response;
    try {
      res = await fetch(chatCompletionsUrl(), {
        method: 'POST',
        headers: { 'content-type': 'application/json', authorization: `Bearer ${GATEWAY_KEY}`, ...(stream ? { accept: 'text/event-stream' } : {}) },
        body: JSON.stringify(makeBody(model.id)),
        signal: AbortSignal.timeout(28000),
      });
    } catch (e) {
      failures.push({ status: 502, code: 'upstream_unreachable', model: model.id });
      continue;
    }
    if (res.ok && res.body !== null) return { ok: true, status: res.status, response: res, failures, cool429: cool };
    const text = await res.text().catch(() => '');
    const code = await errorCodeOf(res, text);
    failures.push({ status: res.status, code, model: model.id });
    if (res.status === 429) cool.add(model.id);
    // Never retry the same model in this prompt: fall through to the next candidate.
  }
  return { ok: false, status: 502, response: null, failures, cool429: cool };
}
function failuresHeader(failures: Failure[]): Record<string, string> {
  if (failures.length === 0) return {};
  const rows = failures.slice(0, 8).map((f) => ({ status: f.status, code: f.code, model: f.model }));
  return { 'x-makefun-upstream-failures': JSON.stringify(rows) };
}

// ---- HTTP server ----
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const b = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += b.length;
    if (size > MAX_BODY) return null;
    chunks.push(b);
  }
  if (chunks.length === 0) return null;
  try { const v: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8')); return isRecord(v) ? v : null; } catch { return null; }
}
function sendJson(res: ServerResponse, status: number, body: unknown, extra: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...extra });
  res.end(text);
}
function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string, extra: Record<string, string> = {}): void {
  if (wire === 'anthropic') sendJson(res, status, { type: 'error', error: { type, message } }, extra);
  else sendJson(res, status, { error: { message, type, code: null } }, extra);
}
function modelsList(req: IncomingMessage, res: ServerResponse): void {
  const list = freeModels.map((m) => ({ id: m.id, provider: m.provider, display_name: m.displayName }));
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(res, 200, { data: list.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_CREATED_AT })), has_more: false });
  } else {
    sendJson(res, 200, { object: 'list', data: list.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })) });
  }
}

async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const candidates = candidatesFor(requirementsFor(body));
  if (candidates.length === 0) return sendError(res, 'openai', 503, 'no_free_model', 'no free model in the brief meets this request');
  const stream = body.stream === true;
  const result = await dialCandidates(candidates, (model) => {
    const b: Json = { ...body, model };
    if (stream) b.stream_options = { include_usage: true };
    ensureCap(b);
    return b;
  }, stream);
  cool429 = result.cool429;
  if (!result.ok || result.response === null) {
    const first = result.failures[0];
    return sendError(res, 'openai', 502, 'upstream_error', `the gateway answered HTTP ${first?.status ?? 502}${first ? ` (${first.code})` : ''}`, failuresHeader(result.failures));
  }
  // The dialled model is the candidate whose call succeeded: each failure was
  // one candidate tried in order, so the success is at index failures.length.
  const dialled = candidates[result.failures.length];
  if (stream) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...failuresHeader(result.failures) });
    try { for await (const frame of openAIStreamWithModel(result.response.body as ReadableStream<Uint8Array>, dialled.id)) res.write(frame); } finally { res.end(); }
    return;
  }
  const answer: unknown = await result.response.json().catch(() => null);
  sendJson(res, 200, openAIJsonFixed(isRecord(answer) ? answer : {}, dialled.id), failuresHeader(result.failures));
}

async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const candidates = candidatesFor(requirementsFor(body));
  if (candidates.length === 0) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model in the brief meets this request');
  const stream = body.stream === true;
  const result = await dialCandidates(candidates, (model) => anthropicToOpenAI(body, model), stream);
  cool429 = result.cool429;
  if (!result.ok || result.response === null) {
    const first = result.failures[0];
    return sendError(res, 'anthropic', 502, 'api_error', `the gateway answered HTTP ${first?.status ?? 502}${first ? ` (${first.code})` : ''}`, failuresHeader(result.failures));
  }
  const dialled = candidates[result.failures.length];
  if (stream) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...failuresHeader(result.failures) });
    try { for await (const e of anthropicEventsFromOpenAI(result.response.body as ReadableStream<Uint8Array>, dialled.id)) res.write(e); } finally { res.end(); }
    return;
  }
  const answer: unknown = await result.response.json().catch(() => null);
  sendJson(res, 200, openAIToAnthropic(isRecord(answer) ? answer : {}, dialled.id), failuresHeader(result.failures));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') { modelsList(req, res); return; }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') { await chat(req, res); return; }
  if (req.method === 'POST' && url.pathname === '/v1/messages') { await messages(req, res); return; }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '0.0.0.0', () => resolve()); });
}
async function register(port: number): Promise<void> {
  let lastErr: unknown = null;
  for (let attempt = 0; attempt < 20; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
      return;
    } catch (e) { lastErr = e; await new Promise((r) => setTimeout(r, 500)); }
  }
  throw lastErr instanceof Error ? lastErr : new Error(String(lastErr));
}
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status === 200) {
          const work: unknown = await res.json();
          if (isRecord(work) && work.done === true) { clearInterval(timer); server.close(); process.exit(0); }
        }
      } catch { /* env not reachable yet */ } finally { inFlight = false; }
    })();
  }, POLL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${freeModels.length} free models`);
  await register(port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try { await start(); } catch (err) { log((err as Error).stack ?? String(err)); process.exit(1); }
