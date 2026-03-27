#!/usr/bin/env python3
"""
MLX Native Anthropic Server — Claude Code on Apple Silicon.
Single-file server: MLX inference + Anthropic Messages API + KV cache quantization.
No proxy. No translation layer. Direct.

Supports full Claude Code feature set:
  - Tool calling (function calling) via Qwen's <tool_call> format
  - SSE streaming (stream: true) for real-time token delivery
  - Tool result messages (role: "tool") passed back via chat template
"""

import json
import os
import re
import sys
import threading
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

import mlx.core as mx
from mlx_lm.utils import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler

# ─── Configuration ───────────────────────────────────────────────────────────

MODEL_PATH = os.environ.get("MLX_MODEL", "mlx-community/Qwen3.5-122B-A10B-4bit")
PORT = int(os.environ.get("MLX_PORT", "4000"))
KV_BITS = int(os.environ.get("MLX_KV_BITS", "4"))
PREFILL_SIZE = int(os.environ.get("MLX_PREFILL_SIZE", "4096"))
DEFAULT_MAX_TOKENS = int(os.environ.get("MLX_MAX_TOKENS", "8192"))

# ─── Globals ─────────────────────────────────────────────────────────────────

model = None
tokenizer = None
generate_lock = threading.Lock()


# ─── Logging ─────────────────────────────────────────────────────────────────

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# ─── Model Loading ───────────────────────────────────────────────────────────

def load_model():
    global model, tokenizer
    log(f"Loading model: {MODEL_PATH}")
    t0 = time.time()
    model, tokenizer = load(MODEL_PATH)
    mx.eval(model.parameters())
    elapsed = time.time() - t0
    log(f"Model loaded in {elapsed:.1f}s")
    log(f"KV cache quantization: {KV_BITS}-bit" if KV_BITS else "KV cache: full precision")


# ─── Think Tag Stripping ────────────────────────────────────────────────────

def strip_think_tags(text):
    """Remove think-block content from Qwen's reasoning output.

    Handles two patterns:
    1. Complete block  — <think>…</think> present in full_text.
    2. Orphan close    — generation prompt already prepended <think>, so
       full_text starts with the *content* of the think block (no opening
       tag) and contains only </think> as the separator.  Strip everything
       from the start of the string up to and including the first </think>.
    """
    # Pattern 1: remove complete <think>…</think> blocks
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Pattern 2: remove leading orphan thinking content before </think>
    if '</think>' in cleaned:
        cleaned = cleaned[cleaned.index('</think>') + len('</think>'):]
    cleaned = cleaned.strip()
    return cleaned if cleaned else text


def clean_response(text):
    """Strip think tags and clean reasoning artifacts."""
    text = strip_think_tags(text)

    # Remove reasoning preamble if present
    if text.lstrip().startswith("Thinking"):
        lines = text.split('\n')
        for i, line in enumerate(lines):
            s = line.strip()
            if any(s.startswith(p) for p in ['```', 'def ', 'class ', 'function ', 'import ', '#', '//']):
                return '\n'.join(lines[i:])

    return text


# ─── Tool Call Parsing ───────────────────────────────────────────────────────

def parse_tool_calls(text):
    """Parse tool calls from model output.

    Handles:
      - Qwen XML: <function=Name><parameter=key>value</parameter></function>
      - Qwen JSON tags: <tool_call>{"name":"...","arguments":{...}}</tool_call>
      - Bare JSON function-call objects.
    """
    tool_calls = []

    # 1) Prefer XML function style used by Qwen chat templates.
    for func_match in re.finditer(r'<function=([A-Za-z0-9_\-]+)>\s*(.*?)\s*</function>', text, re.DOTALL):
        name = func_match.group(1)
        params_block = func_match.group(2)
        args = {}
        for param_match in re.finditer(r'<parameter=([A-Za-z0-9_\-]+)>\s*(.*?)\s*</parameter>', params_block, re.DOTALL):
            key = param_match.group(1)
            raw_value = param_match.group(2).strip()
            try:
                args[key] = json.loads(raw_value)
            except (json.JSONDecodeError, ValueError):
                args[key] = raw_value
        tool_calls.append({
            "type": "tool_use",
            "id": f"toolu_{uuid.uuid4().hex[:20]}",
            "name": name,
            "input": args,
        })

    if tool_calls:
        log(f"  Parsed {len(tool_calls)} XML tool call(s)")
        return tool_calls

    # 2) Fallback to <tool_call>...</tool_call> JSON style.
    for json_block_match in re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL):
        raw = json_block_match.group(1)

        # Some models wrap XML function calls inside <tool_call>...</tool_call>.
        nested_xml_calls = []
        for func_match in re.finditer(r'<function\s*=\s*([^\s>]+)\s*>\s*(.*?)\s*</function>', raw, re.DOTALL):
            name = func_match.group(1)
            params_block = func_match.group(2)
            args = {}
            for param_match in re.finditer(r'<parameter\s*=\s*([^\s>]+)\s*>\s*(.*?)\s*</parameter>', params_block, re.DOTALL):
                key = param_match.group(1)
                raw_value = param_match.group(2).strip()
                try:
                    args[key] = json.loads(raw_value)
                except (json.JSONDecodeError, ValueError):
                    args[key] = raw_value
            nested_xml_calls.append({
                "type": "tool_use",
                "id": f"toolu_{uuid.uuid4().hex[:20]}",
                "name": name,
                "input": args,
            })
        if nested_xml_calls:
            tool_calls.extend(nested_xml_calls)
            continue

        try:
            data = json.loads(raw)
            name = data.get("name", "")
            if not name:
                log(f"  Warning: skipping tool call with missing name: {raw[:80]}")
                continue
            tool_calls.append({
                "type": "tool_use",
                "id": f"toolu_{uuid.uuid4().hex[:20]}",
                "name": name,
                "input": data.get("arguments", data.get("parameters", {})),
            })
        except (json.JSONDecodeError, ValueError) as e:
            log(f"  Warning: failed to parse tool call JSON ({e}): {raw[:80]}")

    if tool_calls:
        log(f"  Parsed {len(tool_calls)} tagged tool call(s)")
        return tool_calls

    # 3) Final fallback: bare JSON call object in plain text.
    for bare_match in re.finditer(
        r'\{"(?:name|function)"\s*:\s*"[^"]+"\s*,\s*"(?:arguments|parameters)"\s*:\s*\{.*?\}\s*\}',
        text,
        re.DOTALL,
    ):
        raw = bare_match.group(0)
        try:
            data = json.loads(raw)
            name = data.get("name") or data.get("function", "")
            if not name:
                continue
            args = data.get("arguments") or data.get("parameters", {})
            if isinstance(args, str):
                args = json.loads(args)
            tool_calls.append({
                "type": "tool_use",
                "id": f"toolu_{uuid.uuid4().hex[:20]}",
                "name": name,
                "input": args if isinstance(args, dict) else {},
            })
        except (json.JSONDecodeError, ValueError, TypeError):
            log(f"  Warning: failed to parse bare tool call JSON: {raw[:80]}")

    if tool_calls:
        log(f"  Parsed {len(tool_calls)} bare JSON tool call(s)")

    return tool_calls


def extract_text_and_tools(raw_text):
    """Return (clean_text, tool_calls) from raw model output.

    Tool calls are extracted from the raw text *before* think-tag stripping so
    that <tool_call> blocks placed inside a second <think> segment by Qwen3.5
    (a common pattern: think → plan text → think-with-tool-call) are not
    silently discarded by the re.sub that removes <think>…</think>.
    """
    # Parse from raw FIRST so tool calls inside any <think> block are captured.
    tool_calls = parse_tool_calls(raw_text)
    # Clean the visible text: strip think tags, remove tool call markup, etc.
    text = clean_response(raw_text)
    clean_text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
    clean_text = re.sub(r'<function=[A-Za-z0-9_\-]+>\s*.*?\s*</function>', '', clean_text, flags=re.DOTALL)
    clean_text = clean_text.strip()
    return clean_text, tool_calls


# ─── Anthropic → Chat-template Message Conversion ───────────────────────────

def convert_tools_to_openai(anthropic_tools):
    """Convert Anthropic tool definitions to the OpenAI/Qwen function-calling format."""
    result = []
    for tool in anthropic_tools:
        result.append({
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            },
        })
    return result


def convert_messages(body):
    """
    Convert Anthropic Messages format to chat messages suitable for
    tokenizer.apply_chat_template().

    Handles:
      - text blocks
      - tool_use blocks (assistant → tool_calls list)
      - tool_result blocks (user → role:"tool" messages)
    """
    messages = []

    # System prompt
    if body.get("system"):
        sys_text = body["system"]
        if isinstance(sys_text, list):
            sys_text = "\n".join(b.get("text", "") for b in sys_text if b.get("type") == "text")
        messages.append({"role": "system", "content": sys_text})

    # Conversation turns
    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if not isinstance(content, list):
            messages.append({"role": role, "content": content})
            continue

        text_blocks = [b for b in content if b.get("type") == "text"]
        tool_use_blocks = [b for b in content if b.get("type") == "tool_use"]
        tool_result_blocks = [b for b in content if b.get("type") == "tool_result"]

        if tool_result_blocks:
            # User message carrying tool results back to the model.
            # Emit any accompanying text first, then one "tool" message per result.
            text_parts = [b.get("text", "") for b in text_blocks if b.get("text")]
            if text_parts:
                messages.append({"role": "user", "content": "\n".join(text_parts)})
            for block in tool_result_blocks:
                tc = block.get("content", "")
                if isinstance(tc, list):
                    tc = "\n".join(b.get("text", str(b)) for b in tc)
                result_text = str(tc)
                if block.get("is_error"):
                    result_text = f"Error: {result_text}"
                messages.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id", ""),
                    "content": result_text,
                })

        elif tool_use_blocks and role == "assistant":
            # Assistant message where the model chose to call tools.
            tool_calls = [
                {
                    "id": b.get("id", f"toolu_{uuid.uuid4().hex[:20]}"),
                    "type": "function",
                    "function": {
                        "name": b.get("name", ""),
                        # Chat templates for Qwen/OpenAI-style tool calls expect a
                        # mapping here, not a JSON string.
                        "arguments": b.get("input", {}) if isinstance(b.get("input", {}), dict) else {},
                    },
                }
                for b in tool_use_blocks
            ]
            text_content = "\n".join(b.get("text", "") for b in text_blocks if b.get("text"))
            msg_dict = {"role": "assistant", "tool_calls": tool_calls}
            if text_content:
                msg_dict["content"] = text_content
            messages.append(msg_dict)

        else:
            # Plain text message
            parts = [b.get("text", "") for b in text_blocks]
            messages.append({"role": role, "content": "\n".join(p for p in parts if p)})

    return messages


def tokenize_messages(messages, tools=None):
    """Apply the model's chat template and return token ids.

    When tools are provided we attempt kwargs in decreasing order of preference:
      1. tools + enable_thinking=False  — Qwen3 family: disabling thinking mode
         forces the model to emit <tool_call> directly instead of writing a
         natural-language plan inside <think>…</think> and then stopping.
      2. tools only                     — models that accept tools but not
         enable_thinking (e.g. Qwen2.5, other OpenAI-compatible templates).
      3. no tools                       — last-resort; model loses tool context
         but at least generates a response.
    """
    base_kwargs = {"add_generation_prompt": True, "tokenize": True}

    # Build the ordered list of kwargs to try
    attempts = []
    if tools:
        attempts.append({**base_kwargs, "tools": tools, "enable_thinking": False})
        attempts.append({**base_kwargs, "tools": tools})
    attempts.append(base_kwargs)

    for i, kwargs in enumerate(attempts):
        try:
            return tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError as e:
            if i < len(attempts) - 1:
                log(f"  Warning: chat template rejected kwargs ({e}), trying simpler config")
                continue
            log(f"  Warning: chat template failed ({e}), falling back to plain text")
            break
        except Exception as e:
            log(f"  Warning: chat template failed ({e}), falling back to plain text")
            break

    # Last-resort plain-text fallback
    text = "\n".join(f"{m['role']}: {m.get('content', '')}" for m in messages)
    text += "\nassistant: "
    return tokenizer.encode(text)


# ─── Core Inference ──────────────────────────────────────────────────────────

def build_gen_kwargs(body):
    """Build MLX stream_generate keyword arguments from the request body."""
    gen_kwargs = {"prefill_step_size": PREFILL_SIZE}
    if KV_BITS:
        gen_kwargs["kv_bits"] = KV_BITS
        gen_kwargs["kv_group_size"] = 64
        gen_kwargs["quantized_kv_start"] = 0
    gen_kwargs["sampler"] = make_sampler(temp=body.get("temperature", 0.7))
    return gen_kwargs


def run_generation(body):
    """
    Run MLX inference to completion.

    Returns (full_text, gen_tokens, prompt_tokens, finish_reason).
    """
    messages = convert_messages(body)
    tools = convert_tools_to_openai(body.get("tools")) if body.get("tools") else None
    token_ids = tokenize_messages(messages, tools=tools)
    prompt_tokens = len(token_ids)
    log(f"  Prompt: {prompt_tokens} tokens")

    gen_kwargs = build_gen_kwargs(body)
    full_text = ""
    gen_tokens = 0
    finish_reason = "end_turn"
    t0 = time.time()

    with generate_lock:
        for response in stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=token_ids,
            max_tokens=body.get("max_tokens", DEFAULT_MAX_TOKENS),
            **gen_kwargs,
        ):
            full_text += response.text
            gen_tokens = response.generation_tokens
            if response.finish_reason == "length":
                finish_reason = "max_tokens"
            elif response.finish_reason == "stop":
                finish_reason = "end_turn"

    elapsed = time.time() - t0
    tps = gen_tokens / elapsed if elapsed > 0 else 0
    log(f"  Generated: {gen_tokens} tokens in {elapsed:.1f}s ({tps:.1f} tok/s)")
    log(f"  Raw output ({len(full_text)} chars): {repr(full_text[:500])}")
    return full_text, gen_tokens, prompt_tokens, finish_reason


# ─── Non-streaming Response ──────────────────────────────────────────────────

def generate_response(body):
    """Run inference and return a complete Anthropic Messages response object."""
    full_text, gen_tokens, prompt_tokens, finish_reason = run_generation(body)
    clean_text, tool_calls = extract_text_and_tools(full_text)

    content_blocks = []
    if clean_text:
        content_blocks.append({"type": "text", "text": clean_text})
    content_blocks.extend(tool_calls)
    if not content_blocks:
        content_blocks.append({"type": "text", "text": "(No output)"})

    if tool_calls:
        finish_reason = "tool_use"

    preview = content_blocks[0].get("text") or content_blocks[0].get("name", "")
    log(f"  ← OK ({gen_tokens} tok) {preview[:80]}...")

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": body.get("model", "claude-sonnet-4-6"),
        "content": content_blocks,
        "stop_reason": finish_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": gen_tokens,
        },
    }


# ─── Streaming Response (SSE) ────────────────────────────────────────────────

def send_sse(handler, event_type, data):
    """Write one SSE event and flush immediately."""
    handler.wfile.write(f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode())
    handler.wfile.flush()


def generate_response_stream(handler, body):
    """
    Run inference, then emit the full Anthropic SSE event stream.

    Generation is buffered so that tool call XML never leaks into the
    text stream.  Events are emitted in the correct Anthropic order:
      message_start → ping → content_block_* → message_delta → message_stop
    """
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    model_name = body.get("model", "claude-sonnet-4-6")

    full_text, gen_tokens, prompt_tokens, finish_reason = run_generation(body)
    clean_text, tool_calls = extract_text_and_tools(full_text)

    if tool_calls:
        finish_reason = "tool_use"

    # ── message_start ──────────────────────────────────────────────────
    send_sse(handler, "message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": model_name,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": prompt_tokens, "output_tokens": 0},
        },
    })
    send_sse(handler, "ping", {"type": "ping"})

    block_index = 0

    # ── text block ─────────────────────────────────────────────────────
    if clean_text:
        text_to_emit = clean_text
    elif tool_calls:
        # tool-only response; no visible text
        text_to_emit = ""
    else:
        text_to_emit = "(No output)"
    send_sse(handler, "content_block_start", {
        "type": "content_block_start",
        "index": block_index,
        "content_block": {"type": "text", "text": ""},
    })
    send_sse(handler, "content_block_delta", {
        "type": "content_block_delta",
        "index": block_index,
        "delta": {"type": "text_delta", "text": text_to_emit},
    })
    send_sse(handler, "content_block_stop", {
        "type": "content_block_stop",
        "index": block_index,
    })
    block_index += 1

    # ── tool_use blocks ────────────────────────────────────────────────
    for tc in tool_calls:
        send_sse(handler, "content_block_start", {
            "type": "content_block_start",
            "index": block_index,
            "content_block": {
                "type": "tool_use",
                "id": tc["id"],
                "name": tc["name"],
                "input": {},
            },
        })
        send_sse(handler, "content_block_delta", {
            "type": "content_block_delta",
            "index": block_index,
            "delta": {
                "type": "input_json_delta",
                "partial_json": json.dumps(tc["input"]),
            },
        })
        send_sse(handler, "content_block_stop", {
            "type": "content_block_stop",
            "index": block_index,
        })
        block_index += 1

    # ── message_delta + message_stop ───────────────────────────────────
    send_sse(handler, "message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": finish_reason, "stop_sequence": None},
        "usage": {"output_tokens": gen_tokens},
    })
    send_sse(handler, "message_stop", {"type": "message_stop"})

    preview = (clean_text or (tool_calls[0]["name"] if tool_calls else "(empty)"))[:80]
    log(f"  ← OK STREAM ({gen_tokens} tok) {preview}...")


# ─── HTTP Handler ────────────────────────────────────────────────────────────

def send_json(handler, status, data):
    resp = json.dumps(data).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", len(resp))
    handler.end_headers()
    handler.wfile.write(resp)


def get_path(full_path):
    return urlparse(full_path).path


class AnthropicHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_HEAD(self):
        log(f"HEAD {self.path}")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

    def do_POST(self):
        path = get_path(self.path)
        content_length = int(self.headers.get('Content-Length', 0))
        raw = self.rfile.read(content_length) if content_length else b'{}'
        body = json.loads(raw)
        streaming = body.get("stream", False)
        log(f"POST {self.path} model={body.get('model','-')} "
            f"max_tokens={body.get('max_tokens','-')} stream={streaming} "
            f"tools={len(body.get('tools') or [])}")

        if path in ("/v1/messages", "/messages"):
            if streaming:
                try:
                    # Close connection after SSE events so HTTP/1.1 clients
                    # don't hang waiting for more data after message_stop.
                    self.close_connection = True
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    generate_response_stream(self, body)
                except Exception as e:
                    log(f"  ← STREAM ERROR: {e}")
                    import traceback
                    traceback.print_exc(file=sys.stderr)
            else:
                try:
                    result = generate_response(body)
                    send_json(self, 200, result)
                except Exception as e:
                    log(f"  ← ERROR: {e}")
                    import traceback
                    traceback.print_exc(file=sys.stderr)
                    send_json(self, 500, {"error": {"type": "server_error", "message": str(e)}})
        else:
            log(f"  Unknown POST: {path}")
            send_json(self, 200, {})

    def do_GET(self):
        path = get_path(self.path)
        log(f"GET {self.path}")

        if path in ("/v1/models", "/models"):
            send_json(self, 200, {
                "object": "list",
                "data": [
                    {"id": "claude-opus-4-6", "object": "model", "created": int(time.time()), "owned_by": "local"},
                    {"id": "claude-sonnet-4-6", "object": "model", "created": int(time.time()), "owned_by": "local"},
                    {"id": "claude-haiku-4-5-20251001", "object": "model", "created": int(time.time()), "owned_by": "local"},
                ],
            })
        elif path == "/health":
            send_json(self, 200, {"status": "ok", "model": MODEL_PATH})
        else:
            send_json(self, 200, {})


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════╗")
    print("║  MLX Native Anthropic Server                    ║")
    print("║  Claude Code → MLX → Apple Silicon (direct)     ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    load_model()

    print()
    print(f"Serving Anthropic Messages API on http://localhost:{PORT}")
    print(f"Model: {MODEL_PATH}")
    print(f"KV cache: {KV_BITS}-bit quantization" if KV_BITS else "KV cache: full precision")
    print()
    print("Claude Code config:")
    print(f"  ANTHROPIC_BASE_URL=http://localhost:{PORT}")
    print(f"  ANTHROPIC_API_KEY=sk-local")
    print()

    server = HTTPServer(("127.0.0.1", PORT), AnthropicHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()
