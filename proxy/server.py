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
    """Remove <think>...</think> blocks from Qwen's reasoning output."""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
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
    """Parse tool calls from Qwen's <tool_call>...</tool_call> output format."""
    tool_calls = []
    for match in re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL):
        try:
            data = json.loads(match.group(1))
            name = data.get("name", "")
            if not name:
                log("  Warning: skipping tool call with missing or empty name")
                continue
            tool_calls.append({
                "type": "tool_use",
                "id": f"toolu_{uuid.uuid4().hex[:20]}",
                "name": name,
                "input": data.get("arguments", data.get("parameters", {})),
            })
        except (json.JSONDecodeError, ValueError):
            pass
    return tool_calls


def extract_text_and_tools(raw_text):
    """Return (clean_text, tool_calls) from raw model output."""
    text = clean_response(raw_text)
    tool_calls = parse_tool_calls(text)
    # Strip tool call XML from the visible text
    clean_text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL).strip()
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
                        "arguments": json.dumps(b.get("input", {})),
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
    """Apply the model's chat template and return token ids."""
    kwargs = {"add_generation_prompt": True, "tokenize": True}
    if tools:
        kwargs["tools"] = tools
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError as e:
        # Tokenizer template doesn't accept 'tools' — retry without it
        log(f"  Warning: chat template rejected tools param ({e}), retrying without tools")
        try:
            return tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True
            )
        except Exception as e2:
            log(f"  Warning: chat template failed ({e2}), falling back to plain text")
    except Exception as e:
        log(f"  Warning: chat template failed ({e}), falling back to plain text")

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
    tools = convert_tools_to_openai(body["tools"]) if body.get("tools") else None
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
        text_to_emit = ""          # tool-only response; no visible text
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
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
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
