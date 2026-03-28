"""
Optional integration with LCME (Local Cognitive Memory Engine).

  https://github.com/gschaidergabriel/lcme

LCME is a separate stack (PyTorch + sentence-transformers + SQLite/vectors).
Enable only when installed and LCME_ENABLED=1.

See requirements-lcme-optional.txt in the repo root.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional

_lcme_lock = threading.Lock()
_lcme_instance: Any = None
_lcme_import_failed = False


def _log(msg: str) -> None:
    import sys
    import time

    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def _get_lcme():
    """Lazy singleton; None if disabled, missing, or import failed."""
    global _lcme_instance, _lcme_import_failed

    enabled = os.environ.get("LCME_ENABLED", "0").lower() in ("1", "true", "yes")
    if not enabled:
        return None
    if _lcme_import_failed:
        return None
    if _lcme_instance is not None:
        return _lcme_instance

    with _lcme_lock:
        if _lcme_import_failed:
            return None
        if _lcme_instance is not None:
            return _lcme_instance
        try:
            from lcme import LCME, LCMEConfig
        except ImportError as e:
            _lcme_import_failed = True
            _log(f"  LCME_ENABLED=1 but lcme not importable: {e}")
            return None

        data_dir = os.environ.get("LCME_DATA_DIR", "~/.lcme/data")
        vector_model = os.environ.get("LCME_VECTOR_MODEL", "all-MiniLM-L6-v2")
        cfg = LCMEConfig(data_dir=data_dir, vector_model=vector_model)
        _lcme_instance = LCME(cfg)
        _log(f"  LCME initialized (data_dir={cfg.resolved_data_dir()})")
        return _lcme_instance


def _last_user_text_snippet(body: Dict[str, Any], max_chars: int = 8000) -> str:
    """Best-effort text from the latest user message for retrieval query."""
    messages: List[Dict[str, Any]] = body.get("messages") or []
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            text = "\n".join(p for p in parts if p)
        else:
            text = str(content)
        text = text.strip()
        if not text:
            continue
        if len(text) > max_chars:
            text = text[:max_chars]
        return text
    return ""


def _merge_system_block(body: Dict[str, Any], block: str) -> None:
    prefix = (
        "\n\n### Long-term memory (LCME)\n"
        "Prior memory snippets below may be useful; they are not guaranteed facts.\n\n"
    )
    addition = prefix + block
    sys_val = body.get("system")
    if not sys_val:
        body["system"] = addition.lstrip()
    elif isinstance(sys_val, str):
        body["system"] = sys_val + addition
    elif isinstance(sys_val, list):
        body["system"] = list(sys_val)
        body["system"].append({"type": "text", "text": addition.lstrip()})


def maybe_enrich_request_with_lcme(body: Dict[str, Any]) -> None:
    """
    If LCME is enabled and available, retrieve context for the latest user
    turn and append it to the system prompt. Mutates body in place.
    """
    mem = _get_lcme()
    if mem is None:
        return

    query = _last_user_text_snippet(body)
    if not query:
        return

    try:
        ctx = mem.get_context_string(query)
    except Exception as e:
        _log(f"  LCME retrieve failed: {e}")
        return

    if not (ctx and str(ctx).strip()):
        return

    limit = int(os.environ.get("LCME_CONTEXT_MAX_CHARS", "3500"))
    ctx_str = str(ctx).strip()
    if len(ctx_str) > limit:
        ctx_str = ctx_str[: limit - 3] + "..."

    _merge_system_block(body, ctx_str)
    _log(f"  LCME injected ~{len(ctx_str)} chars into system")

    if os.environ.get("LCME_INGEST_USER", "0").lower() in ("1", "true", "yes"):
        try:
            mem.ingest(query, origin="user")
        except Exception as e:
            _log(f"  LCME ingest failed: {e}")
