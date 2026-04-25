"""fidelis.scaffold — drift-safe QA system-prompt wrapper for retrieved memory.

The Fidelis QA scaffold technique: a versioned, bounded, idempotent system-prompt
wrapper that lifts QA accuracy on retrieval-augmented questions without modifying
the underlying LLM. Composes with Fidelis retrieval (BM25 + temporal-boost +
runtime-escalate, R@1 = 83.2% zero-LLM at $0/q) and the user's existing LLM.

Public API:
    wrap_system_prompt(qtype, top_score=None) -> str
    wrap_idempotent(qtype, top_score=None, prior="") -> str
    is_scaffolded(text) -> bool
    strip_scaffold(text) -> str
    preflight(text, max_tokens=200) -> PreflightReport
    preflight_or_raise(text, max_tokens=200) -> None
    SCAFFOLD_VERSION = "v0.1.0"

Validated on LongMemEval-S smoke (n=60 stratified) producing per-qtype lifts
of +20pp on knowledge-update, +16pp on preference, +8pp on multi-session over
a minimal-prompt Opus baseline. See experiments/zeroLLM-FLAGSHIP/ for the
evidence chain.
"""

from .preflight import PreflightReport, preflight, preflight_or_raise
from ._core import (
    SCAFFOLD_CLOSE,
    SCAFFOLD_OPEN,
    SCAFFOLD_VERSION,
    is_scaffolded,
    strip_scaffold,
    wrap_idempotent,
    wrap_system_prompt,
)

__all__ = [
    "SCAFFOLD_VERSION",
    "SCAFFOLD_OPEN",
    "SCAFFOLD_CLOSE",
    "wrap_system_prompt",
    "wrap_idempotent",
    "is_scaffolded",
    "strip_scaffold",
    "preflight",
    "preflight_or_raise",
    "PreflightReport",
]
