"""
cogito-ergo — two-stage memory retrieval for AI agents.

Layers (increasing capability, increasing cost):
  Snapshot        — compressed markdown index (~741 tokens), ``cogito snapshot``
  recall_b        — zero-LLM sub-query decomposition + RRF (127ms)
  recall          — integer-pointer LLM filter (~1300ms)
  recall_hybrid   — BM25 + dense + RRF + tiered LLM escalation (93.4% R@1
                    on LongMemEval_S; opt-in, needs ``bm25s`` for best
                    results)

Key innovation: the filter LLM outputs only integer indices (e.g. [3, 7, 12]) —
never memory text — so it cannot corrupt or hallucinate into the content
returned to the main agent. Fidelity is structural, not a prompting convention.
"""

__version__ = "0.3.0"

from cogito.recall import recall  # noqa: F401

# Note: avoid shadowing the ``cogito.recall_hybrid`` submodule. Users who want
# the function directly can do ``from cogito.recall_hybrid import recall_hybrid``.
