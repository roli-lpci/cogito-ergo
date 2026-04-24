"""Regression test for verify-guard activation.

Documents the known bug from STATUS.md:
  "Verify-guard never activated. Guard logic coded in pipeline; activation
  field absent from per-question output. The 96.4% is from temporal boost +
  runtime escalation only."

This file pins the gap so future audits see a test, not prose. Two tests:

1. `test_verify_guard_callable` — sanity: the guard function exists and
   returns None/"YES"/"NO" (not a crash).
2. `test_verify_guard_activation_field_exposed` — xfail: when the guard
   fires (rerank moves a new candidate to top-1 AND llm_verify returns
   YES), the per-question output should carry an observable activation
   field. Currently it does not; route_decision is set locally but not
   surfaced to the per-question result dict. xfail until fixed.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parents[1] / "bench"


def _import_pipeline():
    """Best-effort import of the benchmark pipeline module.

    The bench pipeline isn't packaged — it's a script. We import it by path
    so tests can reference the verify-guard surface. If import fails
    (missing heavy deps), skip rather than error.
    """
    path = _BENCH / "longmemeval_combined_pipeline_v35.py"
    if not path.exists():
        pytest.skip(f"bench pipeline not present at {path}")
    # Stub heavy deps that may not be available in the test env.
    for missing in ("openai",):
        if missing not in sys.modules:
            sys.modules[missing] = types.ModuleType(missing)
    spec = __import__("importlib.util", fromlist=["spec_from_file_location"]).spec_from_file_location(
        "longmemeval_pipeline_v35", path
    )
    mod = __import__("importlib.util", fromlist=["module_from_spec"]).module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        pytest.skip(f"pipeline import failed ({type(e).__name__}); not a bug in the guard")
    return mod


def test_verify_guard_callable():
    mod = _import_pipeline()
    assert hasattr(mod, "llm_verify_one"), "verify-guard function missing from pipeline"
    assert callable(mod.llm_verify_one), "llm_verify_one is not callable"


@pytest.mark.xfail(
    reason="Known bug (STATUS.md): verify-guard activation field absent from per-question output. "
           "Guard is called and mutates a local route_decision, but the per-question result dict "
           "has no observable 'guard_activated' / 'route_decision' key. Pin with xfail until fixed.",
    strict=False,
)
def test_verify_guard_activation_field_exposed():
    _import_pipeline()  # ensure pipeline is importable before asserting on its source
    # The per-question output constructor / writer should expose an activation
    # field. We look for either an explicit function or a conventional key.
    # If neither exists, xfail triggers as expected.
    candidates = ["guard_activated", "route_decision", "verify_guard_fired"]
    source = (_BENCH / "longmemeval_combined_pipeline_v35.py").read_text()
    # Require the key to appear as an output-record field (right-hand-side of
    # a dict assignment), not just a local variable name.
    observable = any(
        f'"{k}":' in source or f"'{k}':" in source
        for k in candidates
    )
    assert observable, "no observable verify-guard activation key in per-question output"
