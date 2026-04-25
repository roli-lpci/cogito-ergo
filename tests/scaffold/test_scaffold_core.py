"""Tests for fidelis_scaffold core. Pure-Python, no external deps."""

from __future__ import annotations



from fidelis.scaffold import (  # noqa: E402
    SCAFFOLD_CLOSE,
    SCAFFOLD_OPEN,
    SCAFFOLD_VERSION,
    is_scaffolded,
    preflight,
    preflight_or_raise,
    strip_scaffold,
    wrap_idempotent,
    wrap_system_prompt,
)

QTYPES = [
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "knowledge-update",
    "multi-session",
    "temporal-reasoning",
]


def test_version_is_set():
    assert SCAFFOLD_VERSION == "v0.1.0"
    assert SCAFFOLD_OPEN == "[FIDELIS-SCAFFOLD-v0.1.0]"
    assert SCAFFOLD_CLOSE == "[/FIDELIS-SCAFFOLD-v0.1.0]"


def test_all_qtypes_produce_scaffolded_output():
    for qt in QTYPES:
        s = wrap_system_prompt(qt)
        assert SCAFFOLD_OPEN in s
        assert SCAFFOLD_CLOSE in s
        assert is_scaffolded(s)


def test_unknown_qtype_falls_through_to_generic():
    s = wrap_system_prompt("not-a-real-qtype")
    assert is_scaffolded(s)
    assert "Quote the relevant passage" in s


def test_top_score_renders_confidence_marker():
    s_high = wrap_system_prompt("single-session-user", top_score=0.85)
    assert "HIGH" in s_high
    s_med = wrap_system_prompt("single-session-user", top_score=0.6)
    assert "MEDIUM" in s_med
    s_low = wrap_system_prompt("single-session-user", top_score=0.3)
    assert "LOW" in s_low
    s_none = wrap_system_prompt("single-session-user", top_score=None)
    assert "unknown" in s_none.lower()


def test_idempotency():
    s1 = wrap_system_prompt("temporal-reasoning", top_score=0.7)
    s2 = wrap_idempotent("temporal-reasoning", top_score=0.7, prior=s1)
    s3 = wrap_idempotent("temporal-reasoning", top_score=0.7, prior=s2)
    # marker count must stay 1 OPEN + 1 CLOSE regardless of how many times wrap is applied
    for s in (s2, s3):
        assert s.count("[FIDELIS-SCAFFOLD-") == 1
        assert s.count("[/FIDELIS-SCAFFOLD-") == 1
    # multiple wraps should be stable
    assert s2 == s3


def test_strip_scaffold_removes_all_markers():
    s = wrap_system_prompt("multi-session")
    stripped = strip_scaffold(s)
    assert not is_scaffolded(stripped)
    # Stripping unwrapped text is idempotent
    assert strip_scaffold(stripped) == stripped


def test_preflight_passes_on_default_scaffolds():
    for qt in QTYPES:
        s = wrap_system_prompt(qt, top_score=0.7)
        rep = preflight(s)
        assert rep.passed, f"{qt} failed preflight: {rep.failures}"


def test_preflight_fails_on_forbidden_tokens():
    bad = wrap_system_prompt("single-session-user") + "<|endoftext|>"
    rep = preflight(bad)
    assert not rep.passed
    assert any("forbidden control token" in f for f in rep.failures)


def test_preflight_fails_on_unbalanced_fences():
    bad = wrap_system_prompt("single-session-user") + "\n```python\nprint('oops')"
    rep = preflight(bad)
    assert not rep.passed


def test_preflight_fails_on_nested_markers():
    s = wrap_system_prompt("single-session-user")
    bad = s + "\n" + s  # double wrap raw, not via wrap_idempotent
    rep = preflight(bad)
    assert not rep.passed


def test_preflight_or_raise_raises_on_failure():
    bad = "<|im_start|>"
    try:
        preflight_or_raise(bad)
    except RuntimeError:
        pass
    else:
        raise AssertionError("preflight_or_raise should have raised")


def test_token_budget_under_200():
    for qt in QTYPES:
        s = wrap_system_prompt(qt, top_score=0.5)
        rep = preflight(s)
        assert rep.metrics["approx_token_count"] <= 200, (
            f"{qt} exceeds budget: {rep.metrics['approx_token_count']}")
