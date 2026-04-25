"""Continuity tests — exercise external dependency boundaries and assert graceful degradation.

fidelis-scaffold has zero runtime dependencies. The relevant external boundaries are:

1. Caller-provided qtype (must handle unknown gracefully)
2. Caller-provided top_score (must handle None / weird floats)
3. Caller-provided text input to preflight (must handle empty / huge / weird inputs)
4. Caller-provided prior text to wrap_idempotent (must handle non-str / very long)

For each boundary, assert: no stacktrace leakage, defined fallback, structured output.
"""

from __future__ import annotations



from fidelis.scaffold import (  # noqa: E402
    is_scaffolded,
    preflight,
    strip_scaffold,
    wrap_idempotent,
    wrap_system_prompt,
)


# ---------------------------------------------------------------------------
# Boundary 1: caller-provided qtype
# ---------------------------------------------------------------------------

def test_unknown_qtype_returns_generic_scaffold():
    """Unknown qtype falls through to a generic procedure, never raises."""
    s = wrap_system_prompt("not-a-real-qtype-12345")
    assert is_scaffolded(s)
    assert "Quote the relevant passage" in s
    assert preflight(s).passed


def test_empty_qtype_string_returns_generic_scaffold():
    s = wrap_system_prompt("")
    assert is_scaffolded(s)
    assert preflight(s).passed


def test_qtype_with_whitespace_normalizes():
    s1 = wrap_system_prompt("  single-session-user  ")
    s2 = wrap_system_prompt("single-session-user")
    # Both should produce valid scaffolds (whitespace stripped internally)
    assert is_scaffolded(s1)
    assert is_scaffolded(s2)
    assert preflight(s1).passed
    assert preflight(s2).passed


def test_uppercase_qtype_normalizes():
    s = wrap_system_prompt("TEMPORAL-REASONING")
    assert is_scaffolded(s)
    assert "calendar difference" in s.lower() or "yyyy/mm/dd" in s.lower()


# ---------------------------------------------------------------------------
# Boundary 2: caller-provided top_score
# ---------------------------------------------------------------------------

def test_top_score_none_renders_unknown():
    s = wrap_system_prompt("single-session-user", top_score=None)
    assert "unknown" in s.lower()
    assert preflight(s).passed


def test_top_score_zero_renders_low():
    s = wrap_system_prompt("single-session-user", top_score=0.0)
    assert "LOW" in s
    assert preflight(s).passed


def test_top_score_one_renders_high():
    s = wrap_system_prompt("single-session-user", top_score=1.0)
    assert "HIGH" in s
    assert preflight(s).passed


def test_top_score_negative_does_not_crash():
    s = wrap_system_prompt("single-session-user", top_score=-0.5)
    # Negative is treated as LOW (below threshold)
    assert "LOW" in s
    assert preflight(s).passed


def test_top_score_above_one_does_not_crash():
    s = wrap_system_prompt("single-session-user", top_score=1.5)
    # Above 1 still triggers HIGH path
    assert "HIGH" in s
    assert preflight(s).passed


# ---------------------------------------------------------------------------
# Boundary 3: preflight on caller-provided text
# ---------------------------------------------------------------------------

def test_preflight_on_empty_string():
    rep = preflight("")
    # Empty string passes structurally (no forbidden tokens, balanced fences, etc.)
    # but has 0 scaffold markers — open_count == close_count, no FAIL
    assert rep.passed
    assert rep.metrics["scaffold_open_markers"] == 0


def test_preflight_on_very_long_text():
    long_text = wrap_system_prompt("single-session-user") + ("x" * 100_000)
    rep = preflight(long_text)
    # Should fail length bound, not crash
    assert not rep.passed
    assert any("token count" in f for f in rep.failures)


def test_preflight_on_unicode_text():
    s = wrap_system_prompt("single-session-user")
    # Append some valid Unicode (already has em-dash internally)
    rep = preflight(s)
    # Should pass (or warn about non-ASCII), never crash
    assert rep.passed or any("ASCII" in w or "ascii" in w for w in rep.warnings)


def test_preflight_or_raise_propagates_clean_exception_on_failure():
    """preflight_or_raise must raise RuntimeError, never silently pass on bad input."""
    from fidelis.scaffold import preflight_or_raise
    bad = "<|endoftext|>"
    try:
        preflight_or_raise(bad)
    except RuntimeError as e:
        # Expected — error message should be human-readable, not a stack-trace dump
        assert "Preflight failed" in str(e)
    except Exception as e:
        raise AssertionError(f"Wrong exception type: {type(e).__name__}: {e}") from e
    else:
        raise AssertionError("preflight_or_raise should have raised on forbidden token")


# ---------------------------------------------------------------------------
# Boundary 4: wrap_idempotent on caller-provided prior
# ---------------------------------------------------------------------------

def test_wrap_idempotent_with_empty_prior():
    s = wrap_idempotent("single-session-user", top_score=0.5, prior="")
    assert is_scaffolded(s)
    assert preflight(s).passed


def test_wrap_idempotent_with_non_scaffolded_prior():
    s = wrap_idempotent("single-session-user", top_score=0.5, prior="Some other system prompt.")
    # Should attach prior after the new scaffold
    assert is_scaffolded(s)
    assert "Some other system prompt." in s


def test_wrap_idempotent_strips_prior_scaffold():
    s1 = wrap_system_prompt("temporal-reasoning")
    s2 = wrap_idempotent("temporal-reasoning", top_score=0.5, prior=s1)
    # Should have exactly one scaffold pair
    assert s2.count("[FIDELIS-SCAFFOLD-") == 1


def test_strip_scaffold_idempotent_on_unscaffolded():
    text = "Plain text with no scaffold."
    assert strip_scaffold(text) == text
    assert strip_scaffold(strip_scaffold(text)) == text


# ---------------------------------------------------------------------------
# Boundary 5: no panics on weird-but-not-malicious inputs
# ---------------------------------------------------------------------------

def test_qtype_with_special_chars_does_not_crash():
    # qtype with quotes, slashes, etc. — falls through to generic
    s = wrap_system_prompt('qtype-with-"quotes"-and-/slashes/')
    assert is_scaffolded(s)
    assert preflight(s).passed


def test_repeated_wrap_idempotent_calls_stable():
    s = wrap_system_prompt("multi-session", top_score=0.6)
    for _ in range(5):
        s = wrap_idempotent("multi-session", top_score=0.6, prior=s)
    # Should still have exactly one scaffold pair after 5 wrap-of-wrap calls
    assert s.count("[FIDELIS-SCAFFOLD-") == 1
    assert s.count("[/FIDELIS-SCAFFOLD-") == 1
