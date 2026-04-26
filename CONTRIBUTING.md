# Contributing

## Running tests

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests in `tests/` are server-free — no Ollama or ChromaDB required. Mark any test that needs a live server with `pytest.mark.skip(reason="requires server")`.

## Adding bench cases

Bench cases live in `bench/`. Each case is a JSON file with `query`, `expected_ids`, and optional `notes`. Run the combined eval with:

```bash
python bench/eval.py
```

Add cases that cover real retrieval failures or regressions. Include the memory seed data in `bench/seeds/` if needed.

## Code style

- Line length: 100
- Formatter/linter: `ruff` (`pip install ruff`, then `ruff check .` and `ruff format .`)
- Target: Python 3.10+

## `top_score` input contract

`wrap_system_prompt(qtype, top_score=...)` accepts any numeric value for `top_score`:

| Input | Behaviour |
|---|---|
| `None` | Treated as unknown → `[retrieval-quality: unknown]` |
| `float` in `[0, 1]` | Used as-is for confidence band selection |
| `float` outside `[0, 1]` (e.g. `-0.5`, `1.5`) | **Clamped** silently to `[0, 1]` — no exception raised |
| `nan` / `inf` / `-inf` | Treated as `None` → `[retrieval-quality: unknown]` |
| `int` `0` / `1` | Coerced to `float 0.0` / `1.0` |
| `bool` `True` / `False` | Coerced to `1.0` / `0.0` (Python bool subclasses int) |

Rationale: retrieval backends may return cosine values fractionally outside `[0, 1]` due to float
arithmetic, or `nan`/`inf` on degenerate inputs. The scaffold uses `top_score` only for a
cosmetic confidence label, so clamping is the principle-of-least-surprise choice — it never
raises and never breaks the caller's pipeline.

See `tests/scaffold/test_top_score_contract.py` for the full contract test suite.

## Pull requests

1. Fork and branch from `main`.
2. Keep changes focused — one feature or fix per PR.
3. Add a test if the change touches retrieval logic.
4. Update `CHANGELOG.md` under an `Unreleased` section.
