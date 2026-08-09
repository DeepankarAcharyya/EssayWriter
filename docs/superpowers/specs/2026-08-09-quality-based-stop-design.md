# Quality-based stop condition

**Date:** 2026-08-09
**Status:** approved, not yet implemented

## Problem

The essay graph stops only when it runs out of revisions. `should_continue` compares
`revision_number` against `max_revisions` (default 2) and returns `END` when the count is
exceeded. Nothing looks at whether the draft is any good: a strong first draft still burns
every remaining revision, and a weak one stops the moment the counter runs out.

## Goal

Stop when the draft is good enough. Keep a revision cap as a backstop, raised to 20 so it
is rarely the reason a run ends.

## Design

### Graph shape

Two conditional edges replace the single one:

```
START → plan → research_plan → generate
                                  │
              ┌───────────────────┴───────────────────┐
              │ revision_number > max_revisions?      │
              ├── yes ──► END                         │
              └── no  ──► reflect                     │
                             │                        │
              ┌──────────────┴──────────────┐         │
              │ score >= quality_threshold? │         │
              ├── yes ──► END               │         │
              └── no  ──► research_critique ──────────┘
```

The cap check stays *before* `reflect` so a run that has exhausted its revisions does not
pay for a critique it cannot act on. The quality check goes *after* `reflect`, the only
point at which a grade exists.

Node functions are unchanged in signature. `graph.py` gains one more
`add_conditional_edges` call and drops the `reflect → research_critique` static edge.

### Structured critique

`reflect` currently returns free text. It switches to structured output so the grade is a
real value rather than something parsed out of prose. New model in `state.py`, alongside
`Queries`:

```python
class Critique(BaseModel):
    """A graded critique of a draft."""

    score: int = Field(
        ge=1, le=10, description="Quality of the draft on a 1-10 scale."
    )
    feedback: str = Field(
        description="Concrete, actionable revisions the writer should make."
    )
```

`EssayNodes.reflect` calls `self._model.with_structured_output(Critique)` and returns
`{"critique": result.feedback, "score": result.score}`.

Downstream nodes are untouched: `research_critique` and `generate` still read `critique`
as a string.

`REFLECTION_PROMPT` gains a rubric so the scale is stable across calls — what a 5, an 8 and
a 10 look like, with 8 stated as the publishable bar. Without an anchored rubric the model
drifts and the threshold means different things on different runs.

### State

`AgentState` gains two keys:

| Key | Meaning |
| --- | --- |
| `score` | Grade from the most recent `reflect`. Starts at 0. |
| `quality_threshold` | Score at which the draft is accepted. |

The threshold lives in state next to `max_revisions` so the stop predicates stay pure
functions of state and can be tested without constructing `Settings`.

### Stop predicates

`should_continue` keeps its name and its cap-only job. A second predicate is added:

```python
def should_continue(state: AgentState) -> str:
    """After a draft: stop at the revision limit, otherwise critique."""
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "reflect"


def is_good_enough(state: AgentState) -> str:
    """After a critique: stop if the draft cleared the quality bar."""
    if state["score"] >= state["quality_threshold"]:
        return END
    return "research_critique"
```

### Config and CLI

| Setting | Default | Environment variable |
| --- | --- | --- |
| `quality_threshold` | 8 | `ESSAYWRITER_QUALITY_THRESHOLD` |
| `max_revisions` | 20 (was 2) | `ESSAYWRITER_MAX_REVISIONS` |

`Settings` gains `quality_threshold`; `DEFAULT_MAX_REVISIONS` changes to 20 and
`DEFAULT_QUALITY_THRESHOLD` is added. `cli.py` gains a `--quality-threshold` flag
threaded through `Settings.from_env` exactly as `--max-revisions` is today, and seeds
`score: 0` and `quality_threshold` into the initial state.

Progress reporting shows the grade so a long run is legible:

```
[reflect] score 7/10 (target 8)
```

Other node lines keep the existing `[node]` form.

### Testing

The project has no tests. This change adds the first suite, using the injection seam that
already exists — `build_graph(nodes=...)` accepts a pre-built `EssayNodes`, so a fake model
and a fake `SearchBackend` drive the whole loop with no network calls.

Cases:

1. A first draft scoring at or above the threshold ends the run after one `generate`.
2. A draft scoring below the threshold loops through `research_critique` and generates again.
3. A draft that never clears the threshold stops at `max_revisions`, returning the last draft.
4. The final state carries the last `score` and `critique`.
5. `should_continue` and `is_good_enough` return the right branch at their boundaries
   (score exactly equal to the threshold accepts; `revision_number == max_revisions` continues).

Fakes live in `tests/conftest.py`: a `FakeChatModel` returning scripted responses (including
a `with_structured_output` that yields scripted `Critique`/`Queries` objects) and a
`FakeSearch` returning fixed snippets. `pytest` is added as a dev dependency group.

## Failure modes

- **Out-of-range score.** Pydantic `ge`/`le` validation rejects it and LangChain's
  structured-output path retries. A persistently malformed response surfaces as an error
  rather than a silently wrong stop decision.
- **Model never clears the bar.** The cap terminates the run at 20 revisions with the best
  draft produced. This is the pre-existing behaviour, just with more headroom.
- **Cost.** Each loop costs three model calls and one research round (up to 3 searches).
  The quality stop is what keeps real runs short; the raised cap is a backstop, not a
  target. The per-loop score in the progress output makes the spend visible while it runs.

## Out of scope

- Plateau detection (stop when the score stalls across revisions). Considered and dropped;
  revisit if runs are observed burning the cap.
- Skipping `research_critique` for near-threshold drafts.
- Any change to `plan`, `research_plan` or `generate`.
