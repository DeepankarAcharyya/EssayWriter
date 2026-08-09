# Quality-Based Stop Condition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the essay loop when the critique grades the draft at or above a quality threshold, keeping the revision count as a backstop raised from 2 to 20.

**Architecture:** `reflect` switches from free text to structured output (`Critique` with a 1–10 `score` and `feedback`). The graph gains a second conditional edge: the revision cap is still checked after `generate`, and a new quality check after `reflect` routes to `END` or `research_critique`. Threshold and score live in `AgentState` so both stop predicates are pure functions of state.

**Tech Stack:** Python 3.11, LangGraph 1.2, LangChain 1.3 + `langchain-anthropic`, Pydantic, Tavily, uv, pytest.

## Global Constraints

- Python `>=3.11`; dependencies are managed with `uv` — never edit `uv.lock` by hand.
- Nodes return only the state keys they update; nothing mutates state in place.
- `EssayNodes` takes its model and search backend by injection. Tests must never make a network call — they always go through `build_graph(nodes=...)`.
- Docstrings are one-line and descriptive, matching the existing house style in `essaywriter/`.
- Quality threshold default: `8`. Revision cap default: `20`. Score scale: integer `1`–`10`.
- Environment variable names: `ESSAYWRITER_QUALITY_THRESHOLD`, `ESSAYWRITER_MAX_REVISIONS`, `ESSAYWRITER_MODEL`, `ESSAYWRITER_RESULTS_PER_QUERY`.

## Amendment to the spec

The spec did not account for LangGraph's default `recursion_limit` of 25. A run at the new
cap of 20 executes `plan` + `research_plan` + up to `3 × 20` loop supersteps ≈ 62, which
would abort with `GraphRecursionError` before reaching the cap. Task 5 sets
`recursion_limit` from `max_revisions` in the CLI run config. This is a required part of
the change, not an optional extra.

---

### Task 1: Test scaffolding and the `Critique` model

Adds the project's first tests. The fakes built here drive every later task, so they come
first even though `Critique` is the only production code in this task.

**Files:**
- Modify: `pyproject.toml` (add a `dev` dependency group)
- Modify: `essaywriter/state.py:25-31` (add `Critique` after `Queries`)
- Create: `tests/__init__.py` (empty — makes `from tests.conftest import ...` resolve)
- Create: `tests/conftest.py`
- Create: `tests/test_state.py`

**Interfaces:**
- Consumes: `essaywriter.state.Queries` (existing), `essaywriter.research.SearchBackend` (existing protocol, one method `gather(queries) -> List[str]`).
- Produces:
  - `essaywriter.state.Critique` — pydantic model, fields `score: int` (1–10 inclusive) and `feedback: str`.
  - `tests.conftest.FakeModel(text_replies=..., critiques=..., queries=...)` — duck-typed chat model with `.invoke(messages)` and `.with_structured_output(schema)`.
  - `tests.conftest.FakeSearch(snippets=...)` — records `.queries`, returns fixed snippets from `.gather()`.

- [ ] **Step 1: Add pytest as a dev dependency**

Run:

```bash
uv add --dev pytest
```

This creates a `[dependency-groups]` table in `pyproject.toml` and updates `uv.lock`.

- [ ] **Step 2: Write the fakes**

Create an empty `tests/__init__.py` first — the test modules import the fakes as
`from tests.conftest import ...`, which only resolves if `tests` is a package.

Then create `tests/conftest.py`. `FakeModel` is duck-typed on purpose — `EssayNodes` only ever
calls `.invoke()` and `.with_structured_output()`, so subclassing `BaseChatModel` would be
machinery with no payoff. Both scripted lists repeat their final entry once exhausted, so a
test that loops more times than expected fails on an assertion rather than an `IndexError`.

```python
"""Fakes that let the whole graph run with no model and no network."""

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from essaywriter.state import Critique, Queries


@dataclass
class FakeResponse:
    """Stands in for a chat model's message; nodes read `.content`."""

    content: str


class FakeModel:
    """Duck-typed chat model returning scripted replies and recording calls."""

    def __init__(
        self,
        text_replies: Sequence[str] = ("text",),
        critiques: Sequence[Critique] = (),
        queries: Sequence[str] = ("q1",),
    ) -> None:
        self._text_replies = list(text_replies)
        self._critiques = list(critiques)
        self._queries = list(queries)
        self.calls: List[str] = []

    def invoke(self, messages) -> FakeResponse:
        self.calls.append("invoke")
        return FakeResponse(_take(self._text_replies))

    def with_structured_output(self, schema):
        return _FakeStructured(self, schema)


class _FakeStructured:
    """What `FakeModel.with_structured_output(schema)` hands back."""

    def __init__(self, model: FakeModel, schema) -> None:
        self._model = model
        self._schema = schema

    def invoke(self, messages):
        if self._schema is Queries:
            self._model.calls.append("queries")
            return Queries(queries=list(self._model._queries))
        self._model.calls.append("critique")
        return _take(self._model._critiques)


class FakeSearch:
    """Search backend that records its queries and returns fixed snippets."""

    def __init__(self, snippets: Sequence[str] = ("snippet",)) -> None:
        self._snippets = list(snippets)
        self.queries: List[str] = []

    def gather(self, queries: Iterable[str]) -> List[str]:
        self.queries.extend(queries)
        return list(self._snippets)


def _take(scripted: list):
    """Pop the next scripted item, repeating the last one forever after."""
    if not scripted:
        raise AssertionError("the fake ran out of scripted responses")
    if len(scripted) == 1:
        return scripted[0]
    return scripted.pop(0)
```

- [ ] **Step 3: Write the failing test for `Critique`**

Create `tests/test_state.py`:

```python
"""The structured shapes the model is asked to produce."""

import pytest
from pydantic import ValidationError

from essaywriter.state import Critique


def test_critique_carries_a_score_and_feedback():
    critique = Critique(score=7, feedback="Add sources to paragraph three.")
    assert critique.score == 7
    assert critique.feedback == "Add sources to paragraph three."


@pytest.mark.parametrize("score", [0, 11, -3])
def test_critique_rejects_scores_off_the_scale(score):
    with pytest.raises(ValidationError):
        Critique(score=score, feedback="...")
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `uv run pytest tests/test_state.py -v`
Expected: FAIL — `ImportError: cannot import name 'Critique' from 'essaywriter.state'`

- [ ] **Step 5: Add the `Critique` model**

In `essaywriter/state.py`, append after the existing `Queries` class:

```python
class Critique(BaseModel):
    """A graded critique of a draft."""

    score: int = Field(
        ge=1,
        le=10,
        description=(
            "Quality of the draft on a 1-10 scale, where 8 or above is "
            "publishable."
        ),
    )
    feedback: str = Field(
        description="Concrete, actionable revisions the writer should make."
    )
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/ -v`
Expected: PASS — 4 passed.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock essaywriter/state.py tests/__init__.py tests/conftest.py tests/test_state.py
git commit -m "test: add pytest scaffolding and a graded Critique model"
```

---

### Task 2: `reflect` returns a grade

**Files:**
- Modify: `essaywriter/nodes.py:73-81` (the `reflect` method) and the import block at `essaywriter/nodes.py:24`
- Modify: `essaywriter/prompts.py:16-18` (`REFLECTION_PROMPT`)
- Create: `tests/test_nodes.py`

**Interfaces:**
- Consumes: `essaywriter.state.Critique`, `tests.conftest.FakeModel`, `tests.conftest.FakeSearch` (all from Task 1).
- Produces: `EssayNodes.reflect(state) -> {"critique": str, "score": int}` — the free-text `critique` key keeps the same meaning for `research_critique` and `generate`; `score` is new.

- [ ] **Step 1: Write the failing test**

Create `tests/test_nodes.py`:

```python
"""Node behaviour, exercised against fakes."""

from essaywriter.nodes import EssayNodes
from essaywriter.state import Critique

from tests.conftest import FakeModel, FakeSearch


def test_reflect_splits_the_grade_from_the_feedback():
    model = FakeModel(critiques=[Critique(score=6, feedback="Add sources.")])
    nodes = EssayNodes(model=model, search=FakeSearch())

    update = nodes.reflect({"draft": "a draft"})

    assert update == {"critique": "Add sources.", "score": 6}
    assert model.calls == ["critique"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_nodes.py -v`
Expected: FAIL — `reflect` returns `{"critique": FakeResponse(...).content}` with no `score` key, so the equality assertion fails.

- [ ] **Step 3: Rewrite `reflect` to use structured output**

In `essaywriter/nodes.py`, extend the state import:

```python
from essaywriter.state import AgentState, Critique, Queries
```

and replace the `reflect` method:

```python
    def reflect(self, state: AgentState) -> Dict[str, Any]:
        """Critique the draft — grade it and list concrete gaps."""
        critique = self._model.with_structured_output(Critique).invoke(
            [
                SystemMessage(content=REFLECTION_PROMPT),
                HumanMessage(content=state["draft"]),
            ]
        )
        return {"critique": critique.feedback, "score": critique.score}
```

- [ ] **Step 4: Anchor the rubric in the prompt**

A bare "score it 1–10" drifts between calls, which makes the threshold mean different
things on different runs. Replace `REFLECTION_PROMPT` in `essaywriter/prompts.py`:

```python
REFLECTION_PROMPT = """You are a teacher grading an essay submission. \
Generate critique and recommendations for the user's submission. \
Provide detailed recommendations, including requests for length, depth, style, etc.

Also grade the essay from 1 to 10 against this rubric:

- 1-3: incoherent, off-topic, or unsupported by evidence.
- 4-5: covers the topic but is thin, generic, or poorly structured.
- 6-7: solid and well-structured, with real gaps in evidence, depth or style.
- 8-9: publishable — well-argued, well-evidenced, and cleanly written. Any \
remaining changes are polish.
- 10: nothing left to improve.

Grade the draft in front of you, not the essay it could become. Do not inflate \
the score to be encouraging."""
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/ -v`
Expected: PASS — 5 passed.

- [ ] **Step 6: Commit**

```bash
git add essaywriter/nodes.py essaywriter/prompts.py tests/test_nodes.py
git commit -m "feat: grade the draft in reflect via structured output"
```

---

### Task 3: State keys and the quality stop predicate

**Files:**
- Modify: `essaywriter/state.py:9-22` (`AgentState`)
- Modify: `essaywriter/nodes.py:100-104` (add `is_good_enough` beside `should_continue`)
- Modify: `tests/test_nodes.py` (append predicate tests)

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `AgentState` keys `score: int` and `quality_threshold: int`.
  - `essaywriter.nodes.is_good_enough(state) -> str` — returns `END` or the string `"research_critique"`.
  - `essaywriter.nodes.should_continue(state) -> str` — unchanged behaviour, returns `END` or `"reflect"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_nodes.py`:

```python
from langgraph.graph import END

from essaywriter.nodes import is_good_enough, should_continue


def test_should_continue_critiques_while_under_the_cap():
    assert should_continue({"revision_number": 2, "max_revisions": 20}) == "reflect"


def test_should_continue_stops_past_the_cap():
    assert should_continue({"revision_number": 21, "max_revisions": 20}) == END


def test_is_good_enough_accepts_a_score_at_the_threshold():
    assert is_good_enough({"score": 8, "quality_threshold": 8}) == END


def test_is_good_enough_researches_below_the_threshold():
    state = {"score": 7, "quality_threshold": 8}
    assert is_good_enough(state) == "research_critique"
```

Put the two new imports at the top of the file with the existing ones rather than leaving
them mid-file.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_nodes.py -v`
Expected: FAIL — `ImportError: cannot import name 'is_good_enough' from 'essaywriter.nodes'`

- [ ] **Step 3: Add the state keys**

In `essaywriter/state.py`, extend `AgentState` with two keys and document them in the
existing class docstring's spirit:

```python
class AgentState(TypedDict):
    """State threaded through every node of the essay graph.

    `content` accumulates: nodes return only the snippets they found and
    LangGraph appends them, so no node mutates the list in place. `score` is
    the grade from the most recent critique; it starts at 0 and is compared
    against `quality_threshold` to decide whether to stop.
    """

    task: str
    plan: str
    draft: str
    critique: str
    content: Annotated[List[str], operator.add]
    revision_number: int
    max_revisions: int
    score: int
    quality_threshold: int
```

- [ ] **Step 4: Add the predicate**

In `essaywriter/nodes.py`, add below `should_continue`:

```python
def is_good_enough(state: AgentState) -> str:
    """After a critique: stop if the draft cleared the quality bar."""
    if state["score"] >= state["quality_threshold"]:
        return END
    return "research_critique"
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/ -v`
Expected: PASS — 9 passed.

- [ ] **Step 6: Commit**

```bash
git add essaywriter/state.py essaywriter/nodes.py tests/test_nodes.py
git commit -m "feat: add the quality stop predicate and its state keys"
```

---

### Task 4: Rewire the graph

**Files:**
- Modify: `essaywriter/graph.py:9` (import) and `essaywriter/graph.py:36-42` (edges)
- Create: `tests/test_graph.py`

**Interfaces:**
- Consumes: `is_good_enough`, `should_continue`, the new `AgentState` keys (Task 3); `FakeModel`, `FakeSearch` (Task 1); `Critique` (Task 1).
- Produces: a compiled graph whose `reflect` node routes conditionally. `build_graph`'s signature is unchanged.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_graph.py`. The helper keeps `max_revisions` small so the runs stay well
inside LangGraph's default recursion limit:

```python
"""End-to-end loop behaviour, driven entirely by fakes."""

from essaywriter.graph import build_graph
from essaywriter.nodes import EssayNodes
from essaywriter.state import Critique

from tests.conftest import FakeModel, FakeSearch


def run(model, search, *, max_revisions=3, quality_threshold=8):
    """Run the graph to completion and return the final state."""
    graph = build_graph(nodes=EssayNodes(model=model, search=search))
    return graph.invoke(
        {
            "task": "a topic",
            "content": [],
            "revision_number": 1,
            "max_revisions": max_revisions,
            "quality_threshold": quality_threshold,
            "score": 0,
        }
    )


def test_a_good_first_draft_ends_the_run():
    model = FakeModel(
        text_replies=["an outline", "draft one"],
        critiques=[Critique(score=9, feedback="Ship it.")],
    )
    search = FakeSearch()

    final = run(model, search)

    assert final["draft"] == "draft one"
    assert final["score"] == 9
    assert search.queries == ["q1"]  # only research_plan searched


def test_a_weak_draft_loops_until_it_clears_the_bar():
    model = FakeModel(
        text_replies=["an outline", "draft one", "draft two"],
        critiques=[
            Critique(score=5, feedback="Thin on evidence."),
            Critique(score=8, feedback="Good enough."),
        ],
    )
    search = FakeSearch()

    final = run(model, search)

    assert final["draft"] == "draft two"
    assert final["score"] == 8
    assert final["critique"] == "Good enough."
    assert len(search.queries) == 2  # the plan, then the critique


def test_the_cap_stops_a_draft_that_never_improves():
    model = FakeModel(
        text_replies=["an outline", "a draft"],
        critiques=[Critique(score=4, feedback="Still weak.")],
    )

    final = run(model, FakeSearch(), max_revisions=3)

    assert final["revision_number"] == 4  # three drafts written
    assert final["score"] == 4
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_graph.py -v`
Expected: FAIL — `test_a_good_first_draft_ends_the_run` fails because `reflect` still routes
unconditionally to `research_critique`, so the run keeps going to the cap and
`search.queries` is longer than `["q1"]`.

- [ ] **Step 3: Rewire the edges**

In `essaywriter/graph.py`, extend the import:

```python
from essaywriter.nodes import EssayNodes, is_good_enough, should_continue
```

and replace the `reflect` edge with a conditional one:

```python
    graph.add_conditional_edges(
        "generate",
        should_continue,
        {END: END, "reflect": "reflect"},
    )
    graph.add_conditional_edges(
        "reflect",
        is_good_enough,
        {END: END, "research_critique": "research_critique"},
    )
    graph.add_edge("research_critique", "generate")
```

The old `graph.add_edge("reflect", "research_critique")` line is deleted — leaving it in
place alongside the conditional edge would fan out to `research_critique` on every pass.

- [ ] **Step 4: Update the module docstring**

`essaywriter/graph.py` line 1 says `"""Wiring of the plan → research → write → critique loop."""`.
Leave it — it still describes the loop accurately.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/ -v`
Expected: PASS — 12 passed.

- [ ] **Step 6: Commit**

```bash
git add essaywriter/graph.py tests/test_graph.py
git commit -m "feat: stop the loop when the critique clears the quality bar"
```

---

### Task 5: Config, CLI, and the recursion limit

**Files:**
- Modify: `essaywriter/config.py:8-39`
- Modify: `essaywriter/cli.py:14-108`
- Create: `tests/test_config.py`
- Modify: `tests/test_graph.py` (append the recursion-headroom test)

**Interfaces:**
- Consumes: `Settings` (existing), `AgentState` keys from Task 3.
- Produces:
  - `Settings.quality_threshold: int` (default 8) and `Settings.max_revisions: int` (default now 20).
  - `essaywriter.cli.recursion_limit(max_revisions: int) -> int`.
  - CLI flag `--quality-threshold`.

- [ ] **Step 1: Write the failing config test**

Create `tests/test_config.py`:

```python
"""Settings resolution from the environment."""

from essaywriter.config import Settings


def test_defaults_favour_quality_over_a_tight_cap(monkeypatch):
    monkeypatch.delenv("ESSAYWRITER_MAX_REVISIONS", raising=False)
    monkeypatch.delenv("ESSAYWRITER_QUALITY_THRESHOLD", raising=False)

    settings = Settings.from_env()

    assert settings.max_revisions == 20
    assert settings.quality_threshold == 8


def test_the_threshold_reads_from_the_environment(monkeypatch):
    monkeypatch.setenv("ESSAYWRITER_QUALITY_THRESHOLD", "6")

    assert Settings.from_env().quality_threshold == 6


def test_explicit_overrides_beat_the_environment(monkeypatch):
    monkeypatch.setenv("ESSAYWRITER_QUALITY_THRESHOLD", "6")

    assert Settings.from_env(quality_threshold=9).quality_threshold == 9
```

- [ ] **Step 2: Write the failing recursion-headroom test**

Append to `tests/test_graph.py`:

```python
from essaywriter.cli import recursion_limit


def test_recursion_limit_clears_a_full_run_at_the_cap():
    # plan + research_plan, then generate + reflect + research_critique per revision.
    assert recursion_limit(20) >= 2 + 3 * 20
```

- [ ] **Step 3: Run both to verify they fail**

Run: `uv run pytest tests/test_config.py tests/test_graph.py -v`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'quality_threshold'`
and `ImportError: cannot import name 'recursion_limit' from 'essaywriter.cli'`.

- [ ] **Step 4: Update `Settings`**

In `essaywriter/config.py`, change the defaults block and the dataclass:

```python
DEFAULT_MODEL = "anthropic:claude-sonnet-4-5"
DEFAULT_MAX_REVISIONS = 20
DEFAULT_QUALITY_THRESHOLD = 8
DEFAULT_RESULTS_PER_QUERY = 2


@dataclass(frozen=True)
class Settings:
    """Everything the agent needs to run, resolved once at startup."""

    model: str = DEFAULT_MODEL
    tavily_api_key: str | None = None
    max_revisions: int = DEFAULT_MAX_REVISIONS
    quality_threshold: int = DEFAULT_QUALITY_THRESHOLD
    results_per_query: int = DEFAULT_RESULTS_PER_QUERY
```

and add the environment entry inside `from_env`'s `defaults` dict, beside `max_revisions`:

```python
            "quality_threshold": int(
                os.environ.get(
                    "ESSAYWRITER_QUALITY_THRESHOLD", DEFAULT_QUALITY_THRESHOLD
                )
            ),
```

- [ ] **Step 5: Update the CLI**

In `essaywriter/cli.py`, add the flag inside `parse_args`, after `--max-revisions`:

```python
    parser.add_argument(
        "--quality-threshold",
        type=int,
        default=None,
        help="Score (1-10) at which the draft is accepted (default: 8).",
    )
```

Add the helper above `write_essay`:

```python
def recursion_limit(max_revisions: int) -> int:
    """Supersteps a full run needs: plan, research, then 3 nodes per revision."""
    return 2 + 3 * max_revisions + 1
```

Replace the body of `write_essay` so it seeds the new state keys, raises the recursion
limit, and reports the grade:

```python
def write_essay(
    topic: str,
    settings: Settings,
    checkpoint_db: str = ":memory:",
    thread_id: str = "1",
    verbose: bool = True,
) -> str:
    """Run the graph to completion and return the final draft."""
    with SqliteSaver.from_conn_string(checkpoint_db) as checkpointer:
        graph = build_graph(checkpointer=checkpointer, settings=settings)
        initial_state = {
            "task": topic,
            "content": [],
            "revision_number": 1,
            "max_revisions": settings.max_revisions,
            "quality_threshold": settings.quality_threshold,
            "score": 0,
        }
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": recursion_limit(settings.max_revisions),
        }

        draft = ""
        for step in graph.stream(initial_state, config):
            for node, update in step.items():
                if verbose:
                    print(_progress(node, update, settings), file=sys.stderr)
                if "draft" in update:
                    draft = update["draft"]
        return draft
```

and add the formatting helper next to `recursion_limit`:

```python
def _progress(node: str, update: dict, settings: Settings) -> str:
    """One stderr line per node, carrying the grade when there is one."""
    if "score" in update:
        return (
            f"[{node}] score {update['score']}/10 "
            f"(target {settings.quality_threshold})"
        )
    return f"[{node}]"
```

Finally, thread the new flag through `main`:

```python
    settings = Settings.from_env(
        model=args.model,
        max_revisions=args.max_revisions,
        quality_threshold=args.quality_threshold,
    )
```

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest tests/ -v`
Expected: PASS — 16 passed.

- [ ] **Step 7: Check the CLI wiring by hand**

Run: `uv run main.py --help`
Expected: the help text lists `--quality-threshold`, and `--max-revisions` still reads
"default: from env, else 2" — update that help string to say `else 20`, since the default
changed:

```python
        help="How many revision loops to allow (default: from env, else 20).",
```

Re-run `uv run main.py --help` and confirm it now reads 20.

- [ ] **Step 8: Commit**

```bash
git add essaywriter/config.py essaywriter/cli.py tests/test_config.py tests/test_graph.py
git commit -m "feat: configure the quality bar and raise the revision cap to 20"
```

---

### Task 6: Documentation

**Files:**
- Modify: `README.md` (env table, CLI flag table, how-it-works diagram and node table, project layout, roadmap)

**Interfaces:**
- Consumes: everything above. Produces: no code.

- [ ] **Step 1: Update the diagram and the loop description**

In `README.md`, the intro line currently reads "It loops until the revision limit is hit."
Replace with:

> It loops until the critique grades the draft at or above the quality bar, or the revision cap is hit.

Replace the ASCII diagram's `generate` branch so both stop points appear:

```
        ┌──────────┐
        │   plan   │  outline the essay
        └────┬─────┘
             ▼
     ┌───────────────┐
     │ research_plan │  gather sources for the outline
     └───────┬───────┘
             ▼
        ┌──────────┐
        │ generate │◄──────────────┐  write / rewrite the draft
        └────┬─────┘               │
             ▼                     │
      ╱ hit the cap? ╲─── yes ──► END
             │ no                  │
             ▼                     │
        ┌──────────┐               │
        │ reflect  │  grade the draft 1-10
        └────┬─────┘               │
             ▼                     │
     ╱ score >= bar? ╲─── yes ──► END
             │ no                  │
             ▼                     │
  ┌────────────────────┐           │
  │ research_critique  │───────────┘  gather sources answering the critique
  └────────────────────┘
```

Replace the paragraph below the node table:

> After `generate` the agent stops if the revision cap is reached. Otherwise `reflect`
> grades the draft from 1 to 10; a score at or above the quality threshold ends the run,
> and anything lower goes to `research_critique` and back around. The cap defaults to 20 —
> it is a backstop, not a target.

Update the `reflect` row of the node table to: "Grade the draft 1–10 and list concrete gaps."

- [ ] **Step 2: Update the config and flag tables**

Add to the environment table:

```
| `ESSAYWRITER_QUALITY_THRESHOLD` | `8` |
```

and change the `ESSAYWRITER_MAX_REVISIONS` default from `2` to `20`.

Add to the CLI flag table:

```
| `--quality-threshold N` | Score (1-10) at which the draft is accepted. |
```

- [ ] **Step 3: Update the project layout and roadmap**

Add to the layout block, after the `essaywriter/` entries:

```
├── tests/               # pytest suite; fakes injected via build_graph(nodes=...)
├── docs/superpowers/    # specs and implementation plans
```

Tick both remaining roadmap boxes:

```markdown
- [x] Test suite covering the loop and the revision cutoff
- [x] Quality-based stop condition, not just a revision cap
```

- [ ] **Step 4: Verify the docs match the code**

Run: `uv run pytest tests/ -v && uv run main.py --help`
Expected: 16 passed, and every flag named in the README's table appears in the help output.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: describe the quality-based stop condition"
```

---

## Self-review

**Spec coverage:** graph shape → Task 4; structured critique and rubric → Task 2; state keys
→ Task 3; stop predicates → Task 3; config and CLI → Task 5; progress reporting → Task 5;
all five test cases → Tasks 3 and 4; failure modes → covered by the pydantic bounds test
(Task 1) and the cap test (Task 4). The recursion limit is an addition to the spec,
documented above and implemented in Task 5.

**Type consistency:** `Critique.score` / `Critique.feedback` are used with those exact names
in Tasks 1, 2 and 4. `is_good_enough` and `should_continue` keep the same names and return
types in Tasks 3, 4 and their tests. `recursion_limit` is defined and consumed in Task 5,
and asserted against in Task 4's file (appended during Task 5).
