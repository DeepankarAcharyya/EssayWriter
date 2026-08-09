# Your LLM Writes First Drafts. Here's How to Make It Write Second Ones.

Ask a language model for an essay and you get a first draft. It is usually decent — coherent, well-organized, correctly punctuated. It is also whatever the model happened to produce on the first pass, and the model has no idea whether that was good.

Human writers do not work that way. They outline, go looking for evidence, write something, decide it is thin in the third section, go find more evidence about the third section, and rewrite. The interesting part is not the writing. It is the loop.

I built a small agent that runs that loop. It is about 500 lines of Python on top of [LangGraph](https://langchain-ai.github.io/langgraph/), and the part worth stealing is not the prompts — it is the stop condition.

## The shape of it

Five nodes, two decisions:

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

Two patterns are stacked here. **Reflection**: the model grades its own output and the grade steers control flow. **Retrieval on demand**: the agent does not fetch everything up front — it fetches once for the outline, then fetches again for whatever the critique says is missing. Round three searches for different things than round one, because by round three the agent knows what it got wrong.

## State first

In LangGraph you define the state before the nodes, and the state definition is where the design decisions actually live:

```python
class AgentState(TypedDict):
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

Most of these fields overwrite. `draft` is the latest draft; `critique` is the latest critique. The one that does not is `content`, the research snippets, annotated with `operator.add`. Nodes return only the snippets they just found, and LangGraph appends them.

That matters more than it looks. The research from round one is still in context in round four. The agent accumulates evidence across the whole run instead of forgetting what it already looked up — and no node has to mutate a shared list, which is exactly the sort of thing that breaks the moment you add concurrency.

## Nodes are boring, and that is the point

Each node takes the state and returns only the keys it changes. `plan` turns the topic into an outline. `research_plan` turns the outline into search queries and runs them. `generate` writes the draft:

```python
def generate(self, state: AgentState) -> Dict[str, Any]:
    content = "\n\n".join(state.get("content") or [])
    response = self._model.invoke([
        SystemMessage(content=WRITER_PROMPT.format(content=content)),
        HumanMessage(
            content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}"
        ),
    ])
    return {
        "draft": response.content,
        "revision_number": state.get("revision_number", 1) + 1,
    }
```

Note there is no separate "revise" node. `generate` handles both cases, because the difference between writing and rewriting is entirely in the context: on the second pass, `content` contains the critique-driven research and the writer prompt says "if the user provides critique, respond with a revised version of your previous attempts." One node, two jobs, no branching.

`reflect` is the one that earns its keep:

```python
def reflect(self, state: AgentState) -> Dict[str, Any]:
    critique = self._model.with_structured_output(Critique).invoke([
        SystemMessage(content=REFLECTION_PROMPT),
        HumanMessage(content=state["draft"]),
    ])
    return {"critique": critique.feedback, "score": critique.score}
```

`with_structured_output` is doing the heavy lifting. `Critique` is a Pydantic model:

```python
class Critique(BaseModel):
    score: int = Field(ge=1, le=10, description=...)
    feedback: str = Field(description="Concrete, actionable revisions...")
```

The grade comes back as an `int`, validated in range, not as the phrase "I'd rate this an 8/10" that I then have to regex out of a paragraph. That is what makes the grade usable as a control-flow signal rather than as decoration.

## The stop condition is the whole design

Here is the part I actually want to argue about.

The obvious way to end an agent loop is a counter: run three revisions, stop. Every reflection tutorial does this. It is also wrong, and the reason is that a counter answers the wrong question. It stops when the agent is *tired*, not when the essay is *done*. Three revisions is too many for a topic the model already knows cold, and too few for one where the first draft was genuinely bad.

So the loop stops on the grade instead:

```python
def should_continue(state: AgentState) -> str:
    """After a draft: stop at the revision limit, otherwise critique and loop."""
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "reflect"


def is_good_enough(state: AgentState) -> str:
    """After a critique: stop if the draft cleared the quality bar."""
    if state["score"] >= state["quality_threshold"]:
        return END
    return "research_critique"
```

Two predicates, both one-liners. `is_good_enough` is the real exit — a draft that scores 8 or better ships immediately, whether that took one round or six. `should_continue` keeps the counter, but demoted: `max_revisions` defaults to 20, which is a backstop against a pathological loop, not a target. In practice the run never reaches it.

This only works if the grade is honest, and by default it is not. Models grade generously; a model grading *its own* draft grades more generously still. The rubric prompt has to fight that explicitly:

```
- 6-7: solid and well-structured, with real gaps in evidence, depth or style.
- 8-9: publishable — well-argued, well-evidenced, and cleanly written.
- 10: nothing left to improve.

Grade the draft in front of you, not the essay it could become. Do not
inflate the score to be encouraging.
```

That last line changed behavior noticeably. Without it, drafts got an encouraging 8 on round one and the loop never ran. The failure mode of a self-grading agent is not harshness — it is a model that gives everything a passing grade and quietly turns your loop into a straight line.

## Wiring

The graph itself is declarative and short:

```python
graph.add_edge(START, "plan")
graph.add_edge("plan", "research_plan")
graph.add_edge("research_plan", "generate")
graph.add_conditional_edges("generate", should_continue,
                            {END: END, "reflect": "reflect"})
graph.add_conditional_edges("reflect", is_good_enough,
                            {END: END, "research_critique": "research_critique"})
graph.add_edge("research_critique", "generate")
return graph.compile(checkpointer=checkpointer)
```

`research_critique → generate` is the back-edge that makes it a loop. Everything else is a straight line. Pass a `SqliteSaver` as the checkpointer and every step is persisted, so a run that dies at revision four resumes at revision four instead of starting over.

## Testing an agent without paying for it

The nodes live on a class that takes its model and search backend as constructor arguments:

```python
class EssayNodes:
    def __init__(self, model: BaseChatModel, search: SearchBackend) -> None:
```

and `build_graph(nodes=...)` accepts a pre-built instance. That one seam means the entire loop runs against fakes — no API key, no network, no tokens:

```python
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
    assert len(search.queries) == 2  # the plan, then the critique
```

The fake model hands back a scripted sequence of grades, so I can assert the *control flow* — that a 5 sends it around again, that an 8 stops it, that the second trip through `research_critique` actually ran a second search. None of that needs a real model, because none of it is about the model's writing quality. It is about the graph.

The tests that would need a real model — is the essay any good? — are the ones a test suite can't answer anyway.

## Running it

```bash
uv run main.py "The economics of open source"
```

Progress goes to stderr, the essay to stdout, so it pipes:

```
[plan]
[research_plan]
[generate]
[reflect] score 6/10 (target 8)
[research_critique]
[generate]
[reflect] score 9/10 (target 8)
```

My favorite test run was "how Spider-Man's DNA was merged with that of a spider — the tradeoffs." Round one came back a 6: coherent, but it listed powers without costing them. The critique asked for metabolic and immunological specifics, `research_critique` went and found transgenic-organism research, and round two came back arguing that ten-ton lifting capacity implies a 6,000–10,000 calorie daily floor and an autoimmune problem with permanently circulating foreign proteins. Same model, same topic, one extra loop. The difference was that something told it the first draft was a 6.

## What I'd add next

Per-section critique rather than whole-draft — a single 1–10 for a five-paragraph essay is a coarse signal, and the agent currently rewrites the whole thing to fix one weak section. Source attribution in the output, so claims trace back to the search results that produced them. And a second grader with a different rubric, because one judge is one judge.

The code is on [GitHub](https://github.com/DeepankarAcharyya/EssayWriter) under MIT. The loop is the part worth copying; the prompts are yours to argue with.
