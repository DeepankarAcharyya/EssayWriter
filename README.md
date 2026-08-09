# EssayWriter Agent

An essay-writing agent — a compact version of an AI researcher.

Given a topic, the agent plans an essay, researches it, drafts it, critiques its own draft, researches the gaps the critique exposes, and revises. It loops until the revision limit is hit.

> **Status:** the graph below is implemented and runnable.

## How it works

The agent is a [LangGraph](https://langchain-ai.github.io/langgraph/) state machine with a research → write → critique loop:

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
         ╱ done? ╲──── yes ──► END │
             │ no                  │
             ▼                     │
        ┌──────────┐               │
        │ reflect  │  critique the draft
        └────┬─────┘               │
             ▼                     │
  ┌────────────────────┐           │
  │ research_critique  │───────────┘  gather sources answering the critique
  └────────────────────┘
```

| Node | Job |
| --- | --- |
| `plan` | Turn the topic into a section-by-section outline. |
| `research_plan` | Turn the outline into search queries; collect the results. |
| `generate` | Write (or rewrite) the draft from the outline, research, and any critique. |
| `reflect` | Critique the draft — grade it and list concrete gaps. |
| `research_critique` | Search for what the critique says is missing. |

After `generate`, the agent stops if the revision limit is reached; otherwise it goes to `reflect` and loops back through `generate`.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for dependency management
- An Anthropic API key, and a [Tavily](https://tavily.com) key for the research nodes

## Setup

```bash
git clone <repo-url>
cd EssayWriter
uv sync
```

Create a `.env` in the project root:

```env
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...
```

`.env` is gitignored — keep your keys out of version control.

Optional environment overrides:

| Variable | Default |
| --- | --- |
| `ESSAYWRITER_MODEL` | `anthropic:claude-sonnet-4-5` |
| `ESSAYWRITER_MAX_REVISIONS` | `2` |
| `ESSAYWRITER_RESULTS_PER_QUERY` | `2` |

## Usage

```bash
uv run main.py "The economics of open source"
uv run main.py "..." --max-revisions 3 -o essay.md
uv run main.py "..." --model anthropic:claude-opus-4-5 --quiet
uv run main.py "..." --checkpoint-db runs.sqlite --thread-id econ   # resumable
```

Installing the project also exposes an `essaywriter` console script, so
`uv run essaywriter "..."` works the same way.

| Flag | Job |
| --- | --- |
| `-o`, `--output PATH` | Write the essay to a file instead of stdout. |
| `--max-revisions N` | Revision loops to allow. Overrides the env default. |
| `--model NAME` | Chat model, e.g. `anthropic:claude-sonnet-4-5`. |
| `--checkpoint-db PATH` | SQLite file for checkpoints (default `:memory:`). |
| `--thread-id ID` | Checkpoint thread; reuse it to resume a run (default `1`). |
| `-q`, `--quiet` | Suppress the per-node progress lines. |

Progress goes to stderr, the essay to stdout (or `--output`), so piping works.
Missing credentials or bad config exit with status `2`.

## Project layout

```
EssayWriter/
├── main.py              # entry point shim
├── essaywriter/
│   ├── __init__.py      # public surface: AgentState, Settings, build_graph
│   ├── cli.py           # argument parsing, run loop, output
│   ├── config.py        # Settings, loaded from .env / environment
│   ├── graph.py         # build_graph() — nodes and edges
│   ├── nodes.py         # EssayNodes: one method per graph node
│   ├── prompts.py       # system prompt per node
│   ├── research.py      # SearchBackend protocol + Tavily implementation
│   └── state.py         # AgentState, Queries
├── pyproject.toml       # project metadata and dependencies
├── uv.lock              # locked dependency versions
├── .python-version      # 3.11
├── .env                 # API keys (not committed)
└── LICENSE              # MIT
```

`EssayNodes` takes its model and search backend by injection, so
`build_graph(nodes=...)` can run the whole loop against fakes without network calls.

## Roadmap

- [x] Define the graph state (topic, plan, draft, critique, sources, revision count)
- [x] Implement the `plan`, `research_plan`, `generate`, `reflect`, `research_critique` nodes
- [x] Wire up a search backend for the research nodes
- [x] Add a CLI that takes a topic and writes the essay to a file
- [x] Checkpointing so long runs can resume
- [ ] Test suite covering the loop and the revision cutoff
- [ ] Quality-based stop condition, not just a revision cap

## License

MIT — see [LICENSE](LICENSE).
