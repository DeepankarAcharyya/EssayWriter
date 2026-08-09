# Report File Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every run writes its essay to `output/<topic-slug>-<timestamp>.md`, without disturbing the existing stdout and `--output` behaviour.

**Architecture:** A new `essaywriter/report.py` owns slug rules, path construction, and file writing. It takes the timestamp as a parameter rather than calling `datetime.now()` internally, so naming is testable at a fixed instant. `cli.main` calls it once, after `write_essay` returns, and keeps its existing output branches unchanged.

**Tech Stack:** Python 3.11, LangGraph, uv, pytest.

## Global Constraints

- Python `>=3.11`; dependencies are managed with `uv` — never edit `uv.lock` by hand. This change adds no dependencies.
- Docstrings are one-line and descriptive, matching the existing house style in `essaywriter/`.
- Tests never make a network call and never write outside pytest's `tmp_path`.
- Output directory: `output/`, fixed — no flag configures it.
- Filename format: `<slug>-<%Y%m%d-%H%M%S>.md`, local time.
- Slug: lowercase; runs of non-alphanumeric characters collapse to one hyphen; leading and trailing hyphens stripped; truncated to 60 characters with any trailing hyphen removed; `essay` when nothing survives. "Alphanumeric" means `str.isalnum()`, which keeps non-Latin characters.
- Existing behaviour that must not change: `-o PATH` still writes its own copy, and with no `-o` the essay is still printed to stdout.

---

### Task 1: The report module

**Files:**
- Create: `essaywriter/report.py`
- Create: `tests/test_report.py`

**Interfaces:**
- Consumes: nothing from the existing package — this module stands alone.
- Produces:
  - `essaywriter.report.OUTPUT_DIR: Path` — `Path("output")`.
  - `essaywriter.report.slugify(topic: str, max_length: int = 60) -> str`
  - `essaywriter.report.report_path(topic: str, when: datetime, directory: Path = OUTPUT_DIR) -> Path`
  - `essaywriter.report.write_report(topic: str, essay: str, when: datetime, directory: Path = OUTPUT_DIR) -> Path`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_report.py`. Every test that touches the filesystem uses pytest's
`tmp_path` — none of them write into the real `output/`.

```python
"""Naming and writing the per-run report file."""

from datetime import datetime

from essaywriter.report import report_path, slugify, write_report

WHEN = datetime(2026, 8, 9, 15, 30, 42)


def test_slugify_lowercases_and_hyphenates_punctuation():
    assert slugify("The Economics of Open Source: a 2026 view") == (
        "the-economics-of-open-source-a-2026-view"
    )


def test_slugify_collapses_runs_of_separators():
    assert slugify("what   now???  really") == "what-now-really"


def test_slugify_trims_leading_and_trailing_separators():
    assert slugify("  !!hello!!  ") == "hello"


def test_slugify_truncates_without_a_trailing_hyphen():
    # 59 a's, a separator, then more: the 60-char cut lands on the separator.
    slug = slugify("a" * 59 + " bcdef")

    assert slug == "a" * 59
    assert len(slug) <= 60


def test_slugify_falls_back_when_nothing_survives():
    assert slugify("???") == "essay"


def test_slugify_keeps_non_latin_characters():
    assert slugify("Москва зимой") == "москва-зимой"


def test_report_path_names_the_file_after_the_topic_and_time(tmp_path):
    path = report_path("Open source", WHEN, directory=tmp_path)

    assert path == tmp_path / "open-source-20260809-153042.md"


def test_report_path_defaults_to_the_output_directory():
    assert report_path("Open source", WHEN).parent.name == "output"


def test_write_report_creates_the_directory_and_writes_the_essay(tmp_path):
    directory = tmp_path / "output"

    path = write_report("Open source", "# An essay\n", WHEN, directory=directory)

    assert path == directory / "open-source-20260809-153042.md"
    assert path.read_text() == "# An essay\n"


def test_write_report_reuses_an_existing_directory(tmp_path):
    tmp_path.mkdir(exist_ok=True)

    path = write_report("Open source", "body", WHEN, directory=tmp_path)

    assert path.read_text() == "body"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'essaywriter.report'`

- [ ] **Step 3: Write the module**

Create `essaywriter/report.py`:

```python
"""Naming and writing the report file for a run."""

from datetime import datetime
from pathlib import Path

OUTPUT_DIR = Path("output")
MAX_SLUG_LENGTH = 60
TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"
FALLBACK_SLUG = "essay"


def slugify(topic: str, max_length: int = MAX_SLUG_LENGTH) -> str:
    """Reduce a topic to lowercase, hyphen-separated, path-safe text."""
    characters = [
        character if character.isalnum() else "-" for character in topic.lower()
    ]
    slug = "".join(characters)
    while "--" in slug:
        slug = slug.replace("--", "-")
    slug = slug.strip("-")[:max_length].rstrip("-")
    return slug or FALLBACK_SLUG


def report_path(
    topic: str, when: datetime, directory: Path = OUTPUT_DIR
) -> Path:
    """Where this topic's report belongs: <directory>/<slug>-<stamp>.md."""
    stamp = when.strftime(TIMESTAMP_FORMAT)
    return directory / f"{slugify(topic)}-{stamp}.md"


def write_report(
    topic: str, essay: str, when: datetime, directory: Path = OUTPUT_DIR
) -> Path:
    """Write the essay to its report path, creating the directory, and return it."""
    path = report_path(topic, when, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(essay)
    return path
```

Note on the truncation order: the slug is stripped *before* slicing so a leading separator
does not consume part of the 60-character budget, and `rstrip("-")` runs *after* the slice
so a cut landing on a separator does not leave a trailing hyphen.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_report.py -v`
Expected: PASS — 10 passed.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS — 32 passed (22 existing + 10 new), no warnings.

- [ ] **Step 6: Commit**

```bash
git add essaywriter/report.py tests/test_report.py
git commit -m "feat: add report-file naming and writing"
```

---

### Task 2: Wire it into the CLI and document it

**Files:**
- Modify: `essaywriter/cli.py` (imports at the top; `main` at lines 110-137)
- Modify: `.gitignore`
- Modify: `README.md`

**Interfaces:**
- Consumes: `essaywriter.report.write_report(topic, essay, when, directory=OUTPUT_DIR) -> Path` from Task 1.
- Produces: no new public API. `main`'s return codes are unchanged (`0` success, `2` bad config).

- [ ] **Step 1: Add the imports**

At the top of `essaywriter/cli.py`, alongside the existing imports:

```python
from datetime import datetime
```

and with the other `essaywriter` imports:

```python
from essaywriter.report import write_report
```

- [ ] **Step 2: Write the report in `main`**

`main` currently ends like this (lines 110-137):

```python
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    try:
        settings = Settings.from_env(
            model=args.model,
            max_revisions=args.max_revisions,
            quality_threshold=args.quality_threshold,
        )
        essay = write_essay(
            topic=args.topic,
            settings=settings,
            checkpoint_db=args.checkpoint_db,
            thread_id=args.thread_id,
            verbose=not args.quiet,
        )
    except ValueError as exc:  # missing credentials, bad config
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.output:
        args.output.write_text(essay)
        if not args.quiet:
            print(f"wrote {args.output}", file=sys.stderr)
    else:
        print(essay)
    return 0
```

Insert the report write between the `except` block and the `if args.output:` branch, and
leave both existing branches exactly as they are:

```python
    report = write_report(args.topic, essay, datetime.now())
    if not args.quiet:
        print(f"wrote {report}", file=sys.stderr)

    if args.output:
        args.output.write_text(essay)
        if not args.quiet:
            print(f"wrote {args.output}", file=sys.stderr)
    else:
        print(essay)
    return 0
```

The report write sits outside the `try` deliberately: a `ValueError` from bad config must
still exit 2 without touching the filesystem, and by this point there is a finished essay
worth saving.

- [ ] **Step 3: Verify the CLI end to end**

The graph needs API keys, so drive `main` through a fake instead. Run this from the project
root:

```bash
uv run python -c "
import sys, tempfile, os
from pathlib import Path
import essaywriter.cli as cli

cli.write_essay = lambda **kwargs: '# A test essay'
os.chdir(tempfile.mkdtemp())
code = cli.main(['The Economics of Open Source: a 2026 view'])
print('exit', code)
print([str(p) for p in Path('output').iterdir()])
"
```

Expected: `exit 0`, a `wrote output/...` line on stderr, the essay on stdout, and one file
listed whose name starts `the-economics-of-open-source-a-2026-view-` and ends `.md`.

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS — 32 passed, no warnings.

- [ ] **Step 5: Ignore the output directory**

Append to `.gitignore`:

```
# Run reports
output/
```

- [ ] **Step 6: Update the README**

In the Usage section, after the line

> Progress goes to stderr, the essay to stdout (or `--output`), so piping works.
> Missing credentials or bad config exit with status `2`.

add:

> Every run also writes `output/<topic>-<timestamp>.md` — the topic lowercased and
> hyphenated, with a local-time stamp — so runs are kept without having to name files by
> hand. `output/` is gitignored.

In the project layout block, add a line after the `tests/` entry:

```
├── output/              # per-run reports (gitignored)
```

- [ ] **Step 7: Commit**

```bash
git add essaywriter/cli.py .gitignore README.md
git commit -m "feat: write every run to a dated report file"
```

---

## Self-review

**Spec coverage:** the module and its three functions → Task 1; slug rules including the
Unicode and empty-topic cases → Task 1 Steps 1 and 3; timestamp format → Task 1; CLI flow
with all three rows of the spec's table → Task 2 Step 2; collisions → accepted by the spec,
no code; testing (all seven spec cases, plus three more covering separator collapsing,
the `OUTPUT_DIR` default, and an existing directory) → Task 1; `.gitignore` and README →
Task 2 Steps 5 and 6.

**Type consistency:** `slugify`, `report_path`, `write_report`, `OUTPUT_DIR` keep the same
names and signatures in Task 1's implementation, Task 1's tests, and Task 2's call site.
`write_report` returns a `Path`, which Task 2 interpolates into the stderr line.
