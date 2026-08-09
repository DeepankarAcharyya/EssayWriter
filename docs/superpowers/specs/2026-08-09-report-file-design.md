# Report file per run

**Date:** 2026-08-09
**Status:** approved, not yet implemented

## Problem

A run's essay goes to stdout, or to `--output` when the flag is given. Nothing is kept.
A run whose output scrolled past is gone, and comparing two runs on the same topic means
remembering to name two files by hand.

## Goal

Every run leaves a dated report on disk under `output/`, named after the topic, without
taking away the stdout and `--output` behaviour that already exists.

## Design

### A module for naming and writing

Path construction and file writing go in a new `essaywriter/report.py`, not in `cli.py`.
`cli.py` is about argument handling and the run loop; slug rules and timestamp formats are
their own concern with their own tests.

```python
"""Naming and writing the report file for a run."""

OUTPUT_DIR = Path("output")


def slugify(topic: str, max_length: int = 60) -> str:
    """Reduce a topic to lowercase, hyphen-separated, path-safe text."""


def report_path(topic: str, when: datetime, directory: Path = OUTPUT_DIR) -> Path:
    """Where this topic's report belongs: <directory>/<slug>-<stamp>.md."""


def write_report(
    topic: str, essay: str, when: datetime, directory: Path = OUTPUT_DIR
) -> Path:
    """Write the essay to its report path, creating the directory, and return it."""
```

`when` is a parameter rather than a `datetime.now()` call inside the module. That is what
makes naming testable at a fixed instant without freezing the clock or monkeypatching
`datetime`. `cli.main` supplies `datetime.now()`.

### Slug rules

1. Lowercase the topic.
2. Replace every run of non-alphanumeric characters with a single hyphen. "Alphanumeric"
   means `str.isalnum()`, which is Unicode-aware — a non-Latin topic keeps its characters
   instead of reducing to nothing.
3. Strip leading and trailing hyphens.
4. Truncate to `max_length` (60) characters, then strip any trailing hyphen the cut left.
5. If nothing survives (a topic like `"???"`), return `essay`.

The timestamp is local time formatted `%Y%m%d-%H%M%S`, giving names that sort
chronologically within a topic:

```
output/the-economics-of-open-source-20260809-153042.md
```

### CLI flow

After `write_essay` returns, `main` writes the report, then keeps its existing behaviour:

| Case | Result |
| --- | --- |
| Every run | `output/<slug>-<stamp>.md` written; directory created if absent; `wrote output/….md` on stderr unless `--quiet` |
| `-o PATH` given | That copy is written too, exactly as today |
| No `-o` | The essay is still printed to stdout, exactly as today |

Piping is therefore unchanged, and every run leaves an archive whether or not `-o` was
used.

### Collisions

Two runs of the same topic within the same second produce the same name, and the second
overwrites the first. Accepted rather than adding a counter suffix: the granularity is
adequate for a CLI a human drives, and a uniquifier is complexity with no demonstrated
need.

### Testing

A new `tests/test_report.py`, working against pytest's `tmp_path`:

1. `slugify` lowercases and hyphenates punctuation runs.
2. `slugify` trims leading and trailing hyphens.
3. `slugify` truncates at 60 characters and leaves no trailing hyphen.
4. `slugify` returns `essay` for a topic with no alphanumeric characters.
5. `slugify` keeps non-Latin alphanumeric characters.
6. `report_path` renders the expected name at a fixed `datetime`.
7. `write_report` creates a missing directory, writes the essay verbatim, and returns the
   path it wrote.

`main`'s wiring stays untested, consistent with the rest of the CLI — the project has no
`tests/test_cli.py` and this change does not add one.

## Housekeeping

- `output/` added to `.gitignore`; reports are run artifacts, not source.
- README: the usage section documents the report file, the project layout gains `output/`,
  and the line about stdout is amended to mention the report.

## Out of scope

- A `--output-dir` flag. `output/` is fixed until there is a reason for it not to be.
- Any change to what the essay contains, or to the graph.
- Metadata in the report (topic header, score, run settings). The file is the essay.
