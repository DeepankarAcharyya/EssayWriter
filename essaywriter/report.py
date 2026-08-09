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
    path.write_text(essay, encoding="utf-8")
    return path
