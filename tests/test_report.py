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
