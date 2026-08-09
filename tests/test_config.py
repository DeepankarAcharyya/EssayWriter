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
