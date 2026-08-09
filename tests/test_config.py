"""Settings resolution from the environment."""

import pytest

from essaywriter.config import Settings


@pytest.fixture(autouse=True)
def _no_dotenv_pollution(monkeypatch, tmp_path):
    """Keep `Settings.from_env`'s `load_dotenv()` from picking up a real `.env`."""
    monkeypatch.chdir(tmp_path)


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


def test_a_quality_threshold_outside_1_to_10_is_rejected():
    with pytest.raises(ValueError, match="quality_threshold"):
        Settings(quality_threshold=11)


def test_a_negative_max_revisions_is_rejected():
    with pytest.raises(ValueError, match="max_revisions"):
        Settings(max_revisions=-1)


def test_max_revisions_of_zero_is_accepted():
    assert Settings(max_revisions=0).max_revisions == 0
