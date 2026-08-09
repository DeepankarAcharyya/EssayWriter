"""CLI wiring: the essay reaches the user even when the report archive fails."""

from essaywriter import cli


def _no_env_pollution(monkeypatch):
    """Keep `Settings.from_env` within valid ranges regardless of the real shell."""
    monkeypatch.delenv("ESSAYWRITER_MAX_REVISIONS", raising=False)
    monkeypatch.delenv("ESSAYWRITER_QUALITY_THRESHOLD", raising=False)


def test_bare_run_prints_essay_and_writes_report(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    _no_env_pollution(monkeypatch)
    monkeypatch.setattr(cli, "write_essay", lambda **kwargs: "# Essay body\n")

    exit_code = cli.main(["A bare run topic"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "# Essay body" in captured.out
    reports = list((tmp_path / "output").glob("*.md"))
    assert len(reports) == 1
    assert reports[0].read_text(encoding="utf-8") == "# Essay body\n"
    assert "wrote output" in captured.err


def test_output_flag_writes_both_files(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    _no_env_pollution(monkeypatch)
    monkeypatch.setattr(cli, "write_essay", lambda **kwargs: "Essay text")
    out_path = tmp_path / "essay.md"

    exit_code = cli.main(["Some topic", "-o", str(out_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert out_path.read_text(encoding="utf-8") == "Essay text"
    reports = list((tmp_path / "output").glob("*.md"))
    assert len(reports) == 1
    assert reports[0].read_text(encoding="utf-8") == "Essay text"
    assert captured.out == ""


def test_report_write_failure_still_delivers_the_essay(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    _no_env_pollution(monkeypatch)
    monkeypatch.setattr(cli, "write_essay", lambda **kwargs: "Essay text")

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(cli, "write_report", _boom)

    exit_code = cli.main(["Some topic"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == "Essay text\n"
    assert "warning: could not write report" in captured.err
    assert not (tmp_path / "output").exists()
