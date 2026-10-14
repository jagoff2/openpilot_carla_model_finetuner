import sys

import pytest

from lane_dataset import cli, capture


def test_incomplete_capture_exits_unsuccessfully(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, "argv", ["lane_dataset", "--root", str(tmp_path), "benchmark"])
    monkeypatch.setattr(capture, "capture_development", lambda *args, **kwargs: {
        "manifest": {"error": {"type": "TimeoutError", "message": "missing aligned sensor frame"}}
    })
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "missing aligned sensor frame" in capsys.readouterr().out


def test_completed_diagnostic_is_not_a_process_failure(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, "argv", ["lane_dataset", "--root", str(tmp_path), "benchmark"])
    monkeypatch.setattr(capture, "capture_development", lambda *args, **kwargs: {
        "status": "quarantined_development", "manifest": {"qa": {"passed": False}}
    })
    cli.main()
    assert "quarantined_development" in capsys.readouterr().out
