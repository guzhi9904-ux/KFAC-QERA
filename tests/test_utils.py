import pytest

from qera_exp import utils


def test_progress_status_includes_percent_elapsed_and_eta(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(utils.time, "time", lambda: 110.0)
    assert utils.progress_status(2, 4, 100.0) == "2/4 (50.0%) elapsed=10s eta=10s"
    assert utils.progress_status(0, 4, 100.0) == "0/4 (0.0%) elapsed=10s eta=unknown"


def test_format_duration() -> None:
    assert utils.format_duration(3661) == "1h01m01s"


def test_heartbeat_rejects_nonpositive_interval(tmp_path) -> None:
    with pytest.raises(ValueError):
        with utils.heartbeat(tmp_path, "test", interval_seconds=0):
            pass
