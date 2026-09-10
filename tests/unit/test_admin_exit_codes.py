# -*- coding: utf-8 -*-
"""Every error path in `cometx admin` must exit non-zero.

`admin` is run unattended -- a monthly growth-report cron, a CI usage report --
where the exit code is the only signal anything reads. Printing `ERROR: ...`
and returning 0 reported success while producing nothing, which is the worst
shape for a scheduled job: it fails silently and the pipeline downstream keeps
whatever stale data it already had.

These tests pin the exit code for each subcommand's failure paths, and pin the
success path to 0 so the fix cannot overshoot into failing a healthy run.
"""

from unittest.mock import MagicMock, patch

import pytest


def _run(argv):
    """Invoke the admin CLI, returning the exit code (0 when it returns
    normally, since a clean run raises no SystemExit)."""
    import cometx.cli.admin as admin_mod

    with patch.object(admin_mod, "API", MagicMock()):
        try:
            admin_mod.main(argv)
        except SystemExit as exc:
            return exc.code
    return 0


def test_usage_report_generation_failure_exits_nonzero():
    import cometx.cli.admin as admin_mod

    with patch.object(
        admin_mod, "generate_usage_report", side_effect=RuntimeError("boom")
    ):
        assert _run(["usage-report", "ws/proj"]) == 1


def test_usage_report_without_workspace_exits_nonzero():
    assert _run(["usage-report"]) == 1


def test_gpu_report_generation_failure_exits_nonzero():
    import cometx.cli.admin as admin_mod

    with patch.object(admin_mod, "gpu_report_main", side_effect=RuntimeError("boom")):
        assert _run(["gpu-report", "ws/proj"]) == 1


def test_gpu_report_without_workspace_exits_nonzero():
    assert _run(["gpu-report"]) == 1


def test_optimizer_report_failure_exits_nonzero():
    """A falsy result is a failure, not an empty success."""
    import cometx.cli.admin as admin_mod

    with patch.object(admin_mod, "generate_json_report", return_value=None):
        assert _run(["optimizer-report", "opt-id"]) == 1


def test_optimizer_report_exception_exits_nonzero():
    import cometx.cli.admin as admin_mod

    with patch.object(
        admin_mod, "generate_json_report", side_effect=RuntimeError("boom")
    ):
        assert _run(["optimizer-report", "opt-id"]) == 1


def test_an_unexpected_error_exits_nonzero():
    """The outer handler: anything the per-action handlers did not catch."""
    import cometx.cli.admin as admin_mod

    with patch.object(admin_mod, "API", side_effect=RuntimeError("no api key")):
        with pytest.raises(SystemExit) as excinfo:
            admin_mod.main(["chargeback-report"])
    assert excinfo.value.code == 1


def test_control_c_exits_130():
    """128 + SIGINT, the shell convention for an interrupted command. Exiting 0
    told a scheduler the run succeeded."""
    import cometx.cli.admin as admin_mod

    with patch.object(admin_mod, "API", side_effect=KeyboardInterrupt):
        with pytest.raises(SystemExit) as excinfo:
            admin_mod.main(["chargeback-report"])
    assert excinfo.value.code == 130


def test_a_broken_exception_str_still_exits_nonzero():
    """`comet_ml.exceptions.NotFound.__str__` returns None when the 404 body is
    not JSON. A bare `str(exc)` would raise inside the handler, skipping the
    exit and letting the command report success -- the exact failure the
    growth-report path was already hardened against."""
    import cometx.cli.admin as admin_mod

    class BrokenStr(Exception):
        def __str__(self):
            return None

    with patch.object(admin_mod, "API", side_effect=BrokenStr()):
        with pytest.raises(SystemExit) as excinfo:
            admin_mod.main(["chargeback-report"])
    assert excinfo.value.code == 1


def test_a_successful_run_still_exits_zero(tmp_path, monkeypatch):
    """The guard must not overshoot: a healthy run reports success."""
    import cometx.cli.admin as admin_mod

    monkeypatch.chdir(tmp_path)
    with patch.object(admin_mod, "fetch_chargeback_report", return_value={"users": {}}):
        assert _run(["chargeback-report"]) == 0
    assert (tmp_path / "comet-chargeback-report.json").exists()
