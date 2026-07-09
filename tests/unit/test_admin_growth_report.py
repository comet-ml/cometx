import datetime
import importlib


def test_creation_event_and_window():
    from cometx.cli.admin_growth_report import CreationEvent, Window

    ev = CreationEvent(
        "opik",
        "wsA",
        "proj-1",
        "project",
        datetime.datetime(2026, 6, 1, tzinfo=datetime.timezone.utc),
    )
    assert ev.kind == "project" and ev.workspace == "wsA"
    w = Window(
        datetime.datetime(2026, 6, 1, tzinfo=datetime.timezone.utc),
        datetime.datetime(2026, 7, 1, tzinfo=datetime.timezone.utc),
    )
    assert w.units == "month"


def test_growth_report_delegate_exists():
    m = importlib.import_module("cometx.cli.admin_growth_report")
    assert hasattr(m, "generate_growth_report")


def test_growth_report_action_registered_in_admin():
    # admin.py builds an ACTION subparser; growth-report must be one of the choices
    from cometx.cli import admin as admin_mod

    src = admin_mod.__doc__ or ""  # noqa: F841
    # smoke: the delegate is imported by admin.py
    import inspect

    admin_src = inspect.getsource(admin_mod)
    assert "generate_growth_report" in admin_src
    assert "growth-report" in admin_src


def _ev(y, m, d, ws="w", uc="p"):
    from cometx.cli.admin_growth_report import CreationEvent

    return CreationEvent(
        "opik",
        ws,
        uc,
        "project",
        datetime.datetime(y, m, d, tzinfo=datetime.timezone.utc),
    )


def _win():
    from cometx.cli.admin_growth_report import Window

    return Window(
        datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
        datetime.datetime(2026, 12, 31, tzinfo=datetime.timezone.utc),
        "month",
    )


def test_bucket_and_continuous_zero_fill():
    from cometx.cli.admin_growth_report import bucket_events, continuous_series

    evs = [_ev(2026, 1, 5), _ev(2026, 1, 20), _ev(2026, 3, 2)]
    counts = bucket_events(evs, _win(), "month")
    assert counts["2026-01"] == 2 and counts["2026-03"] == 1
    series = continuous_series(counts, "month")
    keys = [k for k, _ in series]
    assert keys == ["2026-01", "2026-02", "2026-03"]  # Feb zero-filled
    assert dict(series)["2026-02"] == 0


def test_cumulative_and_growth():
    from cometx.cli.admin_growth_report import cumulative, growth_stats

    cum = cumulative([("2026-01", 2), ("2026-02", 0), ("2026-03", 1)])
    assert [v for _, v in cum] == [2, 2, 3]
    # 2 before window, 3 inside -> pct = 3/2*100 = 150
    evs = [
        _ev(2025, 12, 1),
        _ev(2025, 12, 2),
        _ev(2026, 2, 1),
        _ev(2026, 2, 2),
        _ev(2026, 3, 1),
    ]
    g = growth_stats(evs, _win(), "month")
    assert g["new_in_window"] == 3 and g["total"] == 5 and round(g["pct_growth"]) == 150
