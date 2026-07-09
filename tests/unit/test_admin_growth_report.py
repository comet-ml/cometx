import datetime
import importlib
from unittest.mock import MagicMock


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


def _make_em_api():
    """MagicMock api mimicking verified EM endpoints (see task-C4-context.md)."""
    api = MagicMock()
    api._client.get_from_endpoint.return_value = {
        "projects": [
            {
                "projectId": "p1",
                "projectName": "proj1",
                "ownerUserName": "someone",
                "projectDescription": "",
                "workspaceName": "ws1",
                "numberOfExperiments": 2,
                "lastUpdated": 1700000000000,
                "public": False,
            },
            {
                "projectId": "p2",
                "projectName": "proj2",
                "ownerUserName": "someone",
                "projectDescription": "",
                "workspaceName": "ws1",
                "numberOfExperiments": 0,
                "lastUpdated": 1650000000000,
                "public": False,
            },
        ]
    }

    def get_experiments(ws, proj):
        if proj == "proj1":
            return [
                MagicMock(start_server_timestamp=1695000000000),
                MagicMock(start_server_timestamp=1690000000000),
            ]
        return []

    api.get_experiments.side_effect = get_experiments
    api.get_registry_model_names.return_value = ["modelA", "modelB"]

    def get_registry_model_versions(ws, name):
        return {"modelA": ["1.0.0", "1.0.1"], "modelB": ["1.0.0"]}[name]

    api.get_registry_model_versions.side_effect = get_registry_model_versions
    return api


def test_collect_em_creation_events_use_experiment_proxy():
    from cometx.cli.admin_growth_report import GrowthReporter

    api = _make_em_api()
    reporter = GrowthReporter(api, window="7d", units="month", platforms="em")
    events, _usage = reporter._collect_em(["ws1"])

    assert len(events) == 2
    assert all(e.kind == "em_project" for e in events)
    assert all(e.platform == "em" for e in events)
    assert all(e.workspace == "ws1" for e in events)
    assert not any(e.kind == "registry_model" for e in events)

    # Experiments must be fetched exactly ONCE per project (not once per
    # helper) — the list is fetched in _collect_em and shared between the
    # creation-proxy and the over-time series. 2 projects -> 2 calls.
    assert api.get_experiments.call_count == 2

    by_proj = {e.use_case: e for e in events}
    # proj1: earliest experiment start_server_timestamp used as creation proxy
    assert by_proj["proj1"].created == datetime.datetime.fromtimestamp(
        1690000000000 / 1000, tz=datetime.timezone.utc
    )
    # proj2: no experiments -> falls back to lastUpdated
    assert by_proj["proj2"].created == datetime.datetime.fromtimestamp(
        1650000000000 / 1000, tz=datetime.timezone.utc
    )


def test_collect_em_usage_metrics_experiment_count_and_registry_snapshot():
    from cometx.cli.admin_growth_report import GrowthReporter

    api = _make_em_api()
    reporter = GrowthReporter(api, window="7d", units="month", platforms="em")
    _events, usage = reporter._collect_em(["ws1"])

    exp_metrics = [m for m in usage if m.metric == "EXPERIMENT_COUNT"]

    proj1_metric = next(m for m in exp_metrics if m.project == "proj1")
    assert proj1_metric.value == 2
    assert proj1_metric.platform == "em" and proj1_metric.workspace == "ws1"
    assert proj1_metric.series  # non-empty over-time series

    proj2_metric = next(m for m in exp_metrics if m.project == "proj2")
    assert proj2_metric.value == 0

    ws_total_metric = next(m for m in exp_metrics if m.project is None)
    assert ws_total_metric.value == 2
    assert ws_total_metric.workspace == "ws1"

    reg_models = next(m for m in usage if m.metric == "REGISTRY_MODELS")
    assert reg_models.value == 2
    assert reg_models.series is None
    assert reg_models.workspace == "ws1" and reg_models.platform == "em"

    reg_versions = next(m for m in usage if m.metric == "REGISTRY_VERSIONS")
    assert reg_versions.value == 3
    assert reg_versions.series is None

    # registry metrics are never CreationEvents / never a use case
    assert not any(m.metric.startswith("REGISTRY") and m.series for m in usage)


def test_collect_em_respects_limit_on_workspaces():
    from cometx.cli.admin_growth_report import GrowthReporter

    api = _make_em_api()
    reporter = GrowthReporter(api, window="7d", units="month", platforms="em", limit=1)
    events, usage = reporter._collect_em(["ws1", "ws2"])

    assert all(e.workspace == "ws1" for e in events)
    assert all(m.workspace == "ws1" for m in usage)
    # only one workspace's projects endpoint should have been queried
    assert api._client.get_from_endpoint.call_count == 1


def test_collect_em_skips_bad_project_and_continues(capsys):
    from cometx.cli.admin_growth_report import GrowthReporter

    api = _make_em_api()

    def get_experiments(ws, proj):
        if proj == "proj1":
            raise RuntimeError("boom")
        return []

    api.get_experiments.side_effect = get_experiments
    reporter = GrowthReporter(api, window="7d", units="month", platforms="em")
    events, usage = reporter._collect_em(["ws1"])

    # proj1 blew up but proj2 (and registry metrics) still collected
    assert any(e.use_case == "proj2" for e in events)
    assert not any(e.use_case == "proj1" for e in events)
    assert any(m.metric == "REGISTRY_MODELS" for m in usage)
