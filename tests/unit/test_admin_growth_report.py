import datetime
import importlib
from unittest.mock import MagicMock, patch


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


def _make_opik_api():
    """MagicMock api with a REAL dict `.config` (see task-C5-context.md) so
    host/api_key resolution works without touching MagicMock magic."""
    api = MagicMock()
    api.config = {
        "comet.api_key": "KEY",
        "comet.url_override": "https://example.com/",
    }
    return api


def _make_opik_project(id_, name, created_at):
    project = MagicMock()
    project.id = id_
    project.name = name
    project.created_at = created_at
    return project


def _make_opik_metrics_response(datapoints):
    """datapoints: list of (time, value) -> a fake get_project_metrics resp."""
    result = MagicMock()
    result.data = [MagicMock(time=t, value=v) for t, v in datapoints]
    resp = MagicMock()
    resp.results = [result]
    return resp


@patch(
    "cometx.cli.smoke_test.get_opik_config",
    return_value="https://example.com/opik/api/",
)
@patch("opik.Opik")
def test_collect_opik_creation_events_and_span_count_usage(mock_opik_ctor, _mock_host):
    from cometx.cli.admin_growth_report import GrowthReporter

    created1 = datetime.datetime(2026, 1, 5, tzinfo=datetime.timezone.utc)
    created2 = datetime.datetime(2026, 2, 10, tzinfo=datetime.timezone.utc)
    proj1 = _make_opik_project("id1", "proj1", created1)
    proj2 = _make_opik_project("id2", "proj2", created2)

    fake_page = MagicMock()
    fake_page.content = [proj1, proj2]
    fake_page.total = 2

    mock_client = MagicMock()
    mock_client.rest_client.projects.find_projects.return_value = fake_page

    def get_project_metrics(project_id, **kwargs):
        if project_id == "id1":
            return _make_opik_metrics_response(
                [
                    (datetime.datetime(2026, 1, 10, tzinfo=datetime.timezone.utc), 5),
                    (datetime.datetime(2026, 1, 20, tzinfo=datetime.timezone.utc), 3),
                ]
            )
        return _make_opik_metrics_response(
            [(datetime.datetime(2026, 2, 15, tzinfo=datetime.timezone.utc), 7)]
        )

    mock_client.rest_client.projects.get_project_metrics.side_effect = (
        get_project_metrics
    )
    mock_opik_ctor.return_value = mock_client

    api = _make_opik_api()
    reporter = GrowthReporter(api, window="7d", units="month", platforms="opik")
    events, usage = reporter._collect_opik(["ws1"])

    # one opik_project CreationEvent per project, created == created_at
    assert len(events) == 2
    assert all(e.kind == "opik_project" for e in events)
    assert all(e.platform == "opik" for e in events)
    assert all(e.workspace == "ws1" for e in events)
    by_uc = {e.use_case: e for e in events}
    assert by_uc["proj1"].created == created1
    assert by_uc["proj2"].created == created2

    # opik.Opik constructed with resolved host/api_key for the workspace
    mock_opik_ctor.assert_any_call(
        workspace="ws1", api_key="KEY", host="https://example.com/opik/api/"
    )

    span_metrics = [m for m in usage if m.metric == "SPAN_COUNT"]

    proj1_metric = next(m for m in span_metrics if m.project == "proj1")
    assert proj1_metric.platform == "opik" and proj1_metric.workspace == "ws1"
    assert proj1_metric.value == 8  # sum of proj1 datapoints
    assert proj1_metric.series  # non-empty over-time series

    proj2_metric = next(m for m in span_metrics if m.project == "proj2")
    assert proj2_metric.value == 7

    ws_total_metric = next(m for m in span_metrics if m.project is None)
    assert ws_total_metric.value == 15  # 8 + 7
    assert ws_total_metric.workspace == "ws1"
    assert ws_total_metric.series


@patch(
    "cometx.cli.smoke_test.get_opik_config",
    return_value="https://example.com/opik/api/",
)
@patch("opik.Opik")
def test_collect_opik_respects_limit_on_workspaces(mock_opik_ctor, _mock_host):
    from cometx.cli.admin_growth_report import GrowthReporter

    proj = _make_opik_project(
        "id1", "proj1", datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    )
    fake_page = MagicMock()
    fake_page.content = [proj]
    fake_page.total = 1

    mock_client = MagicMock()
    mock_client.rest_client.projects.find_projects.return_value = fake_page
    mock_client.rest_client.projects.get_project_metrics.return_value = (
        _make_opik_metrics_response([])
    )
    mock_opik_ctor.return_value = mock_client

    api = _make_opik_api()
    reporter = GrowthReporter(
        api, window="7d", units="month", platforms="opik", limit=1
    )
    events, usage = reporter._collect_opik(["ws1", "ws2"])

    assert all(e.workspace == "ws1" for e in events)
    assert all(m.workspace == "ws1" for m in usage)
    assert mock_opik_ctor.call_count == 1


@patch(
    "cometx.cli.smoke_test.get_opik_config",
    return_value="https://example.com/opik/api/",
)
@patch("opik.Opik")
def test_collect_opik_skips_bad_workspace_and_continues(
    mock_opik_ctor, _mock_host, capsys
):
    from cometx.cli.admin_growth_report import GrowthReporter

    proj = _make_opik_project(
        "id1", "proj1", datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    )
    fake_page_ok = MagicMock()
    fake_page_ok.content = [proj]
    fake_page_ok.total = 1

    good_client = MagicMock()
    good_client.rest_client.projects.find_projects.return_value = fake_page_ok
    good_client.rest_client.projects.get_project_metrics.return_value = (
        _make_opik_metrics_response(
            [(datetime.datetime(2026, 1, 2, tzinfo=datetime.timezone.utc), 4)]
        )
    )

    def opik_ctor(workspace, **kwargs):
        if workspace == "ws_bad":
            raise RuntimeError("bad auth")
        return good_client

    mock_opik_ctor.side_effect = opik_ctor

    api = _make_opik_api()
    reporter = GrowthReporter(api, window="7d", units="month", platforms="opik")
    events, usage = reporter._collect_opik(["ws_bad", "ws_good"])

    assert not any(e.workspace == "ws_bad" for e in events)
    assert any(e.workspace == "ws_good" for e in events)
    assert any(m.workspace == "ws_good" and m.metric == "SPAN_COUNT" for m in usage)


def test_collect_opik_missing_dependency_returns_empty(monkeypatch):
    import builtins

    from cometx.cli.admin_growth_report import GrowthReporter

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "opik":
            raise ImportError("no opik")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    api = _make_opik_api()
    reporter = GrowthReporter(api, window="7d", units="month", platforms="opik")
    events, usage = reporter._collect_opik(["ws1"])

    assert events == []
    assert usage == []
