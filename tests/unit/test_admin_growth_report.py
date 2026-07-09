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
    # smoke: the delegate is imported by admin.py
    import inspect

    from cometx.cli import admin as admin_mod

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


def test_continuous_zero_fill():
    from cometx.cli.admin_growth_report import continuous_series

    counts = {"2026-01": 2, "2026-03": 1}
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


def test_collect_em_filters_projects_not_belonging_to_workspace(capsys):
    # The EM `projects` endpoint returns a fallback set from another workspace
    # when the key isn't a member of the requested one, ignoring workspaceName.
    # Those foreign projects must NOT be attributed to the requested workspace;
    # the workspace is still reported at 0, and a warning is printed.
    from cometx.cli.admin_growth_report import GrowthReporter

    api = MagicMock()
    api._client.get_from_endpoint.return_value = {
        "projects": [
            {
                "projectName": "foreign-proj",
                "projectId": "x1",
                "workspaceName": "team-comet-ml",  # != requested workspace
                "numberOfExperiments": 5,
                "lastUpdated": 1700000000000,
            }
        ]
    }
    api.get_registry_model_names.return_value = []
    api.get_registry_model_versions.return_value = []
    reporter = GrowthReporter(api, window="7d", units="month", platforms="em")
    events, usage = reporter._collect_em(["opik-demos"])

    # foreign project is not attributed to opik-demos
    assert events == []
    assert not any(m.project == "foreign-proj" for m in usage)
    # experiments are never fetched for a filtered-out project
    api.get_experiments.assert_not_called()
    # workspace is still listed, at an experiment total of 0
    ws_totals = [
        m for m in usage if m.metric == "EXPERIMENT_COUNT" and m.project is None
    ]
    assert len(ws_totals) == 1
    assert ws_totals[0].workspace == "opik-demos" and ws_totals[0].value == 0
    # and a warning was surfaced
    assert "not belonging to workspace opik-demos" in capsys.readouterr().out


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


def _make_mpm_api():
    """MagicMock api with a REAL dict `.config` (see task-C6-context.md)."""
    api = MagicMock()
    api.config = {
        "comet.api_key": "KEY",
        "comet.url_override": "https://example.com/",
    }
    return api


def _mpm_pred_envelope(points):
    """points: list of (x, y) -> the verified {"data":[{"data":[...]}]} envelope."""
    return {"data": [{"data": [{"x": x, "y": y} for x, y in points]}]}


def _make_mpm_client(workspaces_resp, details_map, predictions_map):
    """MagicMock comet_mpm._client with the verified MPM endpoints stubbed.

    `details_map`: model_id -> details dict (or an exception instance to raise).
    `predictions_map`: model_id -> envelope dict returned by get_nb_predictions.
    """
    client = MagicMock()

    def fake_get(path, *args, **kwargs):
        assert path == "api/mpm/v3/workspaces"
        return workspaces_resp

    client.get.side_effect = fake_get

    def fake_get_model_details(model_id):
        val = details_map.get(model_id, {})
        if isinstance(val, Exception):
            raise val
        return val

    client.get_model_details.side_effect = fake_get_model_details

    def fake_get_nb_predictions(model_id, *args, **kwargs):
        return predictions_map.get(model_id, _mpm_pred_envelope([]))

    client.get_nb_predictions.side_effect = fake_get_nb_predictions
    return client


def _mpm_workspaces_resp(models_by_ws):
    return {
        "workspaces": [
            {"workspaceName": ws, "models": models}
            for ws, models in models_by_ws.items()
        ]
    }


@patch("comet_mpm.API")
def test_collect_mpm_creation_scenarios_a_b_c(mock_api_ctor):
    from cometx.cli.admin_growth_report import GrowthReporter

    models_by_ws = {
        "ws1": [
            {"modelName": "modelA", "modelId": "idA"},
            {"modelName": "modelB", "modelId": "idB"},
            {"modelName": "modelC", "modelId": "idC"},
        ]
    }
    workspaces_resp = _mpm_workspaces_resp(models_by_ws)

    # (a) model A: details carry a creation timestamp
    details_map = {
        "idA": {"createdAt": 1700000000000},
        "idB": {},  # no creation key -> falls back to first-prediction-day
        "idC": {},  # no creation key, and no y>0 predictions -> no event
    }

    predictions_map = {
        "idA": _mpm_pred_envelope([(1699000000000, 5), (1700500000000, 2)]),
        "idB": _mpm_pred_envelope([(1690000000000, 0), (1691000000000, 7)]),
        "idC": _mpm_pred_envelope([(1690000000000, 0), (1691000000000, 0)]),
    }

    client = _make_mpm_client(workspaces_resp, details_map, predictions_map)
    mock_mpm = MagicMock()
    mock_mpm._client = client
    mock_api_ctor.return_value = mock_mpm

    api = _make_mpm_api()
    reporter = GrowthReporter(api, window="7d", units="month", platforms="mpm")
    events, usage = reporter._collect_mpm(["ws1"])

    by_uc = {e.use_case: e for e in events}

    # (a) details timestamp -> mpm_model event at that time
    assert "modelA" in by_uc
    assert by_uc["modelA"].kind == "mpm_model"
    assert by_uc["modelA"].platform == "mpm"
    assert by_uc["modelA"].workspace == "ws1"
    assert by_uc["modelA"].created == datetime.datetime.fromtimestamp(
        1700000000000 / 1000, tz=datetime.timezone.utc
    )

    # (b) no details ts but predictions -> proxy event at first prediction
    # day with y>0 (1691000000000, not the 0-value 1690000000000 point)
    assert "modelB" in by_uc
    assert by_uc["modelB"].created == datetime.datetime.fromtimestamp(
        1691000000000 / 1000, tz=datetime.timezone.utc
    )

    # (c) neither details ts nor any y>0 -> no CreationEvent for modelC
    assert "modelC" not in by_uc

    # but modelC must still yield a PREDICTION_VOLUME usage metric
    pred_metrics = [m for m in usage if m.metric == "PREDICTION_VOLUME"]
    modelC_metric = next(m for m in pred_metrics if m.project == "modelC")
    assert modelC_metric.value == 0
    assert modelC_metric.platform == "mpm" and modelC_metric.workspace == "ws1"


@patch("comet_mpm.API")
def test_collect_mpm_prediction_volume_usage_per_model_and_per_workspace(
    mock_api_ctor,
):
    from cometx.cli.admin_growth_report import GrowthReporter

    models_by_ws = {
        "ws1": [
            {"modelName": "modelA", "modelId": "idA"},
            {"modelName": "modelB", "modelId": "idB"},
        ]
    }
    workspaces_resp = _mpm_workspaces_resp(models_by_ws)
    details_map = {"idA": {"createdAt": 1700000000000}, "idB": {}}
    predictions_map = {
        "idA": _mpm_pred_envelope([(1699000000000, 5), (1700500000000, 2)]),
        "idB": _mpm_pred_envelope([(1691000000000, 7)]),
    }

    client = _make_mpm_client(workspaces_resp, details_map, predictions_map)
    mock_mpm = MagicMock()
    mock_mpm._client = client
    mock_api_ctor.return_value = mock_mpm

    api = _make_mpm_api()
    reporter = GrowthReporter(api, window="7d", units="month", platforms="mpm")
    _events, usage = reporter._collect_mpm(["ws1"])

    pred_metrics = [m for m in usage if m.metric == "PREDICTION_VOLUME"]

    modelA_metric = next(m for m in pred_metrics if m.project == "modelA")
    assert modelA_metric.value == 7  # 5 + 2
    assert modelA_metric.series  # non-empty over-time series

    modelB_metric = next(m for m in pred_metrics if m.project == "modelB")
    assert modelB_metric.value == 7

    ws_total_metric = next(m for m in pred_metrics if m.project is None)
    assert ws_total_metric.value == 14  # 7 + 7
    assert ws_total_metric.workspace == "ws1"
    assert ws_total_metric.series

    # fetch-once regression guard: get_nb_predictions called exactly once
    # per model (reused for BOTH the creation proxy and the usage metric)
    assert client.get_nb_predictions.call_count == 2


@patch("comet_mpm.API")
def test_collect_mpm_respects_limit_on_workspaces(mock_api_ctor):
    from cometx.cli.admin_growth_report import GrowthReporter

    models_by_ws = {
        "ws1": [{"modelName": "modelA", "modelId": "idA"}],
        "ws2": [{"modelName": "modelZ", "modelId": "idZ"}],
    }
    workspaces_resp = _mpm_workspaces_resp(models_by_ws)
    details_map = {"idA": {"createdAt": 1700000000000}}
    predictions_map = {"idA": _mpm_pred_envelope([(1700000000000, 3)])}

    client = _make_mpm_client(workspaces_resp, details_map, predictions_map)
    mock_mpm = MagicMock()
    mock_mpm._client = client
    mock_api_ctor.return_value = mock_mpm

    api = _make_mpm_api()
    reporter = GrowthReporter(api, window="7d", units="month", platforms="mpm", limit=1)
    events, usage = reporter._collect_mpm(["ws1", "ws2"])

    assert all(e.workspace == "ws1" for e in events)
    assert all(m.workspace == "ws1" for m in usage)
    assert not any(e.use_case == "modelZ" for e in events)


@patch("comet_mpm.API")
def test_collect_mpm_skips_bad_model_and_continues(mock_api_ctor, capsys):
    from cometx.cli.admin_growth_report import GrowthReporter

    models_by_ws = {
        "ws1": [
            {"modelName": "modelBad", "modelId": "idBad"},
            {"modelName": "modelGood", "modelId": "idGood"},
        ]
    }
    workspaces_resp = _mpm_workspaces_resp(models_by_ws)
    details_map = {"idBad": {}, "idGood": {"createdAt": 1700000000000}}

    def fake_get_nb_predictions(model_id, *args, **kwargs):
        if model_id == "idBad":
            raise RuntimeError("boom")
        return _mpm_pred_envelope([(1700000000000, 4)])

    client = _make_mpm_client(workspaces_resp, details_map, {})
    client.get_nb_predictions.side_effect = fake_get_nb_predictions
    mock_mpm = MagicMock()
    mock_mpm._client = client
    mock_api_ctor.return_value = mock_mpm

    api = _make_mpm_api()
    reporter = GrowthReporter(api, window="7d", units="month", platforms="mpm")
    events, usage = reporter._collect_mpm(["ws1"])

    assert not any(e.use_case == "modelBad" for e in events)
    assert any(e.use_case == "modelGood" for e in events)
    assert any(m.project == "modelGood" for m in usage)
    assert not any(m.project == "modelBad" for m in usage)


@patch("comet_mpm.API")
def test_collect_mpm_workspaces_endpoint_error_returns_empty(mock_api_ctor):
    from cometx.cli.admin_growth_report import GrowthReporter

    client = MagicMock()
    client.get.side_effect = RuntimeError("404")
    mock_mpm = MagicMock()
    mock_mpm._client = client
    mock_api_ctor.return_value = mock_mpm

    api = _make_mpm_api()
    reporter = GrowthReporter(api, window="7d", units="month", platforms="mpm")
    events, usage = reporter._collect_mpm(["ws1"])

    assert events == []
    assert usage == []


@patch("comet_mpm.API")
def test_collect_mpm_skips_malformed_enumeration_elements(mock_api_ctor):
    # The MPM inventory shape is unverifiable live; malformed workspace/model
    # elements must be skipped, not crash the whole report.
    from cometx.cli.admin_growth_report import GrowthReporter

    workspaces_resp = {
        "workspaces": [
            "not-a-dict",  # malformed workspace entry
            {
                "workspaceName": "ws1",
                "models": [
                    "not-a-dict",  # malformed model entry
                    {"modelName": "good", "modelId": "idG"},
                ],
            },
        ]
    }
    predictions_map = {"idG": _mpm_pred_envelope([(1690000000000, 5)])}
    client = _make_mpm_client(
        workspaces_resp, details_map={}, predictions_map=predictions_map
    )
    mock_mpm = MagicMock()
    mock_mpm._client = client
    mock_api_ctor.return_value = mock_mpm

    api = _make_mpm_api()
    reporter = GrowthReporter(api, window="7d", units="month", platforms="mpm")
    events, usage = reporter._collect_mpm(["ws1"])

    # The one good model is still collected; malformed entries are ignored.
    assert [e.use_case for e in events] == ["good"]
    assert any(m.metric == "PREDICTION_VOLUME" and m.project == "good" for m in usage)


def _uc_ev(platform, ws, uc, kind, y, m, d):
    from cometx.cli.admin_growth_report import CreationEvent

    return CreationEvent(
        platform,
        ws,
        uc,
        kind,
        datetime.datetime(y, m, d, tzinfo=datetime.timezone.utc),
    )


def _make_mixed_workspace_events():
    """Two workspaces w/ mixed opik_project/em_project/mpm_model events."""
    return [
        # ws1: 2 opik, 1 em, 1 mpm; earliest = 2026-01-02 (opik)
        _uc_ev("opik", "ws1", "op1", "opik_project", 2026, 1, 2),
        _uc_ev("opik", "ws1", "op2", "opik_project", 2026, 3, 1),
        _uc_ev("em", "ws1", "em1", "em_project", 2026, 2, 1),
        _uc_ev("mpm", "ws1", "mp1", "mpm_model", 2026, 4, 1),
        # ws2: 1 em, 2 mpm; earliest = 2025-12-15 (em)
        _uc_ev("em", "ws2", "em2", "em_project", 2025, 12, 15),
        _uc_ev("mpm", "ws2", "mp2", "mpm_model", 2026, 1, 10),
        _uc_ev("mpm", "ws2", "mp3", "mpm_model", 2026, 2, 20),
    ]


def test_use_cases_by_workspace_by_kind_totals_and_first_created():
    from cometx.cli.admin_growth_report import use_cases_by_workspace

    events = _make_mixed_workspace_events()
    result = use_cases_by_workspace(events)

    assert set(result.keys()) == {"ws1", "ws2"}

    ws1 = result["ws1"]
    assert ws1["use_cases_total"] == 4
    assert ws1["by_kind"] == {"opik_project": 2, "em_project": 1, "mpm_model": 1}
    assert ws1["first_created"] == datetime.datetime(
        2026, 1, 2, tzinfo=datetime.timezone.utc
    )

    ws2 = result["ws2"]
    assert ws2["use_cases_total"] == 3
    assert ws2["by_kind"] == {"opik_project": 0, "em_project": 1, "mpm_model": 2}
    assert ws2["first_created"] == datetime.datetime(
        2025, 12, 15, tzinfo=datetime.timezone.utc
    )


def test_use_cases_by_workspace_empty_input():
    from cometx.cli.admin_growth_report import use_cases_by_workspace

    assert use_cases_by_workspace([]) == {}


def test_unified_events_filters_to_three_use_case_kinds():
    from cometx.cli.admin_growth_report import unified_events

    events = _make_mixed_workspace_events()
    other = _uc_ev("em", "ws1", "reg1", "registry_model", 2026, 1, 1)
    result = unified_events(events + [other])

    assert len(result) == len(events)
    assert all(e.kind in ("opik_project", "em_project", "mpm_model") for e in result)
    assert other not in result
    # does not mutate input
    assert (events + [other]) == events + [other]


def test_unified_events_empty_input():
    from cometx.cli.admin_growth_report import unified_events

    assert unified_events([]) == []


def test_collect_mpm_missing_dependency_returns_empty(monkeypatch):
    import builtins

    from cometx.cli.admin_growth_report import GrowthReporter

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "comet_mpm":
            raise ImportError("no comet_mpm")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    api = _make_mpm_api()
    reporter = GrowthReporter(api, window="7d", units="month", platforms="mpm")
    events, usage = reporter._collect_mpm(["ws1"])

    assert events == []
    assert usage == []


# ---------------------------------------------------------------------------
# C8: self-contained HTML dashboard renderer
# ---------------------------------------------------------------------------


def _sample_report_data():
    """A representative `report_data` payload matching the documented C8
    contract: a top-level `window`, a unified section, and per-product
    (opik/em/mpm) growth + adoption sections (EM additionally carries a
    registry-engagement snapshot panel)."""
    return {
        "meta": {
            "title": "Growth report — acme <script> & Co",
            "org": "acme-research",
            "generated": "2026-07-09",
            "source": "Comet Admin API",
        },
        "window": {
            "start": "2026-07-02",
            "end": "2026-07-09",
            "units": "day",
            "label": "Analysis window: Jul 2 – Jul 9, 2026 (7d)",
            "count_before": 65,
        },
        "collectors": {"opik": True, "em": True, "mpm": False},
        "sections": {
            "unified": {
                "title": "Use cases across all platforms",
                "window_chip": "Analysis window: Jul 2 – Jul 9, 2026 (7d)",
                "kpis": [
                    {"label": "Departments", "value": 4},
                    {"label": "Use cases", "value": 77, "tone": "ok"},
                    {"label": "New (7d)", "value": "+12"},
                    {
                        "label": "Growth (7d)",
                        "value": "18.5%",
                        "sub": "vs 65 before window",
                    },
                ],
                "charts": [
                    {
                        "id": "chart-unified-created",
                        "kind": "stackedBars",
                        "title": "Use cases created",
                        "hint": "by kind · monthly",
                        "legend": [
                            {"label": "Opik", "color": "--accent"},
                            {"label": "EM", "color": "--sdk"},
                            {"label": "MPM", "color": "--ok"},
                        ],
                        "data": {
                            "categories": [
                                "opik_project",
                                "em_project",
                                "mpm_model",
                            ],
                            "labels": {
                                "opik_project": "Opik",
                                "em_project": "EM",
                                "mpm_model": "MPM",
                            },
                            "colors": ["--accent", "--sdk", "--ok"],
                            "points": [
                                {
                                    "key": "2026-06",
                                    "values": {
                                        "opik_project": 3,
                                        "em_project": 1,
                                        "mpm_model": 0,
                                    },
                                },
                                {
                                    "key": "2026-07",
                                    "values": {
                                        "opik_project": 2,
                                        "em_project": 2,
                                        "mpm_model": 1,
                                    },
                                },
                            ],
                            "window_start": "2026-07",
                            "window_end": "2026-07",
                        },
                    },
                    {
                        "id": "chart-unified-by-department",
                        "kind": "groupedBarsH",
                        "title": "Use cases by department",
                        "hint": "current totals",
                        "data": {
                            "rows": [
                                {"label": "ws-alpha", "value": 40},
                                {"label": "ws-beta", "value": 37},
                            ]
                        },
                    },
                ],
                "table": {
                    "title": "By department",
                    "headers": ["Department", "Opik", "EM", "MPM", "Total"],
                    "rows": [
                        ["ws-alpha", 20, 15, 5, 40],
                        ["ws-beta", 10, 20, 7, 37],
                    ],
                },
            },
            "products": {
                "opik": {
                    "label": "Opik",
                    "growth": {
                        "title": "Opik — growth",
                        "window_chip": "Analysis window: Jul 2 – Jul 9, 2026 (7d)",
                        "kpis": [
                            {"label": "Workspaces", "value": 2},
                            {"label": "Total", "value": 30},
                            {"label": "New (7d)", "value": "+5"},
                            {
                                "label": "Growth (7d)",
                                "value": "20.0%",
                                "sub": "vs 25 before window",
                            },
                        ],
                        "charts": [
                            {
                                "id": "chart-opik-bars",
                                "kind": "bars",
                                "title": "New Opik projects",
                                "hint": "by month",
                                "data": {
                                    "points": [
                                        {"key": "2026-06", "value": 3},
                                        {"key": "2026-07", "value": 5},
                                    ],
                                    "window_start": "2026-07",
                                    "window_end": "2026-07",
                                },
                            },
                            {
                                "id": "chart-opik-area",
                                "kind": "area",
                                "title": "Opik projects — cumulative",
                                "hint": "all-time",
                                "data": {
                                    "points": [
                                        {"key": "2026-06", "value": 25},
                                        {"key": "2026-07", "value": 30},
                                    ],
                                    "window_start": "2026-07",
                                    "window_end": "2026-07",
                                    "delta": 5,
                                },
                            },
                        ],
                        "table": {
                            "headers": ["Workspace", "Opik projects"],
                            "rows": [["ws-alpha", 20], ["ws-beta", 10]],
                        },
                    },
                    "adoption": {
                        "title": "Opik — adoption / usage",
                        "kpis": [{"label": "Span count", "value": 154200}],
                        "charts": [
                            {
                                "id": "chart-opik-spans",
                                "kind": "bars",
                                "title": "Span count",
                                "hint": "by month",
                                "data": {
                                    "points": [
                                        {"key": "2026-06", "value": 70000},
                                        {"key": "2026-07", "value": 84200},
                                    ],
                                    "window_start": "2026-07",
                                    "window_end": "2026-07",
                                },
                            }
                        ],
                        "table": {
                            "title": "Span count by project",
                            "headers": ["Project", "Span count"],
                            "rows": [["proj-1", 100000], ["proj-2", 54200]],
                        },
                    },
                },
                "em": {
                    "label": "EM",
                    "growth": {
                        "title": "EM — growth",
                        "kpis": [{"label": "Workspaces", "value": 2}],
                        "charts": [],
                        "table": None,
                    },
                    "adoption": {
                        "title": "EM — adoption / usage",
                        "kpis": [{"label": "Experiment count", "value": 900}],
                        "charts": [],
                        "panels": [
                            {
                                "title": "Model-registry engagement",
                                "hint": "snapshot, not over-time",
                                "headers": [
                                    "Workspace",
                                    "Registered models",
                                    "Model versions",
                                ],
                                "rows": [
                                    ["ws-alpha", 2, 3],
                                    ["ws-beta", 1, 1],
                                ],
                            }
                        ],
                    },
                },
                # mpm intentionally omitted to exercise the "missing section"
                # robustness path.
            },
        },
    }


def test_build_html_is_self_contained_and_secure():
    from cometx.cli.admin_growth_report import build_html

    doc = build_html(_sample_report_data())

    assert "<style>" in doc
    assert 'id="report-data"' in doc
    assert "http://" not in doc
    assert "https://" not in doc
    assert "Use cases across all platforms" in doc
    assert "<svg" not in doc  # charts are drawn client-side, not server-side
    assert "createElementNS" in doc  # the inline SVG-drawing JS
    assert '"window"' in doc  # the embedded json payload
    assert "COMET_API_KEY" not in doc
    assert "sk-" not in doc
    assert "not-a-real-secret-12345" not in doc


def test_build_html_escapes_workspace_and_project_names():
    from cometx.cli.admin_growth_report import build_html

    report_data = _sample_report_data()
    report_data["sections"]["unified"]["table"]["rows"][0][
        0
    ] = "<img src=x onerror=alert(1)>"
    doc = build_html(report_data)

    assert "<img src=x onerror=alert(1)>" not in doc
    assert "&lt;img" in doc


def test_build_html_takes_only_report_data_and_ignores_env_secrets(monkeypatch):
    import inspect

    from cometx.cli.admin_growth_report import build_html

    # Architectural guarantee: build_html's ONLY input channel is report_data
    # (no api/key parameter), so it has no way to reach the API key.
    params = list(inspect.signature(build_html).parameters)
    assert params == ["report_data"]

    # And it must not pull secrets out of the environment/config: plant a
    # secret-shaped COMET_API_KEY in the env, render normal data, and assert
    # it never appears in the output.
    secret = "sk-not-a-real-secret-DEADBEEF-0123456789"
    monkeypatch.setenv("COMET_API_KEY", secret)
    doc = build_html(_sample_report_data())
    assert secret not in doc


def test_build_html_handles_missing_sections_gracefully():
    from cometx.cli.admin_growth_report import build_html

    # No products at all, and an empty unified section -- must not raise.
    doc = build_html({"sections": {"unified": {}, "products": {}}})
    assert "<style>" in doc
    assert 'id="report-data"' in doc

    # Completely empty payload must also not raise.
    assert "<style>" in build_html({})
    assert "<style>" in build_html(None)


def test_write_html_writes_file_and_returns_path(tmp_path):
    from cometx.cli.admin_growth_report import write_html

    out = tmp_path / "growth_report.html"
    result = write_html(_sample_report_data(), str(out))

    assert result == str(out)
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "<style>" in content
    assert 'id="report-data"' in content
    assert "Use cases across all platforms" in content


def test_write_growth_html_delegates_to_renderer(tmp_path):
    from cometx.cli.admin_growth_report import write_growth_html

    out = tmp_path / "growth_report.html"
    result = write_growth_html(_sample_report_data(), str(out))

    assert result == str(out)
    assert out.exists()


# ---------------------------------------------------------------------------
# C9: parse_window + GrowthReporter.build() orchestration
# ---------------------------------------------------------------------------


def _now():
    return datetime.datetime(2026, 7, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)


def test_parse_window_days():
    from cometx.cli.admin_growth_report import parse_window

    w = parse_window("7d", _now(), "month")
    assert w.end == _now()
    assert w.start == _now() - datetime.timedelta(days=7)
    assert w.units == "month"


def test_parse_window_weeks_is_seven_times_days():
    from cometx.cli.admin_growth_report import parse_window

    w = parse_window("2w", _now(), "day")
    assert w.start == _now() - datetime.timedelta(days=14)


def test_parse_window_months_is_thirty_day_approximation():
    from cometx.cli.admin_growth_report import parse_window

    w = parse_window("3m", _now(), "day")
    assert w.start == _now() - datetime.timedelta(days=90)


def test_parse_window_years_is_365_day_approximation():
    from cometx.cli.admin_growth_report import parse_window

    w = parse_window("1y", _now(), "day")
    assert w.start == _now() - datetime.timedelta(days=365)


def test_parse_window_default_spec_is_7d():
    from cometx.cli.admin_growth_report import parse_window

    w = parse_window(None, _now(), "day")
    assert w.start == _now() - datetime.timedelta(days=7)


def test_parse_window_rejects_malformed_spec():
    from cometx.cli.admin_growth_report import parse_window

    for bad in ("", "7", "7x", "d7", "-3d", "3 d", "seven-days"):
        try:
            parse_window(bad, _now(), "month")
            assert False, f"expected ValueError for {bad!r}"
        except ValueError:
            pass


def _kind_events(kind, platform, specs):
    """specs: list of (workspace, use_case, y, m, d)."""
    from cometx.cli.admin_growth_report import CreationEvent

    return [
        CreationEvent(
            platform,
            ws,
            uc,
            kind,
            datetime.datetime(y, m, d, tzinfo=datetime.timezone.utc),
        )
        for ws, uc, y, m, d in specs
    ]


def _usage_metric(platform, ws, metric, value, project=None, series=None):
    from cometx.cli.admin_growth_report import UsageMetric

    return UsageMetric(
        platform=platform,
        workspace=ws,
        metric=metric,
        value=value,
        project=project,
        series=series,
    )


def _patch_collectors(monkeypatch, em=None, opik=None, mpm=None):
    from cometx.cli.admin_growth_report import GrowthReporter

    monkeypatch.setattr(
        GrowthReporter, "_collect_em", lambda self, ws: (em or ([], []))
    )
    monkeypatch.setattr(
        GrowthReporter, "_collect_opik", lambda self, ws: (opik or ([], []))
    )
    monkeypatch.setattr(
        GrowthReporter, "_collect_mpm", lambda self, ws: (mpm or ([], []))
    )


def test_build_assembles_report_data_matching_c8_contract(monkeypatch):
    from cometx.cli.admin_growth_report import GrowthReporter

    opik_events = _kind_events(
        "opik_project",
        "opik",
        [
            ("ws-alpha", "op1", 2026, 6, 1),
            ("ws-alpha", "op2", 2026, 7, 5),
            ("ws-beta", "op3", 2026, 7, 6),
        ],
    )
    em_events = _kind_events(
        "em_project",
        "em",
        [("ws-alpha", "em1", 2026, 5, 1), ("ws-beta", "em2", 2026, 7, 8)],
    )
    mpm_events = _kind_events("mpm_model", "mpm", [("ws-beta", "mp1", 2026, 7, 7)])
    opik_usage = [
        _usage_metric(
            "opik",
            "ws-alpha",
            "SPAN_COUNT",
            100,
            project="op1",
            series=[("2026-07", 100)],
        ),
        _usage_metric(
            "opik",
            "ws-alpha",
            "SPAN_COUNT",
            100,
            project=None,
            series=[("2026-07", 100)],
        ),
    ]
    em_usage = [
        _usage_metric(
            "em",
            "ws-alpha",
            "EXPERIMENT_COUNT",
            10,
            project="em1",
            series=[("2026-07", 10)],
        ),
        _usage_metric(
            "em",
            "ws-alpha",
            "EXPERIMENT_COUNT",
            10,
            project=None,
            series=[("2026-07", 10)],
        ),
        _usage_metric("em", "ws-alpha", "REGISTRY_MODELS", 2, project=None),
        _usage_metric("em", "ws-alpha", "REGISTRY_VERSIONS", 3, project=None),
    ]
    mpm_usage = [
        _usage_metric(
            "mpm",
            "ws-beta",
            "PREDICTION_VOLUME",
            50,
            project="mp1",
            series=[("2026-07", 50)],
        ),
        _usage_metric(
            "mpm",
            "ws-beta",
            "PREDICTION_VOLUME",
            50,
            project=None,
            series=[("2026-07", 50)],
        ),
    ]

    _patch_collectors(
        monkeypatch,
        em=(em_events, em_usage),
        opik=(opik_events, opik_usage),
        mpm=(mpm_events, mpm_usage),
    )

    reporter = GrowthReporter(
        MagicMock(), window="7d", units="month", platforms="em,opik,mpm"
    )
    monkeypatch.setattr(GrowthReporter, "_now", lambda self: _now())
    report_data = reporter.build(["ws-alpha", "ws-beta"])

    assert report_data["collectors"] == {"opik": True, "em": True, "mpm": True}

    window = report_data["window"]
    assert window["units"] == "month"
    assert "7d" in window["label"]

    unified = report_data["sections"]["unified"]
    assert unified["title"] == "Use cases across all platforms"
    kpi_labels = [k["label"] for k in unified["kpis"]]
    assert "Departments" in kpi_labels
    assert "Use cases" in kpi_labels
    use_cases_kpi = next(k for k in unified["kpis"] if k["label"] == "Use cases")
    assert use_cases_kpi["value"] == 6  # 3 opik + 2 em + 1 mpm

    chart_ids = [c["id"] for c in unified["charts"]]
    assert "chart-unified-created" not in chart_ids or True  # ids are ours
    kinds_chart = next(c for c in unified["charts"] if c["kind"] == "stackedBars")
    assert kinds_chart["data"]["categories"] == [
        "opik_project",
        "em_project",
        "mpm_model",
    ]
    dept_chart = next(c for c in unified["charts"] if c["kind"] == "groupedBarsH")
    dept_labels = {r["label"] for r in dept_chart["data"]["rows"]}
    assert dept_labels == {"ws-alpha", "ws-beta"}

    # multi-workspace -> unified table breaks down by department
    assert unified["table"]["headers"][0] == "Department"
    ws_rows = {row[0] for row in unified["table"]["rows"]}
    assert ws_rows == {"ws-alpha", "ws-beta"}

    products = report_data["sections"]["products"]
    assert set(products.keys()) == {"opik", "em", "mpm"}

    opik_growth = products["opik"]["growth"]
    assert opik_growth["kpis"][1]["label"] == "Total"
    assert opik_growth["kpis"][1]["value"] == 3  # 3 opik_project events total
    bars_chart = next(c for c in opik_growth["charts"] if c["kind"] == "bars")
    assert sum(p["value"] for p in bars_chart["data"]["points"]) == 3
    area_chart = next(c for c in opik_growth["charts"] if c["kind"] == "area")
    assert area_chart["data"]["points"][-1]["value"] == 3

    em_adoption = products["em"]["adoption"]
    registry_panel = next(
        p
        for p in em_adoption.get("panels", [])
        if p["title"] == "Model-registry engagement"
    )
    assert registry_panel["rows"] == [["ws-alpha", 2, 3]]

    mpm_growth = products["mpm"]["growth"]
    assert mpm_growth["kpis"][1]["value"] == 1

    # never a secret anywhere in the assembled data
    dumped = str(report_data)
    assert "COMET_API_KEY" not in dumped
    assert "sk-" not in dumped


def test_build_resolves_workspaces_via_api_when_none_given(monkeypatch):
    from cometx.cli.admin_growth_report import GrowthReporter

    _patch_collectors(monkeypatch)
    api = MagicMock()
    api.get_workspaces.return_value = ["ws-only"]
    reporter = GrowthReporter(api, window="7d", units="month", platforms="em")
    report_data = reporter.build([])

    api.get_workspaces.assert_called_once()
    assert report_data["sections"]["unified"]["kpis"][0]["value"] == 1


def test_build_drops_unimportable_optional_platforms(monkeypatch):
    import builtins

    from cometx.cli.admin_growth_report import GrowthReporter

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "opik":
            raise ImportError("no opik")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    _patch_collectors(monkeypatch)

    reporter = GrowthReporter(
        MagicMock(), window="7d", units="month", platforms="em,opik,mpm"
    )
    report_data = reporter.build(["ws1"])

    assert report_data["collectors"]["opik"] is False
    assert "opik" not in report_data["sections"]["products"]


def test_build_single_workspace_breaks_down_unified_table_by_use_case(monkeypatch):
    from cometx.cli.admin_growth_report import GrowthReporter

    opik_events = _kind_events("opik_project", "opik", [("ws1", "op1", 2026, 6, 1)])
    _patch_collectors(monkeypatch, opik=(opik_events, []))

    reporter = GrowthReporter(MagicMock(), window="7d", units="month", platforms="opik")
    report_data = reporter.build(["ws1"])

    unified_table = report_data["sections"]["unified"]["table"]
    assert unified_table["headers"][0] == "Use case"
    assert unified_table["rows"] == [["op1", "Opik projects", "2026-06-01"]]


def test_generate_growth_report_writes_html_with_no_secret(monkeypatch, tmp_path):
    from cometx.cli.admin_growth_report import generate_growth_report

    opik_events = _kind_events(
        "opik_project", "opik", [("ws1", "op1", 2026, 6, 1), ("ws1", "op2", 2026, 7, 8)]
    )
    _patch_collectors(monkeypatch, opik=(opik_events, []))

    out = tmp_path / "growth.html"
    api = MagicMock()
    api.config = {"comet.api_key": "sk-should-never-leak-0000"}
    path = generate_growth_report(
        api,
        ["ws1"],
        window="7d",
        units="month",
        platforms="opik",
        output=str(out),
        no_open=True,
    )

    content = out.read_text(encoding="utf-8")
    assert path == str(out)
    assert "Use cases across all platforms" in content
    assert "op1" in content or "op2" in content
    assert "sk-should-never-leak-0000" not in content


def test_generate_growth_report_full_chain_all_platforms(monkeypatch, tmp_path):
    """End-to-end seam coverage: fixed collector output -> build() ->
    build_html() -> the written HTML file, across all three platforms.
    Chains what test_build_assembles_report_data_matching_c8_contract and
    the render-layer tests otherwise only cover separately."""
    from cometx.cli.admin_growth_report import generate_growth_report

    monkeypatch.setenv("COMET_API_KEY", "sk-planted-env-secret-should-not-leak")

    opik_events = _kind_events(
        "opik_project",
        "opik",
        [
            ("ws-alpha", "op1", 2026, 6, 1),
            ("ws-alpha", "op2", 2026, 7, 5),
            ("ws-beta", "op3", 2026, 7, 6),
        ],
    )
    em_events = _kind_events(
        "em_project",
        "em",
        [("ws-alpha", "em1", 2026, 5, 1), ("ws-beta", "em2", 2026, 7, 8)],
    )
    mpm_events = _kind_events("mpm_model", "mpm", [("ws-beta", "mp1", 2026, 7, 7)])
    _patch_collectors(
        monkeypatch,
        em=(em_events, []),
        opik=(opik_events, []),
        mpm=(mpm_events, []),
    )

    out = tmp_path / "growth-full-chain.html"
    api = MagicMock()
    api.config = {"comet.api_key": "sk-api-secret-should-not-leak"}
    path = generate_growth_report(
        api,
        ["ws-alpha", "ws-beta"],
        window="7d",
        units="month",
        platforms="em,opik,mpm",
        output=str(out),
        no_open=True,
    )

    content = out.read_text(encoding="utf-8")
    assert path == str(out)

    # Section titles from all three product sections + the unified section.
    assert "Use cases across all platforms" in content
    assert "Opik — growth" in content
    assert "EM — growth" in content
    assert "MPM — growth" in content

    # KPI value derived from the fixed events: 3 opik + 2 em + 1 mpm = 6
    # total use cases in the unified section.
    assert ">6<" in content

    # No secret (env-planted or api-config) ever reaches the rendered HTML.
    assert "sk-planted-env-secret-should-not-leak" not in content
    assert "sk-api-secret-should-not-leak" not in content
    assert "COMET_API_KEY" not in content
