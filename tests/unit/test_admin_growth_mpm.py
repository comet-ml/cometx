from unittest.mock import MagicMock

import pytest


def _api(member_payload=None, member_error=None, registry=None, details=None):
    """A fake comet_ml API.

    `member_payload`: the mpm/v3/workspaces JSON (or `member_error` to raise).
    `registry`: {workspace: [model names]} or an Exception per workspace.
    `details`: {(workspace, model): details dict} or an Exception.
    """
    api = MagicMock()
    api.config = {"comet.url_override": "https://comet.example.com"}
    api.api_key = "KEY"

    def get(url, headers=None, params=None):
        assert url == "https://comet.example.com/api/mpm/v3/workspaces"
        assert headers == {"Authorization": "KEY"}
        if member_error is not None:
            raise member_error
        response = MagicMock()
        response.json.return_value = member_payload or {"workspaces": []}
        return response

    def get_registry_models(ws):
        value = (registry or {}).get(ws, [])
        if isinstance(value, Exception):
            raise value
        return [{"modelName": n, "registryModelId": "id-" + n} for n in value]

    def get_registry_model_details(ws, name):
        value = (details or {})[(ws, name)]
        if isinstance(value, Exception):
            raise value
        return value

    api._client.get.side_effect = get
    api._client.get_registry_models.side_effect = get_registry_models
    api._client.get_registry_model_details.side_effect = get_registry_model_details
    return api


def test_member_workspaces_answered_from_one_call():
    from cometx.cli.admin_growth_mpm import fetch_mpm_presence

    api = _api(
        member_payload={
            "workspaces": [
                {
                    "workspaceName": "fraud",
                    "models": [{"modelId": "m1", "modelName": "scorer"}],
                },
                {"workspaceName": "research", "models": []},
                # not in the requested set: ignored
                {"workspaceName": "other", "models": [{"modelId": "x"}]},
            ]
        }
    )
    presence, lookup = fetch_mpm_presence(api, ["fraud", "research"])
    assert presence == {"fraud": [{"id": "m1", "name": "scorer"}], "research": []}
    assert lookup == "ok"
    # every requested workspace was covered by v3: no registry calls
    api._client.get_registry_models.assert_not_called()


def test_non_member_workspaces_fall_back_to_registry():
    from cometx.cli.admin_growth_mpm import fetch_mpm_presence

    api = _api(
        member_payload={"workspaces": [{"workspaceName": "fraud", "models": []}]},
        registry={"credit": ["pd", "lgd", "ead"]},
        details={
            ("credit", "pd"): {"isMonitored": True},
            # Jackson may drop the `is` prefix: accept `monitored` too
            ("credit", "lgd"): {"monitored": True},
            ("credit", "ead"): {"isMonitored": False},
        },
    )
    presence, _lookup = fetch_mpm_presence(api, ["fraud", "credit"])
    assert presence["fraud"] == []
    # sorted by name for stable output
    assert presence["credit"] == [
        {"id": "id-lgd", "name": "lgd"},
        {"id": "id-pd", "name": "pd"},
    ]


def test_v3_failure_uses_registry_for_everything():
    from cometx.cli.admin_growth_mpm import fetch_mpm_presence

    api = _api(
        member_error=RuntimeError("404"),
        registry={"a": ["m"], "b": []},
        details={("a", "m"): {"isMonitored": True}},
    )
    presence, lookup = fetch_mpm_presence(api, ["a", "b"])
    assert presence == {"a": [{"id": "id-m", "name": "m"}], "b": []}
    assert lookup == "error"  # a bare exception with no HTTP status


@pytest.mark.parametrize(
    "registry, details",
    [
        # listing fails
        ({"w": RuntimeError("403")}, {}),
        # one detail lookup fails
        (
            {"w": ["m1", "m2"]},
            {("w", "m1"): {"isMonitored": True}, ("w", "m2"): RuntimeError("500")},
        ),
        # flag missing from the response
        ({"w": ["m1"]}, {("w", "m1"): {"modelName": "m1"}}),
    ],
)
def test_unknown_is_none_not_empty(registry, details):
    from cometx.cli.admin_growth_mpm import fetch_mpm_presence

    api = _api(registry=registry, details=details)
    assert fetch_mpm_presence(api, ["w"])[0] == {"w": None}


def test_apply_mpm_presence_leaves_unknown_workspaces_untouched():
    from cometx.cli.admin_growth_mpm import apply_mpm_presence
    from cometx.cli.admin_growth_users import parse_workspaces

    cb = {
        "workspaces": [{"name": "a"}, {"name": "b"}, {"name": "c"}],
        "users": {"licensedUsers": []},
    }
    out = apply_mpm_presence(
        cb, {"a": [{"id": "m1", "name": "scorer"}], "b": [], "c": None}
    )
    assert cb["workspaces"][0] == {"name": "a"}  # input not mutated
    ws = {w.name: w for w in parse_workspaces(out)}
    assert (ws["a"].mpm_enabled, ws["a"].num_monitored_models) == (True, 1)
    assert ws["a"].monitored_models == ("scorer",)
    assert (ws["b"].mpm_enabled, ws["b"].num_monitored_models) == (False, 0)
    assert (ws["c"].mpm_enabled, ws["c"].num_monitored_models) == (None, None)


def test_apply_mpm_presence_drops_stale_fields_on_failed_lookup():
    from cometx.cli.admin_growth_mpm import apply_mpm_presence
    from cometx.cli.admin_growth_users import parse_workspaces

    cb = {
        "workspaces": [
            {"name": "w", "mpmEnabled": True, "monitoredModels": [{"id": "old"}]}
        ]
    }
    (w,) = parse_workspaces(apply_mpm_presence(cb, {"w": None}))
    assert (w.mpm_enabled, w.num_monitored_models) == (None, None)


def _chargeback():
    return {
        "workspaces": [
            {"name": "fraud", "members": [{"userName": "alice"}]},
            {"name": "credit", "members": [{"userName": "bob"}]},
        ],
        "users": {
            "licensedUsers": [
                {"username": "alice", "email": "a", "lastUsedAt": 1},
                {"username": "bob", "email": "b", "lastUsedAt": 1},
            ]
        },
    }


def test_build_without_mpm_makes_no_mpm_calls():
    from cometx.cli.admin_growth_report import GrowthReporter

    api = _api()
    r = GrowthReporter(api, window="7d", units="month")
    r.build([], chargeback=_chargeback())
    api._client.get_registry_models.assert_not_called()
    _users, ws_records, kpis = r.last_parsed()
    assert all(w.mpm_enabled is None for w in ws_records)
    assert "mpm_workspaces_unchecked" not in {k[0] for k in kpis}


def test_build_with_mpm_merges_presence_and_reports_unchecked():
    from cometx.cli.admin_growth_report import GrowthReporter

    api = _api(
        member_payload={
            "workspaces": [
                {
                    "workspaceName": "fraud",
                    "models": [{"modelId": "m1", "modelName": "scorer"}],
                }
            ]
        },
        registry={"credit": RuntimeError("403")},
    )
    r = GrowthReporter(api, window="7d", units="month", mpm=True)
    report = r.build([], chargeback=_chargeback())
    _users, ws_records, kpis = r.last_parsed()
    ws = {w.name: w for w in ws_records}
    assert ws["fraud"].num_monitored_models == 1
    assert ws["credit"].mpm_enabled is None  # unknown, not "no MPM"
    by_name = {k[0]: k[1] for k in kpis}
    assert by_name["mpm_workspaces"] == 1
    assert by_name["mpm_workspaces_unchecked"] == 1
    overview = report["sections"]["unified"]
    kpis_by_label = {k["label"]: k for k in overview["kpis"]}
    models_kpi = kpis_by_label["Monitored models"]
    # partial: a lower bound, labeled as such
    assert models_kpi["value"] == "≥ 1"
    assert models_kpi["sub"] == "MPM checked in 1/2 workspaces"
    ws_kpi = kpis_by_label["MPM workspaces"]
    # percentage over the checked workspace, not all workspaces
    assert ws_kpi["sub"] == "100% of 1 checked; 1 unknown"


def test_build_with_mpm_all_lookups_failed_still_reports_unchecked():
    from cometx.cli.admin_growth_report import GrowthReporter

    api = _api(
        member_error=RuntimeError("404"),
        registry={"fraud": RuntimeError("403"), "credit": RuntimeError("403")},
    )
    r = GrowthReporter(api, window="7d", units="month", mpm=True)
    r.build([], chargeback=_chargeback())
    _users, _ws, kpis = r.last_parsed()
    by_name = {k[0]: k[1] for k in kpis}
    assert "mpm_workspaces" not in by_name
    assert by_name["mpm_workspaces_unchecked"] == 2


def test_build_with_mpm_only_checks_scoped_workspaces():
    from cometx.cli.admin_growth_report import GrowthReporter

    api = _api(member_error=RuntimeError("404"), registry={"credit": []})
    r = GrowthReporter(api, window="7d", units="month", mpm=True)
    r.build(["credit"], chargeback=_chargeback())
    called = [c.args[0] for c in api._client.get_registry_models.call_args_list]
    assert called == ["credit"]


@pytest.mark.parametrize(
    "override, expected",
    [
        (
            "https://comet.example.com/clientlib/",
            "https://comet.example.com/api/mpm/v3/workspaces",
        ),
        (
            "https://comet.example.com/comet/clientlib",
            "https://comet.example.com/comet/api/mpm/v3/workspaces",
        ),
        (
            "https://comet.example.com",
            "https://comet.example.com/api/mpm/v3/workspaces",
        ),
    ],
)
def test_mpm_url_drops_sdk_clientlib_segment(override, expected):
    from cometx.cli.admin_growth_mpm import _mpm_url

    api = MagicMock()
    api.config = {"comet.url_override": override}
    assert _mpm_url(api) == expected


def _status_error(status):
    exc = RuntimeError("GET failed with status code %d" % status)
    exc.response = MagicMock(status_code=status)
    return exc


@pytest.mark.parametrize(
    "error, expected, warning",
    [
        (_status_error(401), "refused", "refused this key"),
        (_status_error(403), "refused", "refused this key"),
        (_status_error(404), "not_found", "may not be installed"),
        (_status_error(502), "error", "failed"),
        (RuntimeError("connection reset"), "error", "failed"),
    ],
)
def test_member_lookup_failures_are_classified_and_warned(
    capsys, error, expected, warning
):
    from cometx.cli.admin_growth_mpm import fetch_mpm_presence

    api = _api(member_error=error, registry={"w": []})
    presence, lookup = fetch_mpm_presence(api, ["w"])
    assert lookup == expected
    # the fallback still runs
    assert presence == {"w": []}
    out = capsys.readouterr().out
    assert "Warning: the MPM API" in out and warning in out
    if expected == "refused":
        # the refused case says why the fallback may undercount
        assert "organization admin" in out


def test_member_lookup_shape_error_is_error_not_ok():
    from cometx.cli.admin_growth_mpm import fetch_mpm_presence

    api = _api(member_payload={"unexpected": []}, registry={"w": []})
    _presence, lookup = fetch_mpm_presence(api, ["w"])
    assert lookup == "error"


@pytest.mark.parametrize(
    "error, lookup, note",
    [
        (None, "ok", None),
        (_status_error(401), "refused", "MPM API refused this key; counts may be low"),
        (_status_error(404), "not_found", "MPM API not found on this server"),
    ],
)
def test_build_records_member_lookup_in_csv_and_html(error, lookup, note):
    from cometx.cli.admin_growth_report import GrowthReporter

    api = _api(
        member_error=error,
        member_payload={"workspaces": [{"workspaceName": "fraud", "models": []}]},
        registry={"fraud": [], "credit": []},
    )
    r = GrowthReporter(api, window="7d", units="month", mpm=True)
    report = r.build([], chargeback=_chargeback())
    _users, _ws, kpis = r.last_parsed()
    text_by_name = {k[0]: k[3] for k in kpis}
    assert text_by_name["mpm_member_lookup"] == lookup
    models_kpi = next(
        k
        for k in report["sections"]["unified"]["kpis"]
        if k["label"] == "Monitored models"
    )
    if note:
        # even with every workspace checked, a failed lookup is flagged
        assert note in models_kpi["sub"]
    else:
        assert models_kpi["sub"] == "MPM (is_monitored)"


def test_build_without_mpm_has_no_member_lookup_kpi():
    from cometx.cli.admin_growth_report import GrowthReporter

    r = GrowthReporter(_api(), window="7d", units="month")
    r.build([], chargeback=_chargeback())
    assert "mpm_member_lookup" not in {k[0] for k in r.last_parsed()[2]}
