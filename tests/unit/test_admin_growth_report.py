import datetime
import importlib
from unittest.mock import MagicMock


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


def _win():
    from cometx.cli.admin_growth_report import Window

    return Window(
        datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
        datetime.datetime(2026, 12, 31, tzinfo=datetime.timezone.utc),
        "month",
    )


def _sample_report_data():
    """A representative chargeback-only `report_data` payload: a top-level
    `window` plus the four chargeback sections (Organization overview, Users,
    Leaderboards, Personal vs service accounts). No `products`/`collectors` —
    those belonged to the retired SDK/platform-direct methodology."""
    return {
        "meta": {
            "title": "Growth report — acme <script> & Co",
            "org": "acme-research",
            "generated": "2026-07-09",
            "source": "Comet Admin API (chargeback)",
            "scope": "Org-wide: 4 workspaces, 12 users (chargeback)",
        },
        "window": {
            "start": "2026-07-02",
            "end": "2026-07-09",
            "units": "day",
            "label": "Analysis window: Jul 2 – Jul 9, 2026 (7d)",
            "count_before": 0,
        },
        "sections": {
            "unified": {
                "title": "Organization overview (chargeback)",
                "window_chip": "Analysis window: Jul 2 – Jul 9, 2026 (7d)",
                "kpis": [
                    {"label": "Total workspaces", "value": 4},
                    {
                        "label": "Total projects",
                        "value": 77,
                        "sub": "EM projects",
                        "tone": "ok",
                    },
                    {
                        "label": "New in 7d (% of base)",
                        "value": "18.5%",
                        "sub": "+12 new",
                    },
                    {
                        "label": "Active workspaces %",
                        "value": "75.0%",
                        "sub": "3/4 active",
                    },
                ],
                "charts": [
                    {
                        "id": "chart-unified-platform-mix",
                        "kind": "groupedBarsH",
                        "title": "Workspace platform mix",
                        "hint": "org-wide (chargeback); Opik is a per-user proxy",
                        "data": {
                            "rows": [
                                {"label": "EM only", "value": 2},
                                {"label": "Opik only", "value": 1},
                                {"label": "EM + Opik", "value": 1},
                                {"label": "Neither", "value": 0},
                            ]
                        },
                    },
                    {
                        "id": "chart-unified-workspace-churn",
                        "kind": "barsLine",
                        "title": "Workspaces added vs. deleted",
                        "hint": "org-wide (chargeback) · monthly · deletion is a proxy",
                        "legend": [
                            {"label": "Added", "color": "--ok"},
                            {"label": "Deleted", "color": "--warn"},
                            {"label": "Growth rate", "color": "--accent"},
                        ],
                        "data": {
                            "points": [
                                {
                                    "key": "2026-06",
                                    "values": {"added": 3, "deleted": 0, "rate": 0.0},
                                },
                                {
                                    "key": "2026-07",
                                    "values": {"added": 1, "deleted": 0, "rate": 33.3},
                                },
                            ],
                            "bars": ["added", "deleted"],
                            "line": "rate",
                            "bar_labels": {"added": "Added", "deleted": "Deleted"},
                            "bar_colors": ["--ok", "--warn"],
                            "line_label": "Growth rate",
                            "line_color": "--accent",
                            "window_start": None,
                            "window_end": None,
                        },
                    },
                ],
                "table": {
                    "title": "By workspace (org-wide, chargeback)",
                    "headers": [
                        "Workspace",
                        "Members",
                        "Projects",
                        "Experiments",
                        "Data (MB)",
                    ],
                    "rows": [
                        ["ws-alpha", 3, 15, 40, 120],
                        ["ws-beta", 2, 20, 37, 95],
                    ],
                },
            },
            "people": {
                "title": "Users",
                "window_chip": "Analysis window: Jul 2 – Jul 9, 2026 (7d)",
                "kpis": [
                    {"label": "Total users", "value": 12},
                    {"label": "Active users (60d)", "value": 9},
                    {"label": "Active users %", "value": "75.0%"},
                    {
                        "label": "New in 7d (% of base)",
                        "value": "9.1%",
                        "sub": "+1 new",
                    },
                ],
                "charts": [
                    {
                        "id": "chart-people-active-total",
                        "kind": "lines",
                        "title": "Active vs. total users",
                        "hint": "active window 60d · monthly",
                        "legend": [
                            {"label": "Total", "color": "--sdk"},
                            {"label": "Active", "color": "--ok"},
                        ],
                        "data": {
                            "categories": ["total", "active"],
                            "labels": {"total": "Total", "active": "Active"},
                            "colors": ["--sdk", "--ok"],
                            "points": [
                                {
                                    "key": "2026-06",
                                    "values": {"total": 10, "active": 7},
                                },
                                {
                                    "key": "2026-07",
                                    "values": {"total": 12, "active": 9},
                                },
                            ],
                            "window_start": None,
                            "window_end": None,
                        },
                    }
                ],
            },
            "leaderboards": {
                "title": "Leaderboards",
                "charts": [
                    {
                        "id": "chart-lb-ws-experiments-top",
                        "kind": "groupedBarsH",
                        "title": "Top 5 workspaces by experiments",
                        "hint": "org-wide, exact (chargeback per-workspace)",
                        "data": {
                            "rows": [
                                {"label": "ws-alpha", "value": 40},
                                {"label": "ws-beta", "value": 37},
                            ]
                        },
                    },
                    {
                        "id": "chart-lb-user-opik_span_count-top",
                        "kind": "groupedBarsH",
                        "title": "Top 5 users by Opik spans",
                        "hint": "org-wide (chargeback)",
                        "data": {
                            "rows": [
                                {"label": "alice", "value": 5000},
                                {"label": "bob", "value": 1200},
                            ]
                        },
                    },
                ],
            },
            "personal_vs_service": {
                "title": "Personal vs. service accounts",
                "charts": [
                    {
                        "id": "chart-personal-vs-service-experiments",
                        "kind": "groupedBarsH",
                        "title": "Personal vs. service accounts: experiments",
                        "hint": "Source: heuristic (regex); admin "
                        "service-accounts API unavailable.",
                        "data": {
                            "rows": [
                                {"label": "Personal", "value": 70},
                                {"label": "Service", "value": 7},
                            ]
                        },
                    }
                ],
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
    assert "Organization overview (chargeback)" in doc
    assert "<svg" not in doc  # charts are drawn client-side, not server-side
    assert "createElementNS" in doc  # the inline SVG-drawing JS
    assert '"window"' in doc  # the embedded json payload
    assert "COMET_API_KEY" not in doc
    assert "sk-" not in doc
    assert "not-a-real-secret-12345" not in doc


def test_build_html_tables_are_collapsible_and_collapsed_by_default():
    from cometx.cli.admin_growth_render import render_table

    table = {
        "title": "By workspace",
        "headers": ["Workspace", "Total"],
        "rows": [["opik-demos", 16], ["scout-test-leo", 5]],
    }
    html = render_table(table)

    # native <details>/<summary> disclosure, no JS
    assert "<details" in html and "<summary" in html
    # collapsed by default -> no `open` attribute on the details element
    assert '<details class="tablecard" open' not in html
    assert '<details class="tablecard">' in html
    # summary shows the title + the row count (2 rows)
    assert "By workspace" in html
    assert '<span class="count">2</span>' in html
    # the rows are still present (inside the collapsed body)
    assert "opik-demos" in html and "scout-test-leo" in html
    # and the whole thing renders into the full document too
    from cometx.cli.admin_growth_report import build_html

    assert "<details" in build_html(_sample_report_data())


def test_charts_have_interactive_hover_tooltip_infra():
    # Hover must be a real interactive tooltip (div + guide via attachTip),
    # not flaky native SVG <title> elements.
    from cometx.cli.admin_growth_report import build_html

    doc = build_html(_sample_report_data())
    assert "attachTip" in doc  # the shared hover helper
    assert "charttip" in doc  # the positioned tooltip element/class
    assert 'class: "guide"' in doc  # the vertical hover guide line


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

    # An empty unified section (and a stray legacy key) -- must not raise.
    doc = build_html({"sections": {"unified": {}}})
    assert "<style>" in doc
    assert 'id="report-data"' in doc

    # Completely empty payload must also not raise.
    assert "<style>" in build_html({})
    assert "<style>" in build_html(None)


def test_build_html_has_no_products_or_collectors():
    from cometx.cli.admin_growth_render import build_html

    report = {
        "meta": {
            "title": "T",
            "generated": "x",
            "source": "s",
            "scope": "Org-wide (chargeback)",
        },
        "window": {"label": "w"},
        "sections": {
            "unified": {
                "title": "Organization overview (chargeback)",
                "kpis": [],
                "charts": [],
                "table": None,
            }
        },
    }
    doc = build_html(report)
    assert "product-heading" not in doc
    assert 'aria-label="collector status"' not in doc
    assert "drawStacked" not in doc
    assert "drawArea" not in doc
    assert "Organization overview (chargeback)" in doc


def test_write_html_writes_file_and_returns_path(tmp_path):
    from cometx.cli.admin_growth_report import write_html

    out = tmp_path / "growth_report.html"
    result = write_html(_sample_report_data(), str(out))

    assert result == str(out)
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "<style>" in content
    assert 'id="report-data"' in content
    assert "Organization overview (chargeback)" in content


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


def test_people_section_built_from_chargeback():
    from cometx.cli.admin_growth_report import GrowthReporter

    api = MagicMock()
    api.config = {"comet.url_override": "https://c.example.com"}
    api.api_key = "K"
    cb = {
        "workspaces": [],
        "users": {
            "licensedUsers": [
                {
                    "username": "alice",
                    "email": "a@x",
                    "lastUsedAt": 4_000,
                    "createdAt": 1,
                    "experimentCount": 5,
                    "dataLoggedMb": 1.0,
                    "opikSpanCount": 9,
                    "suspended": False,
                },
            ]
        },
    }
    r = GrowthReporter(
        api,
        window="7d",
        units="month",
        active_window="60d",
        leaderboard_top_n=5,
    )
    from cometx.cli.admin_growth_users import parse_users

    section = r._build_people_section(parse_users(cb), now_ms=4_500)
    assert section["title"] == "Users"
    labels = [k["label"] for k in section["kpis"]]
    assert "Active users %" in labels
    # workspace metrics live in the Organization overview section
    assert "Active workspaces %" not in labels
    assert any(x.startswith("Active users (") for x in labels)


def test_people_section_user_growth_kpi_with_window():
    from cometx.cli.admin_growth_report import GrowthReporter
    from cometx.cli.admin_growth_users import parse_users

    before = int(
        datetime.datetime(2025, 6, 1, tzinfo=datetime.timezone.utc).timestamp() * 1000
    )
    in_win = int(
        datetime.datetime(2026, 3, 1, tzinfo=datetime.timezone.utc).timestamp() * 1000
    )
    now_ms = int(
        datetime.datetime(2026, 7, 1, tzinfo=datetime.timezone.utc).timestamp() * 1000
    )
    cb = {
        "workspaces": [],
        "users": {
            "licensedUsers": [
                {
                    "username": "old1",
                    "email": "o1",
                    "createdAt": before,
                    "lastUsedAt": now_ms,
                },
                {
                    "username": "old2",
                    "email": "o2",
                    "createdAt": before,
                    "lastUsedAt": now_ms,
                },
                {
                    "username": "new1",
                    "email": "n1",
                    "createdAt": in_win,
                    "lastUsedAt": now_ms,
                },
            ]
        },
    }
    api = MagicMock()
    r = GrowthReporter(api, window="7d", units="month", active_window="60d")
    section = r._build_people_section(parse_users(cb), now_ms=now_ms, window=_win())
    growth = next(k for k in section["kpis"] if k["label"].startswith("New in"))
    # one new account in-window vs two before => 50%, "+1 new"
    assert growth["value"] == "50.0%"
    assert growth["sub"] == "+1 new"


def test_malformed_chargeback_degrades_people_section_without_crashing(monkeypatch):
    """A chargeback fetch that succeeds but returns a shape parse_users()
    cannot handle must degrade to no people section, not crash build()."""
    import cometx.cli.admin_growth_report as agr
    from cometx.cli.admin_growth_report import GrowthReporter

    api = MagicMock()
    api.config = {"comet.url_override": "https://c.example.com"}
    api.api_key = "K"
    api.get_workspaces.return_value = []

    # Malformed-but-successfully-fetched payload: a licensed-user entry that
    # is a plain string, not a dict, so parse_users()'s `raw.get(...)` call
    # raises AttributeError instead of returning parsed records.
    malformed = {"workspaces": [], "users": {"licensedUsers": ["not-a-dict"]}}
    monkeypatch.setattr(agr, "fetch_chargeback_report", lambda *a, **k: malformed)

    r = GrowthReporter(
        api,
        window="7d",
        units="month",
        active_window="60d",
        leaderboard_top_n=5,
    )

    data = r.build([])  # must not raise
    assert "people" not in data["sections"]


def test_malformed_chargeback_degrades_leaderboards_and_personal_vs_service(
    monkeypatch, capsys
):
    """Same malformed chargeback as above, but exercising the leaderboards /
    personal_vs_service wiring: `leaderboard_users` is assigned from
    `parse_users(chargeback)` inside the leaderboards `try` block, then
    referenced later (`if leaderboard_users:`) in the personal-vs-service
    block. Before the fix, a `parse_users` failure left `leaderboard_users`
    unbound, so the later reference raised `UnboundLocalError` -- masked by
    the personal-vs-service `except`, but printing a misleading "referenced
    before assignment" warning that misattributes the failure. `build()`
    must not raise either way, but after the fix neither the leaderboards
    nor personal_vs_service sections are present and no misleading
    UnboundLocalError-style warning is printed."""
    import cometx.cli.admin_growth_report as agr
    from cometx.cli.admin_growth_report import GrowthReporter

    api = MagicMock()
    api.config = {"comet.url_override": "https://c.example.com"}
    api.api_key = "K"
    api.get_workspaces.return_value = []

    malformed = {"workspaces": [], "users": {"licensedUsers": ["not-a-dict"]}}
    monkeypatch.setattr(agr, "fetch_chargeback_report", lambda *a, **k: malformed)

    r = GrowthReporter(
        api,
        window="7d",
        units="month",
        active_window="60d",
        leaderboard_top_n=5,
    )

    data = r.build([])  # must not raise

    assert "people" not in data["sections"]
    assert "leaderboards" not in data["sections"]
    assert "personal_vs_service" not in data["sections"]

    captured = capsys.readouterr()
    assert "referenced before assignment" not in captured.out
    assert "not associated with a value" not in captured.out
    assert "leaderboard_users" not in captured.out


def test_fetch_service_accounts_parses_name_list():
    from cometx.cli.admin_growth_report import _fetch_service_accounts

    api = MagicMock()
    api.config = {"comet.url_override": "https://c.example.com"}
    api.api_key = "K"
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"serviceAccounts": [{"name": "svc-a"}, {"name": "svc-b"}]}
    api._client.get.return_value = resp

    names = _fetch_service_accounts(api)

    assert names == {"svc-a", "svc-b"}
    called_url = api._client.get.call_args[0][0]
    assert called_url == "https://c.example.com/api/admin/service-accounts"


def test_fetch_service_accounts_returns_none_on_failure():
    from cometx.cli.admin_growth_report import _fetch_service_accounts

    api = MagicMock()
    api.config = {"comet.url_override": "https://c.example.com"}
    api.api_key = "K"
    api._client.get.side_effect = RuntimeError("boom")

    assert _fetch_service_accounts(api) is None


def _fetch_with_payload(payload):
    from cometx.cli.admin_growth_report import _fetch_service_accounts

    api = MagicMock()
    api.config = {"comet.url_override": "https://c.example.com"}
    api.api_key = "K"
    resp = MagicMock(status_code=200)
    resp.json.return_value = payload
    api._client.get.return_value = resp
    return _fetch_service_accounts(api)


def test_fetch_service_accounts_returns_none_on_unrecognized_shape():
    # A successful response whose shape isn't a list and doesn't carry any
    # known container key -- unparseable, so callers must fall back to the
    # regex heuristic (None), not silently report zero service accounts.
    assert _fetch_with_payload({"weird": 123}) is None


def test_fetch_service_accounts_returns_none_when_entries_all_unparseable():
    # A recognized bare-list container, but every entry is a type we can't
    # extract a name from -- still unparseable overall, so None (not an
    # empty set, which would misleadingly imply "admin API says zero").
    assert _fetch_with_payload([1, 2, 3]) is None


def test_fetch_service_accounts_returns_empty_set_for_genuinely_empty_container():
    # A recognized, genuinely-empty container IS a real "zero service
    # accounts" answer from the admin API -- keep it as an empty set (not
    # None) so classify_accounts still reports source="admin_api".
    assert _fetch_with_payload({"serviceAccounts": []}) == set()
    assert _fetch_with_payload([]) == set()


def test_build_personal_vs_service_section_admin_api_source():
    from cometx.cli.admin_growth_report import GrowthReporter
    from cometx.cli.admin_growth_users import parse_users

    api = MagicMock()
    r = GrowthReporter(api, window="7d", units="month")
    cb = {
        "workspaces": [],
        "users": {
            "licensedUsers": [
                {
                    "username": "alice",
                    "email": "a@x",
                    "experimentCount": 40,
                    "dataLoggedMb": 1.0,
                    "opikSpanCount": 5,
                    "lastUsedAt": 1,
                    "suspended": False,
                },
                {
                    "username": "bob",
                    "email": "b@x",
                    "experimentCount": 1,
                    "dataLoggedMb": 0.0,
                    "opikSpanCount": 0,
                    "lastUsedAt": 1,
                    "suspended": False,
                },
            ]
        },
    }
    users = parse_users(cb)

    section = r._build_personal_vs_service_section(users, service_account_names={"bob"})

    assert section["title"] == "Personal vs. service accounts"
    assert "hint" not in section
    exp_chart = next(c for c in section["charts"] if "experiments" in c["title"])
    assert "admin API" in exp_chart["hint"]
    rows = {row["label"]: row["value"] for row in exp_chart["data"]["rows"]}
    assert rows == {"Personal": 40, "Service": 1}


def test_build_personal_vs_service_section_heuristic_fallback_omits_empty_metric():
    from cometx.cli.admin_growth_report import GrowthReporter
    from cometx.cli.admin_growth_users import parse_users

    api = MagicMock()
    r = GrowthReporter(api, window="7d", units="month")
    cb = {
        "workspaces": [],
        "users": {
            "licensedUsers": [
                {
                    "username": "svc-pipeline",
                    "email": "p@x",
                    "experimentCount": 3,
                    "dataLoggedMb": 0.0,
                    "lastUsedAt": 1,
                },
                {
                    "username": "dana",
                    "email": "d@x",
                    "experimentCount": 7,
                    "dataLoggedMb": 0.0,
                    "lastUsedAt": 1,
                },
            ]
        },
    }
    users = parse_users(cb)

    section = r._build_personal_vs_service_section(users, service_account_names=None)

    assert "hint" not in section
    # No user has opik_span_count set -> spans metric all-zero -> omitted
    assert all("spans" not in c["title"] for c in section["charts"])
    exp_chart = next(c for c in section["charts"] if "experiments" in c["title"])
    assert "heuristic" in exp_chart["hint"]
    rows = {row["label"]: row["value"] for row in exp_chart["data"]["rows"]}
    assert rows == {"Personal": 7, "Service": 3}


def test_build_personal_vs_service_section_none_when_no_users():
    from cometx.cli.admin_growth_report import GrowthReporter

    api = MagicMock()
    r = GrowthReporter(api, window="7d", units="month")
    assert r._build_personal_vs_service_section([], service_account_names=None) is None


def test_assemble_report_data_wires_personal_vs_service_section(monkeypatch):
    """build() end-to-end: chargeback present -> personal_vs_service section
    appears in report_data["sections"], sourced from the admin API when
    _fetch_service_accounts succeeds."""
    import cometx.cli.admin_growth_report as agr
    from cometx.cli.admin_growth_report import GrowthReporter

    api = MagicMock()
    api.config = {"comet.url_override": "https://c.example.com"}
    api.api_key = "K"
    api.get_workspaces.return_value = []
    cb = {
        "workspaces": [],
        "users": {
            "licensedUsers": [
                {
                    "username": "alice",
                    "email": "a@x",
                    "experimentCount": 40,
                    "dataLoggedMb": 1.0,
                    "opikSpanCount": 5,
                    "lastUsedAt": 1,
                    "suspended": False,
                },
                {
                    "username": "bob",
                    "email": "b@x",
                    "experimentCount": 1,
                    "dataLoggedMb": 0.0,
                    "opikSpanCount": 0,
                    "lastUsedAt": 1,
                    "suspended": False,
                },
            ]
        },
    }
    monkeypatch.setattr(agr, "fetch_chargeback_report", lambda *a, **k: cb)
    monkeypatch.setattr(agr, "_fetch_service_accounts", lambda api: {"bob"})

    r = GrowthReporter(
        api,
        window="7d",
        units="month",
        active_window="60d",
        leaderboard_top_n=5,
    )
    data = r.build([])

    section = data["sections"]["personal_vs_service"]
    assert "hint" not in section
    assert all("admin API" in c["hint"] for c in section["charts"])


def test_assemble_report_data_degrades_personal_vs_service_without_crashing(
    monkeypatch,
):
    """If _fetch_service_accounts (or the section builder) blows up, build()
    must not crash -- the section is simply omitted."""
    import cometx.cli.admin_growth_report as agr
    from cometx.cli.admin_growth_report import GrowthReporter

    api = MagicMock()
    api.config = {"comet.url_override": "https://c.example.com"}
    api.api_key = "K"
    api.get_workspaces.return_value = []
    cb = {
        "workspaces": [],
        "users": {
            "licensedUsers": [
                {
                    "username": "alice",
                    "email": "a@x",
                    "experimentCount": 1,
                    "dataLoggedMb": 0.0,
                    "lastUsedAt": 1,
                    "suspended": False,
                },
            ]
        },
    }
    monkeypatch.setattr(agr, "fetch_chargeback_report", lambda *a, **k: cb)

    def _boom(api):
        raise RuntimeError("boom")

    monkeypatch.setattr(agr, "_fetch_service_accounts", _boom)

    r = GrowthReporter(
        api,
        window="7d",
        units="month",
        active_window="60d",
        leaderboard_top_n=5,
    )
    data = r.build([])  # must not raise

    assert "personal_vs_service" not in data["sections"]


def _cb_lb():
    return {
        "workspaces": [
            {"name": "team-a", "members": [{"userName": "alice"}, {"userName": "bob"}]},
            {"name": "team-b", "members": [{"userName": "alice"}]},
        ],
        "users": {
            "licensedUsers": [
                {
                    "username": "alice",
                    "email": "a@x",
                    "experimentCount": 40,
                    "dataLoggedMb": 10.0,
                    "opikSpanCount": 5000,
                    "lastUsedAt": 1,
                },
                {
                    "username": "bob",
                    "email": "b@x",
                    "experimentCount": 1,
                    "dataLoggedMb": 2.0,
                    "opikSpanCount": 0,
                    "lastUsedAt": 1,
                },
            ]
        },
    }


def test_scope_chargeback_filters_workspaces_and_users():
    from cometx.cli.admin_growth_report import _scope_chargeback

    cb = {
        "workspaces": [
            {"name": "team-a", "members": [{"userName": "alice"}, {"userName": "bob"}]},
            {"name": "team-b", "members": [{"userName": "carol"}]},
        ],
        "users": {
            "licensedUsers": [
                {"username": "alice"},
                {"username": "bob"},
                {"username": "carol"},
            ]
        },
    }
    scoped = _scope_chargeback(cb, {"team-a"})
    assert [w["name"] for w in scoped["workspaces"]] == ["team-a"]
    assert {u["username"] for u in scoped["users"]["licensedUsers"]} == {"alice", "bob"}


def test_scope_label_org_vs_scoped():
    from cometx.cli.admin_growth_report import GrowthReporter

    org = GrowthReporter._scope_label(None, 165, 137)
    assert org.startswith("Org-wide: 165 workspaces, 137 users")
    assert org.endswith("(chargeback)")
    assert GrowthReporter._scope_label({"a", "b"}, 165, 137) == (
        "Scoped to 2 selected workspace(s) (per-user totals remain org-wide)"
    )
    assert GrowthReporter._scope_label(None, None, None) == "Org-wide (chargeback)"


def test_leaderboards_workspaces_org_wide_from_chargeback():
    from cometx.cli.admin_growth_report import GrowthReporter
    from cometx.cli.admin_growth_users import parse_users, parse_workspaces

    # chargeback with EXACT per-workspace experiment/project counts
    cb = {
        "workspaces": [
            {
                "name": "team-a",
                "numberOfExperiments": 40,
                "projects": [{}, {}],
                "members": [{"userName": "alice"}],
            },
            {
                "name": "team-b",
                "numberOfExperiments": 90,
                "projects": [{}],
                "members": [{"userName": "bob"}],
            },
        ],
        "users": {
            "licensedUsers": [
                {
                    "username": "alice",
                    "email": "a",
                    "opikSpanCount": 5000,
                    "lastUsedAt": 1,
                },
                {"username": "bob", "email": "b", "opikSpanCount": 0, "lastUsedAt": 1},
            ]
        },
    }
    api = MagicMock()
    r = GrowthReporter(api, window="7d", units="month", leaderboard_top_n=5)
    section = r._build_leaderboards_section(
        users=parse_users(cb),
        ws_records=parse_workspaces(cb),
    )
    exp_top = next(
        c for c in section["charts"] if c["id"] == "chart-lb-ws-experiments-top"
    )
    # exact per-workspace numberOfExperiments: team-b(90) > team-a(40)
    assert [row["label"] for row in exp_top["data"]["rows"]] == ["team-b", "team-a"]
    assert exp_top["data"]["rows"][0]["value"] == 90
    assert "exact" in exp_top["hint"]
    # projects leaderboard (exact) + Opik spans (per-user proxy) also present
    assert any(c["id"] == "chart-lb-ws-projects-top" for c in section["charts"])
    assert any(c["id"] == "chart-lb-ws-spans-top" for c in section["charts"])


def test_unified_section_org_overview_from_chargeback():
    from cometx.cli.admin_growth_report import GrowthReporter
    from cometx.cli.admin_growth_users import parse_users, parse_workspaces

    jan = int(
        datetime.datetime(2026, 1, 15, tzinfo=datetime.timezone.utc).timestamp() * 1000
    )
    feb = int(
        datetime.datetime(2026, 2, 15, tzinfo=datetime.timezone.utc).timestamp() * 1000
    )
    now_ms = int(
        datetime.datetime(2026, 7, 1, tzinfo=datetime.timezone.utc).timestamp() * 1000
    )
    cb = {
        "workspaces": [
            {
                "name": "team-a",
                "numberOfExperiments": 40,
                "projects": [{}, {}],
                "members": [{"userName": "alice"}],
            },
            {
                "name": "team-b",
                "numberOfExperiments": 90,
                "projects": [{}],
                "members": [{"userName": "bob"}],
            },
        ],
        "users": {
            "licensedUsers": [
                {
                    "username": "alice",
                    "email": "a",
                    "createdAt": jan,
                    "lastUsedAt": now_ms,
                },
                {
                    "username": "bob",
                    "email": "b",
                    "createdAt": feb,
                    "lastUsedAt": now_ms,
                },
            ]
        },
    }
    api = MagicMock()
    r = GrowthReporter(api, window="7d", units="month")
    section = r._build_unified_section(
        _win(),
        people_users=parse_users(cb),
        ws_records=parse_workspaces(cb),
        now_ms=now_ms,
        active_window_days=30,
    )
    assert section["title"] == "Organization overview (chargeback)"
    kpi_labels = [k["label"] for k in section["kpis"]]
    # "Total experiments" was replaced by a workspace-growth KPI (analog of the
    # Users section's user-growth KPI), labelled "New in <window> (% of base)".
    assert any(x.startswith("New in") for x in kpi_labels)
    assert "Total experiments" not in kpi_labels
    growth = next(k for k in section["kpis"] if k["label"].startswith("New in"))
    # alice's team-a created in Jan (in-window), bob's team-b in Feb (in-window),
    # none before the window start -> 0 before -> 0.0%, "+2 new"
    assert growth["sub"] == "+2 new (est. from earliest member)"
    ids = [c["id"] for c in section["charts"]]
    # Chargeback charts stay; SDK creation timelines move to per-product sections.
    assert "chart-unified-platform-mix" in ids
    assert "chart-unified-workspaces-created" not in ids
    assert "chart-unified-projects-cumulative" not in ids
    # Added-vs-deleted is now a bars+line combo, not a plain line chart.
    churn = next(
        c for c in section["charts"] if c["id"] == "chart-unified-workspace-churn"
    )
    assert churn["kind"] == "barsLine"
    assert churn["data"]["bars"] == ["added", "deleted"]
    assert churn["data"]["line"] == "rate"
    assert all("rate" in p["values"] for p in churn["data"]["points"])
    assert section["table"]["title"] == "By workspace (org-wide, chargeback)"


def test_unified_churn_rate_uses_net_surviving_base():
    """The churn growth-rate denominator is the *surviving* total at the start
    of each bucket, so it must subtract deletions from earlier buckets. If it
    only accumulated additions the Mar rate below would read 50% (1/2) instead
    of the correct 100% (1/1, since one of the two Jan workspaces was deleted
    in Feb)."""
    from cometx.cli.admin_growth_report import GrowthReporter
    from cometx.cli.admin_growth_users import parse_users, parse_workspaces

    def _ms(y, mo):
        return int(
            datetime.datetime(y, mo, 15, tzinfo=datetime.timezone.utc).timestamp()
            * 1000
        )

    jan, feb, mar = _ms(2026, 1), _ms(2026, 2), _ms(2026, 3)
    now_ms = int(
        datetime.datetime(2026, 7, 1, tzinfo=datetime.timezone.utc).timestamp() * 1000
    )
    cb = {
        "workspaces": [
            {"name": "ws-1", "members": [{"userName": "u1"}]},
            {"name": "ws-2", "members": [{"userName": "u2"}]},
            {"name": "ws-3", "members": [{"userName": "u3"}]},
        ],
        "users": {
            "licensedUsers": [
                {"username": "u1", "email": "u1", "createdAt": jan},
                # u2's workspace is fully deleted in Feb (every member deleted)
                {"username": "u2", "email": "u2", "createdAt": jan, "deletedAt": feb},
                {"username": "u3", "email": "u3", "createdAt": mar},
            ]
        },
    }
    api = MagicMock()
    r = GrowthReporter(api, window="7d", units="month")
    section = r._build_unified_section(
        _win(),
        people_users=parse_users(cb),
        ws_records=parse_workspaces(cb),
        now_ms=now_ms,
        active_window_days=30,
    )
    churn = next(
        c for c in section["charts"] if c["id"] == "chart-unified-workspace-churn"
    )
    by_key = {p["key"]: p["values"] for p in churn["data"]["points"]}
    # Surviving base at Mar start = 2 added - 1 deleted = 1 -> 1/1 * 100.
    assert by_key["2026-03"]["added"] == 1
    assert by_key["2026-03"]["rate"] == 100.0


def test_build_raises_when_chargeback_unavailable(monkeypatch):
    import cometx.cli.admin_growth_report as agr
    from cometx.cli.admin_growth_report import GrowthReporter, GrowthReportError

    def boom(*a, **k):
        raise RuntimeError("status_code: 403, body: {'message': 'Forbidden'}")

    monkeypatch.setattr(agr, "fetch_chargeback_report", boom)
    api = MagicMock()
    r = GrowthReporter(api, window="7d", units="month")
    try:
        r.build([])
        assert False, "expected GrowthReportError"
    except GrowthReportError as exc:
        assert "admin API key" in str(exc)


def test_generate_growth_report_signature_has_no_sdk_kwargs():
    import inspect

    from cometx.cli.admin_growth_report import generate_growth_report

    params = inspect.signature(generate_growth_report).parameters
    for gone in ("platforms", "limit", "include_users"):
        assert gone not in params
    for kept in (
        "window",
        "units",
        "active_window",
        "leaderboard_top_n",
        "exclude_personal",
        "personal_pattern",
        "output",
        "no_open",
    ):
        assert kept in params


def test_exclude_personal_drops_matching_workspaces_from_chargeback():
    from cometx.cli.admin_growth_report import GrowthReporter

    cb = {
        "workspaces": [
            {
                "name": "team-a",
                "numberOfExperiments": 5,
                "projects": [{}],
                "members": [{"userName": "a"}],
            },
            {
                "name": "personal-bob",
                "numberOfExperiments": 1,
                "projects": [{}],
                "members": [{"userName": "bob"}],
            },
        ],
        "users": {
            "report": [
                {"username": "a", "email": "a", "lastUsedAt": 1},
                {"username": "bob", "email": "b", "lastUsedAt": 1},
            ]
        },
    }
    api = MagicMock()
    r = GrowthReporter(
        api,
        window="7d",
        units="month",
        exclude_personal=True,
        personal_pattern=r"^personal-",
    )
    filtered = r._filter_personal_chargeback(cb)
    names = [w["name"] for w in filtered["workspaces"]]
    assert names == ["team-a"]
    # original is not mutated
    assert len(cb["workspaces"]) == 2


def test_units_adverb_avoids_dayly_typo():
    from cometx.cli.admin_growth_report import GrowthReporter

    api = MagicMock()
    assert GrowthReporter(api, window="7d", units="day")._units_adverb() == "daily"
    assert GrowthReporter(api, window="7d", units="week")._units_adverb() == "weekly"
    assert GrowthReporter(api, window="7d", units="month")._units_adverb() == "monthly"
    assert GrowthReporter(api, window="7d", units="hour")._units_adverb() == "hourly"


def test_scope_label_uses_post_filter_count_not_requested_args():
    from cometx.cli.admin_growth_report import GrowthReporter

    # three workspaces requested, but only one survives scoping/--exclude-personal
    label = GrowthReporter._scope_label({"a", "b", "c"}, 100, 50, scoped_count=1)
    assert label == (
        "Scoped to 1 selected workspace(s) (per-user totals remain org-wide)"
    )
    # falls back to len(scope) when no scoped_count is given
    assert GrowthReporter._scope_label({"a", "b"}, 100, 50) == (
        "Scoped to 2 selected workspace(s) (per-user totals remain org-wide)"
    )


def test_personal_vs_service_honors_empty_admin_set():
    # An empty set() from the admin service-accounts API is authoritative
    # ("zero service accounts"), so the split must report the admin_api source,
    # NOT fall back to the regex heuristic.
    from cometx.cli.admin_growth_report import GrowthReporter
    from cometx.cli.admin_growth_users import parse_users

    cb = {
        "workspaces": [],
        "users": {
            "report": [
                {
                    "username": "alice",
                    "email": "a",
                    "experimentCount": 5,
                    "dataLoggedMb": 2.0,
                    "lastUsedAt": 1,
                },
            ]
        },
    }
    api = MagicMock()
    r = GrowthReporter(api, window="7d", units="month")
    section = r._build_personal_vs_service_section(parse_users(cb), set())
    assert section is not None
    assert "admin API" in section["charts"][0]["hint"]


def test_short_api_error_redacts_header_cookie_body_on_fallback():
    from cometx.cli.admin_growth_report import _short_api_error

    # No status_code/'message' to parse -> fallback must drop the sensitive tail
    leaky = (
        "Connection failed headers: {'set-cookie': 'session=SECRET', "
        "'content-security-policy': \"base-uri 'self'\"} body: {'token': 'sk-xyz'}"
    )
    out = _short_api_error(leaky)
    assert out == "Connection failed"
    for bad in ("set-cookie", "SECRET", "content-security-policy", "sk-xyz", "body:"):
        assert bad not in out


def test_personal_vs_service_heuristic_hint_says_unavailable_not_empty():
    # service_account_names=None means the admin fetch FAILED; the hint must not
    # imply the admin API returned an empty list.
    from cometx.cli.admin_growth_report import GrowthReporter
    from cometx.cli.admin_growth_users import parse_users

    cb = {
        "workspaces": [],
        "users": {
            "report": [
                {
                    "username": "alice",
                    "email": "a",
                    "experimentCount": 5,
                    "dataLoggedMb": 2.0,
                    "lastUsedAt": 1,
                },
            ]
        },
    }
    api = MagicMock()
    r = GrowthReporter(api, window="7d", units="month")
    section = r._build_personal_vs_service_section(parse_users(cb), None)
    assert section is not None
    hint = section["charts"][0]["hint"]
    assert "unavailable" in hint
    assert "returned no service accounts" not in hint


def test_short_api_error_condenses_verbose_sdk_error():
    from cometx.cli.admin_growth_report import _short_api_error

    verbose = (
        "headers: {'content-security-policy': \"base-uri 'self'\"}, "
        "status_code: 400, body: {'code': 400, 'message': 'No such workspace!'}"
    )
    assert _short_api_error(verbose) == "400: No such workspace!"
    assert _short_api_error("plain boom") == "plain boom"
    long = "x" * 300
    assert len(_short_api_error(long)) <= 160


def test_num_renders_integral_floats_as_int():
    from cometx.cli.admin_growth_report import _num

    assert _num(3.0) == 3
    assert isinstance(_num(3.0), int)
    assert _num(3.5) == 3.5
    assert _num(7) == 7


def test_lb_value_rounds_fractional_metric_and_passes_none():
    # #5d: leaderboard bar values must render like workspace rows -- an
    # em_score of 12.7 (data_logged_mb folded in) shows as 13, not 12.7,
    # and a metric a user lacks (None) stays absent rather than becoming 0.
    from cometx.cli.admin_growth_report import _lb_value

    assert _lb_value(12.7) == 13
    assert isinstance(_lb_value(12.7), int)
    assert _lb_value(4.0) == 4
    assert _lb_value(None) is None
    assert _lb_value(9) == 9
