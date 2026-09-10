#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit tests for cometx.cli.admin_growth_csv (Glue-ready CSV fact tables)."""

import os.path

import cometx

NOW = 1_720_000_000_000  # fixed ms; ~2024-07-03 UTC
DATE = "2026-09-03"

# Directory containing the `cometx` package -- i.e. the repo root for a
# source checkout. Derived from the imported package rather than hardcoded
# so it stays correct regardless of where pytest is invoked from.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(cometx.__file__)))


def _users():
    from cometx.cli.admin_growth_users import UserRecord

    return [
        UserRecord(
            username="alice",
            email="a@x.com",
            created_at=NOW - 100,
            deleted_at=None,
            suspended=False,
            last_used_at=NOW,
            experiment_count=1240,
            data_logged_mb=8320.5,
            opik_span_count=45120,
            em_last_used_at=NOW,
            opik_last_used_at=None,
            workspaces=["research", "platform"],
        ),
        UserRecord(
            username="carol",
            email="c@x.com",
            created_at=NOW - 100,
            deleted_at=None,
            suspended=True,
            last_used_at=None,
            experiment_count=0,
            data_logged_mb=0.0,
            opik_span_count=None,
            em_last_used_at=None,
            opik_last_used_at=None,
            workspaces=[],
        ),
        UserRecord(
            username="dave",
            email="d@x.com",
            created_at=NOW - 100,
            deleted_at=NOW,
            suspended=False,
            last_used_at=NOW,
            experiment_count=5,
            data_logged_mb=1.0,
            opik_span_count=0,
            em_last_used_at=None,
            opik_last_used_at=None,
            workspaces=["research"],
        ),
    ]


def test_users_header_is_exact_and_ordered():
    from cometx.cli.admin_growth_csv import USERS_HEADER

    assert USERS_HEADER == [
        "report_date",
        "username",
        "email",
        "created_at",
        "last_used_at",
        "em_last_used_at",
        "opik_last_used_at",
        "is_suspended",
        "is_service_account",
        "experiment_count",
        "data_logged_mb",
        "opik_span_count",
        "deleted_at",
    ]


def test_multi_workspace_user_appears_exactly_once():
    """The core decision: no workspace column, so no row duplication."""
    from cometx.cli.admin_growth_csv import build_users_rows

    rows = build_users_rows(_users(), DATE)
    usernames = [r[1] for r in rows]
    assert usernames.count("alice") == 1


def test_deleted_users_excluded_suspended_included():
    from cometx.cli.admin_growth_csv import USERS_HEADER, build_users_rows

    rows = build_users_rows(_users(), DATE)
    by_name = {r[1]: dict(zip(USERS_HEADER, r)) for r in rows}
    assert "dave" not in by_name  # deleted_at set
    assert by_name["carol"]["is_suspended"] == 1


def test_epoch_ms_becomes_iso_date_and_none_becomes_empty():
    from cometx.cli.admin_growth_csv import USERS_HEADER, build_users_rows

    rows = build_users_rows(_users(), DATE)
    alice = dict(zip(USERS_HEADER, rows[0]))
    assert alice["last_used_at"] == "2024-07-03"
    assert alice["opik_last_used_at"] == ""


def test_report_date_on_every_row():
    from cometx.cli.admin_growth_csv import build_users_rows

    rows = build_users_rows(_users(), DATE)
    assert all(r[0] == DATE for r in rows)


def test_numbers_are_plain_no_separators_or_percent():
    from cometx.cli.admin_growth_csv import USERS_HEADER, build_users_rows

    rows = build_users_rows(_users(), DATE)
    alice = dict(zip(USERS_HEADER, rows[0]))
    assert alice["experiment_count"] == 1240
    # floats render as plain decimal strings (never scientific notation)
    assert float(alice["data_logged_mb"]) == 8320.5
    for value in alice.values():
        assert "," not in str(value)
        assert "%" not in str(value)


def test_missing_opik_span_count_is_empty_not_zero():
    """None means 'not reported', which is distinct from a real zero."""
    from cometx.cli.admin_growth_csv import USERS_HEADER, build_users_rows

    rows = build_users_rows(_users(), DATE)
    carol = {r[1]: dict(zip(USERS_HEADER, r)) for r in rows}["carol"]
    assert carol["opik_span_count"] == ""


def test_service_accounts_from_admin_api_names():
    from cometx.cli.admin_growth_csv import USERS_HEADER, build_users_rows

    rows = build_users_rows(_users(), DATE, service_account_names={"alice"})
    by_name = {r[1]: dict(zip(USERS_HEADER, r)) for r in rows}
    assert by_name["alice"]["is_service_account"] == 1
    assert by_name["carol"]["is_service_account"] == 0


def test_service_accounts_fall_back_to_heuristic():
    from cometx.cli.admin_growth_csv import USERS_HEADER, build_users_rows
    from cometx.cli.admin_growth_users import UserRecord

    svc = UserRecord(
        username="svc-nightly-etl",
        email="svc@x.com",
        created_at=NOW,
        deleted_at=None,
        suspended=False,
        last_used_at=NOW,
        experiment_count=1,
        data_logged_mb=1.0,
        opik_span_count=1,
        workspaces=[],
    )
    rows = build_users_rows([svc], DATE, service_account_names=None)
    row = dict(zip(USERS_HEADER, rows[0]))
    assert row["is_service_account"] == 1


def test_workspaces_header_and_rows():
    from cometx.cli.admin_growth_csv import (
        WORKSPACES_HEADER,
        build_workspaces_rows,
    )
    from cometx.cli.admin_growth_users import WorkspaceRecord

    assert WORKSPACES_HEADER == [
        "report_date",
        "workspace",
        "member_count",
        "num_projects",
        "num_experiments",
        "data_mb",
    ]
    ws = [
        WorkspaceRecord(
            name="research",
            num_experiments=2130,
            data_mb=12422.75,
            num_projects=24,
            members=("alice", "bob"),
        )
    ]
    rows = build_workspaces_rows(ws, DATE)
    # data_mb renders as a plain decimal string (never scientific notation)
    assert rows[0][:5] == [DATE, "research", 2, 24, 2130]
    assert float(rows[0][5]) == 12422.75


def test_org_kpi_rows_are_long_format():
    from cometx.cli.admin_growth_csv import ORG_KPIS_HEADER, build_org_kpi_rows

    assert ORG_KPIS_HEADER == [
        "report_date",
        "metric_name",
        "metric_value",
        "metric_unit",
        "metric_text",
    ]
    # A 3-tuple still works; metric_text defaults to empty.
    rows = build_org_kpi_rows([("total_users", 6, "count")], DATE)
    assert rows == [[DATE, "total_users", 6, "count", ""]]
    # A 4-tuple carries its text payload through.
    rows = build_org_kpi_rows(
        [("service_account_source", "", "label", "admin_api")], DATE
    )
    assert rows == [[DATE, "service_account_source", "", "label", "admin_api"]]


def test_label_metrics_keep_metric_value_numeric():
    """A single string in `metric_value` would make a Glue crawler type the
    whole column as `string`, forcing a cast on every SUM/AVG in QuickSight.
    Non-numeric payloads belong in `metric_text`."""
    from cometx.cli.admin_growth_csv import collect_org_kpis

    kpis = collect_org_kpis(
        users=[],
        ws_records=[],
        stats=None,
        growth=None,
        split={
            "personal": {"experiments": 1, "data": 1.0, "spans": 1},
            "service": {"experiments": 0, "data": 0, "spans": 0},
            "source": "heuristic",
        },
        active_window_days=60,
    )
    by_name = {name: (value, unit, text) for name, value, unit, text in kpis}
    value, unit, text = by_name["service_account_source"]
    assert value == ""  # nothing non-numeric in metric_value
    assert unit == "label"
    assert text == "heuristic"
    # The rule generalizes by UNIT, not by metric name: a `label` metric may
    # carry metric_text and leaves metric_value empty; every other unit must
    # leave metric_text empty and keep metric_value numeric-parseable (floats
    # render as plain decimal strings, so parse the VALUE rather than checking
    # its Python type).
    for name, (val, unit_, txt) in by_name.items():
        if unit_ == "label":
            assert val == "", name
        else:
            assert txt == "", name
            if val != "":
                float(val)  # raises if a non-numeric leaked into metric_value


def test_empty_input_yields_no_rows():
    from cometx.cli.admin_growth_csv import (
        build_org_kpi_rows,
        build_users_rows,
        build_workspaces_rows,
    )

    assert build_users_rows([], DATE) == []
    assert build_workspaces_rows([], DATE) == []
    assert build_org_kpi_rows([], DATE) == []


def test_collect_org_kpis_emits_expected_metrics():
    from cometx.cli.admin_growth_csv import collect_org_kpis
    from cometx.cli.admin_growth_users import WorkspaceRecord

    ws = [
        WorkspaceRecord(
            name="research",
            num_experiments=2130,
            data_mb=12422.75,
            num_projects=24,
            members=("alice",),
        )
    ]
    kpis = collect_org_kpis(
        users=_users(),
        ws_records=ws,
        stats={"total": 2, "active": 1, "adoption_pct": 50.0},
        growth={"new_in": 1, "before": 5, "pct": 20.0},
        split={
            "personal": {"experiments": 100, "data": 5.0, "spans": 10},
            "service": {"experiments": 900, "data": 50.0, "spans": 90},
            "source": "admin_api",
        },
        active_window_days=60,
    )
    by_name = {name: (value, unit) for name, value, unit, _text in kpis}

    assert by_name["total_users"] == (2, "count")
    assert by_name["active_users_60d"] == (1, "count")
    # floats render as plain decimal strings for Glue; compare numerically
    assert float(by_name["active_users_pct"][0]) == 50.0
    assert by_name["active_users_pct"][1] == "percent"
    assert by_name["new_users_in_window"] == (1, "count")
    assert by_name["total_workspaces"] == (1, "count")
    assert by_name["total_projects"] == (24, "count")
    assert by_name["total_experiments"] == (2130, "count")
    assert float(by_name["total_data_mb"][0]) == 12422.75
    assert by_name["total_data_mb"][1] == "megabytes"
    assert by_name["personal_experiments"] == (100, "count")
    assert by_name["service_experiments"] == (900, "count")
    # provenance rides in metric_text so metric_value stays numeric
    assert by_name["service_account_source"] == ("", "label")
    text_by_name = {name: text for name, _v, _u, text in kpis}
    assert text_by_name["service_account_source"] == "admin_api"


def test_collect_org_kpis_tolerates_missing_sections():
    """A degraded run (no stats/growth/split) still yields the workspace
    totals rather than raising."""
    from cometx.cli.admin_growth_csv import collect_org_kpis

    kpis = collect_org_kpis(
        users=[],
        ws_records=[],
        stats=None,
        growth=None,
        split=None,
        active_window_days=60,
    )
    by_name = {name: value for name, value, _unit, _text in kpis}
    assert by_name["total_workspaces"] == 0
    assert "total_users" not in by_name


def test_write_growth_csvs_creates_three_files(tmp_path):
    from cometx.cli.admin_growth_csv import write_growth_csvs

    out = tmp_path / "out"
    paths = write_growth_csvs(
        users=_users(),
        ws_records=[],
        kpis=[("total_users", 2, "count")],
        out_dir=str(out),
        report_date=DATE,
    )
    assert len(paths) == 3
    names = sorted(p.name for p in out.iterdir())
    assert names == [
        "growth_org_kpis.csv",
        "growth_users.csv",
        "growth_workspaces.csv",
    ]


def test_written_csv_round_trips_through_dictreader(tmp_path):
    import csv as _csv

    from cometx.cli.admin_growth_csv import write_growth_csvs

    out = tmp_path / "out"
    write_growth_csvs(
        users=_users(),
        ws_records=[],
        kpis=[],
        out_dir=str(out),
        report_date=DATE,
        service_account_names=set(),
    )
    with open(out / "growth_users.csv", newline="") as fp:
        rows = list(_csv.DictReader(fp))
    assert [r["username"] for r in rows] == ["alice", "carol"]
    assert rows[0]["report_date"] == DATE
    assert rows[0]["is_service_account"] == "0"


def test_empty_section_still_writes_header_only_file(tmp_path):
    """A Glue crawler needs the header to infer a schema even with no rows."""
    import csv as _csv

    from cometx.cli.admin_growth_csv import WORKSPACES_HEADER, write_growth_csvs

    out = tmp_path / "out"
    write_growth_csvs(
        users=[], ws_records=[], kpis=[], out_dir=str(out), report_date=DATE
    )
    with open(out / "growth_workspaces.csv", newline="") as fp:
        rows = list(_csv.reader(fp))
    assert rows == [WORKSPACES_HEADER]


def test_write_growth_csvs_creates_nested_missing_dirs(tmp_path):
    from cometx.cli.admin_growth_csv import write_growth_csvs

    out = tmp_path / "a" / "b" / "c"
    write_growth_csvs(
        users=[], ws_records=[], kpis=[], out_dir=str(out), report_date=DATE
    )
    assert (out / "growth_users.csv").exists()


def test_write_growth_csvs_rejects_path_that_is_a_file(tmp_path):
    from cometx.cli.admin_growth_csv import write_growth_csvs

    clash = tmp_path / "notadir"
    clash.write_text("x")
    try:
        write_growth_csvs(
            users=[],
            ws_records=[],
            kpis=[],
            out_dir=str(clash),
            report_date=DATE,
        )
    except Exception as exc:
        assert "not a directory" in str(exc).lower()
    else:
        raise AssertionError("expected an error when out_dir is a file")


def _non_ascii_user():
    from cometx.cli.admin_growth_users import UserRecord

    return UserRecord(
        username="josé.álvarez",
        email="josé.álvarez@exämple.com",
        created_at=NOW - 100,
        deleted_at=None,
        suspended=False,
        last_used_at=NOW,
        experiment_count=3,
        data_logged_mb=1.5,
        opik_span_count=7,
        em_last_used_at=NOW,
        opik_last_used_at=None,
        workspaces=["recherche"],
    )


def test_non_ascii_username_round_trips_as_utf8(tmp_path):
    """A non-ASCII username survives the write and reads back intact as UTF-8."""
    import csv as _csv

    from cometx.cli.admin_growth_csv import write_growth_csvs

    out = tmp_path / "out"
    write_growth_csvs(
        users=[_non_ascii_user()],
        ws_records=[],
        kpis=[],
        out_dir=str(out),
        report_date=DATE,
        service_account_names=set(),
    )
    with open(out / "growth_users.csv", newline="", encoding="utf-8") as fp:
        rows = list(_csv.DictReader(fp))
    assert [r["username"] for r in rows] == ["josé.álvarez"]
    assert rows[0]["email"] == "josé.álvarez@exämple.com"


def test_non_ascii_username_survives_a_c_locale_process(tmp_path):
    """The real regression: `_write_csv` must not depend on the ambient locale.

    Without an explicit `encoding="utf-8"`, `open()` falls back to the platform
    locale, and a `LANG=C` cron/systemd box raises UnicodeEncodeError mid-write
    -- leaving a truncated CSV for Glue to crawl. This cannot be simulated
    in-process: `open()` resolves its default encoding in C (patching
    `locale.getpreferredencoding` has no effect), and CPython's PEP 538 locale
    coercion turns `LANG=C` back into UTF-8 unless disabled. So the write runs
    in a genuinely ASCII-locale subprocess.
    """
    import csv as _csv
    import os
    import subprocess
    import sys

    out = tmp_path / "out"
    # The non-ASCII payload must NOT travel through argv: under an ASCII
    # locale CPython cannot decode its own command line and dies before
    # running a single statement (`python -c` would fail with "Unable to
    # decode the command from the command line"). Write the script to a
    # UTF-8 file with an explicit coding declaration and pass the PATH --
    # argv then stays pure ASCII while the source is still read as UTF-8.
    script = "\n".join(
        [
            "# -*- coding: utf-8 -*-",
            "import locale",
            "from cometx.cli.admin_growth_csv import write_growth_csvs",
            "from cometx.cli.admin_growth_users import UserRecord",
            # Guard against a vacuous pass: if the subprocess somehow came up
            # in UTF-8, fail loudly instead of claiming the bug is fixed.
            "enc = locale.getencoding().lower().replace('-', '').replace('_', '')",
            "assert enc in ('ascii', 'usascii', 'ansix3.41968'), enc",
            "u = UserRecord(username={username!r}, email={email!r},".format(
                username="josé.álvarez", email="josé.álvarez@exämple.com"
            ),
            "               created_at={0!r}, deleted_at=None, suspended=False,".format(
                NOW - 100
            ),
            "               last_used_at={0!r}, experiment_count=3,".format(NOW),
            "               data_logged_mb=1.5, opik_span_count=7,",
            "               em_last_used_at={0!r}, opik_last_used_at=None,".format(NOW),
            "               workspaces=['recherche'])",
            "write_growth_csvs(users=[u], ws_records=[], kpis=[],",
            "                  out_dir={0!r}, report_date={1!r},".format(
                str(out), DATE
            ),
            "                  service_account_names=set())",
        ]
    )
    script_path = tmp_path / "write_non_ascii.py"
    script_path.write_text(script, encoding="utf-8")

    env = dict(os.environ)
    env.update(
        {
            "LC_ALL": "C",
            "LANG": "C",
            # Defeat the two CPython escape hatches that would silently restore
            # UTF-8 and make this test vacuous.
            "PYTHONCOERCECLOCALE": "0",
            "PYTHONUTF8": "0",
            "PYTHONIOENCODING": "utf-8",  # so a traceback can still be printed
            # Running a script BY PATH sets sys.path[0] to the script's own
            # directory (tmp_path), not the CWD, so `import cometx` would fail
            # on a checkout without an editable install (as on CI). Point at
            # the repo root explicitly rather than relying on install mode.
            "PYTHONPATH": _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", ""),
        }
    )
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    with open(out / "growth_users.csv", newline="", encoding="utf-8") as fp:
        rows = list(_csv.DictReader(fp))
    assert [r["username"] for r in rows] == ["josé.álvarez"]


def test_deleted_at_column_is_last_and_empty_for_emitted_rows():
    """`deleted_at` is exposed so the column is present and typed for Glue,
    but deleted users are still filtered out -- so it is always empty on the
    rows we actually emit. Appended last, per the stable-column-order rule."""
    from cometx.cli.admin_growth_csv import USERS_HEADER, build_users_rows

    assert USERS_HEADER[-1] == "deleted_at"

    rows = build_users_rows(_users(), DATE)
    by_name = {r[1]: dict(zip(USERS_HEADER, r)) for r in rows}
    assert "dave" not in by_name  # still excluded (deleted_at was set)
    assert all(r["deleted_at"] == "" for r in by_name.values())


def test_deleted_users_kpi_counts_the_excluded_rows():
    from cometx.cli.admin_growth_csv import collect_org_kpis

    kpis = collect_org_kpis(
        users=_users(),  # alice, carol (suspended), dave (deleted)
        ws_records=[],
        stats={"total": 2, "active": 1, "adoption_pct": 50.0},
        growth=None,
        split=None,
        active_window_days=60,
    )
    by_name = {name: value for name, value, _unit, _text in kpis}
    assert by_name["deleted_users"] == 1


def _kpis_for(users):
    from cometx.cli.admin_growth_csv import collect_org_kpis

    non_suspended = [u for u in users if not u.suspended]
    return {
        name: value
        for name, value, _unit, _text in collect_org_kpis(
            users=users,
            ws_records=[],
            stats={"total": len(non_suspended), "active": 1, "adoption_pct": 50.0},
            growth=None,
            split=None,
            active_window_days=60,
        )
    }


def test_users_in_table_kpi_equals_the_row_count():
    """`users_in_table` is published directly so a dashboard never has to
    derive the row count from the other metrics."""
    from cometx.cli.admin_growth_csv import build_users_rows

    users = _users()  # 1 plain, 1 suspended, 1 deleted
    assert _kpis_for(users)["users_in_table"] == len(build_users_rows(users, DATE))


def test_users_in_table_holds_when_a_user_is_both_deleted_and_suspended():
    """Regression: the arithmetic identity
    `total_users - deleted_users + suspended` UNDERCOUNTS when a user carries
    both flags -- they are absent from total_users (suspended) *and* counted in
    deleted_users (deleted), so subtracting removes them twice. The published
    `users_in_table` metric must stay correct regardless."""
    from cometx.cli.admin_growth_csv import build_users_rows
    from cometx.cli.admin_growth_users import UserRecord

    def _u(name, suspended=False, deleted_at=None):
        return UserRecord(
            username=name,
            email=name + "@x.com",
            created_at=NOW - 100,
            deleted_at=deleted_at,
            suspended=suspended,
            last_used_at=NOW,
            experiment_count=1,
            data_logged_mb=1.0,
            opik_span_count=1,
            workspaces=[],
        )

    users = [
        _u("live"),
        _u("suspended", suspended=True),
        _u("deleted", deleted_at=NOW),
        _u("deleted_and_suspended", suspended=True, deleted_at=NOW),
    ]
    kpis = _kpis_for(users)
    rows = build_users_rows(users, DATE)

    assert len(rows) == 2  # live + suspended
    assert kpis["users_in_table"] == len(rows)

    # Demonstrate the old identity is genuinely wrong here, so nobody
    # reintroduces it as "simpler".
    suspended_live = sum(1 for u in users if u.suspended and u.deleted_at is None)
    broken = kpis["total_users"] - kpis["deleted_users"] + suspended_live
    assert broken != len(rows)


def test_floats_never_use_scientific_notation():
    """Athena's CSV SerDe does not parse `1e-05` / `1e+20` as a double -- the
    value silently becomes NULL in the dashboard. Python's default repr flips
    to exponent form outside roughly 1e-5..1e16, and the low end is reachable
    (a near-empty workspace reporting a tiny data_mb)."""
    from cometx.cli.admin_growth_csv import _num_or_empty

    for value in (1e-5, 1e-7, 1e16, 1e20, 1.5e16):
        rendered = str(_num_or_empty(value))
        assert "e" not in rendered.lower(), (value, rendered)


def test_real_world_floats_round_trip_exactly():
    """Values of the magnitude actually seen in chargeback must not be
    reformatted or lose precision."""
    from cometx.cli.admin_growth_csv import _num_or_empty

    for value in (20545.68, 88834.25, 71200.75, 12422.75, 0.0, 118.5):
        assert float(_num_or_empty(value)) == value


def test_no_float_is_quantized():
    """Avoiding scientific notation must not cost precision. A fixed format
    like `%.6f` would render 0.1234567 as 0.123457 -- trading a silent NULL
    in Athena for a silently wrong number, which is worse."""
    from cometx.cli.admin_growth_csv import _num_or_empty

    for value in (
        0.1234567,
        1.23456789,
        1 / 3,
        0.1 + 0.2,
        1e-7,
        1e-5,
        1.5e16,
        123456789012345.6,
    ):
        rendered = str(_num_or_empty(value))
        assert "e" not in rendered.lower(), (value, rendered)
        assert float(rendered) == value, (value, rendered)


def test_non_finite_floats_become_empty_not_garbage():
    """`nan`/`inf` have no honest CSV representation; emitting the literal
    text would make Glue type the column as a string."""
    from cometx.cli.admin_growth_csv import _num_or_empty

    assert _num_or_empty(float("nan")) == ""
    assert _num_or_empty(float("inf")) == ""
    assert _num_or_empty(float("-inf")) == ""


def test_integers_pass_through_unformatted():
    """Ints have arbitrary precision and never go exponential; leave them be."""
    from cometx.cli.admin_growth_csv import _num_or_empty

    assert _num_or_empty(42) == 42
    assert _num_or_empty(0) == 0
    assert _num_or_empty(388400) == 388400


def test_workspace_numerics_go_through_the_float_guard():
    """Regression: `build_workspaces_rows` emitted raw floats, so a workspace
    with a tiny `totalSizeInMb` wrote `4e-06` (NULL to Athena) and a NaN/inf
    from the API wrote literal text (forcing Glue to type the column as
    `string`). The users table and org KPIs already guarded this."""
    from cometx.cli.admin_growth_csv import build_workspaces_rows
    from cometx.cli.admin_growth_users import WorkspaceRecord

    ws = [
        WorkspaceRecord(
            name="tiny",
            num_experiments=1,
            data_mb=4e-06,
            num_projects=1,
            members=(),
        ),
        WorkspaceRecord(
            name="broken",
            num_experiments=float("nan"),
            data_mb=float("inf"),
            num_projects=1,
            members=(),
        ),
    ]
    rows = {r[1]: r for r in build_workspaces_rows(ws, DATE)}

    tiny = str(rows["tiny"][5])
    assert "e" not in tiny.lower()
    assert float(tiny) == 4e-06  # exact, not quantized

    assert rows["broken"][4] == ""  # nan
    assert rows["broken"][5] == ""  # inf


def test_scope_kpi_distinguishes_scoped_from_org_wide():
    """Without this, a workspace-filtered export is byte-shaped exactly like
    an org-wide one -- same filenames, same headers -- and `total_workspaces`
    just reads lower. Loaded into the same Glue partition that looks like an
    org that shrank overnight."""
    from cometx.cli.admin_growth_csv import collect_org_kpis
    from cometx.cli.admin_growth_users import WorkspaceRecord

    def _ws(*names):
        return [
            WorkspaceRecord(
                name=n, num_experiments=1, data_mb=1.0, num_projects=1, members=()
            )
            for n in names
        ]

    def _kpis(scope, ws_records):
        kpis = collect_org_kpis(
            users=[],
            ws_records=ws_records,
            stats=None,
            growth=None,
            split=None,
            active_window_days=60,
            scope=scope,
        )
        return {name: (value, unit, text) for name, value, unit, text in kpis}

    org = _kpis(None, _ws("alpha", "beta"))["scope"]
    assert org == ("", "label", "organization")

    scoped = _kpis({"beta", "alpha"}, _ws("alpha", "beta"))
    value, unit, text = scoped["scope"]
    assert value == ""  # metric_value stays numeric-only
    assert unit == "label"
    assert text == "workspaces:alpha,beta"  # sorted, so it is stable per run
    # request matched reality, so no separate `scope_requested` row
    assert "scope_requested" not in scoped


def test_scope_reports_what_was_exported_not_what_was_asked_for():
    """Regression: `scope` serialized the raw CLI request, so it could name a
    workspace that does not exist or was dropped by `--exclude-personal` --
    sending a dashboard that filters on it to an empty result. It must
    describe the records actually present; the request is preserved
    separately when the two differ."""
    from cometx.cli.admin_growth_csv import collect_org_kpis
    from cometx.cli.admin_growth_users import WorkspaceRecord

    surviving = [
        WorkspaceRecord(
            name="team-a", num_experiments=1, data_mb=1.0, num_projects=1, members=()
        )
    ]
    kpis = {
        name: text
        for name, _v, _u, text in collect_org_kpis(
            users=[],
            ws_records=surviving,
            stats=None,
            growth=None,
            split=None,
            active_window_days=60,
            scope={"team-a", "ghost"},
        )
    }
    assert kpis["scope"] == "workspaces:team-a"  # only what is really there
    assert kpis["scope_requested"] == "workspaces:ghost,team-a"


def test_booleans_in_numeric_fields_become_empty_not_true_false():
    """`bool` subclasses `int`, so a malformed payload carrying `true` in a
    numeric field would write the literal `True` into the column and make a
    Glue crawler type it as `string`. Coercing to 1/0 would be worse -- it
    invents a count the source never reported."""
    from cometx.cli.admin_growth_csv import (
        USERS_HEADER,
        _num_or_empty,
        build_users_rows,
    )
    from cometx.cli.admin_growth_users import parse_users

    assert _num_or_empty(True) == ""
    assert _num_or_empty(False) == ""
    # a real number is untouched by the guard
    assert _num_or_empty(0) == 0
    assert _num_or_empty(5) == 5

    payload = {
        "workspaces": [],
        "users": {
            "report": [
                {
                    "username": "a",
                    "email": "a@x.com",
                    "createdAt": NOW,
                    "lastUsedAt": NOW,
                    "experimentCount": True,
                    "dataLoggedMb": False,
                    "opikSpanCount": 5,
                    "suspended": True,
                    "deletedAt": None,
                }
            ]
        },
    }
    row = dict(zip(USERS_HEADER, build_users_rows(parse_users(payload), DATE)[0]))
    assert row["experiment_count"] == ""
    assert row["data_logged_mb"] == ""
    assert row["opik_span_count"] == 5
    # the genuine boolean columns still emit 0/1 -- they never go through
    # `_num_or_empty`, and this guards against a fix that breaks them
    assert row["is_suspended"] == 1


def test_build_org_kpi_rows_guards_metric_value():
    """The single writer of `metric_value` applies the same numeric guard the
    users and workspaces builders apply, so a hand-assembled KPI tuple cannot
    put a `4e-06` (NULL to Athena's CSV SerDe) or a bare `True` (which makes a
    Glue crawler type the column `string`) into an otherwise-numeric column."""
    from cometx.cli.admin_growth_csv import build_org_kpi_rows

    rows = build_org_kpi_rows(
        [
            ("total_data_mb", 4e-06, "megabytes"),
            ("some_flag", True, "count"),
            ("total_users", 6, "count"),
        ],
        DATE,
    )
    values = [r[2] for r in rows]
    assert "e" not in str(values[0]) and float(values[0]) == 4e-06
    assert values[1] == ""  # a bool is not a count the source ever reported
    assert values[2] == 6  # ordinary ints pass through untouched


def test_build_org_kpi_rows_guard_is_idempotent():
    """`collect_org_kpis` already normalizes; running the guard again on its
    output must not change anything, or the two would disagree."""
    from cometx.cli.admin_growth_csv import build_org_kpi_rows, collect_org_kpis

    kpis = collect_org_kpis(
        users=[],
        ws_records=[],
        stats={"total": 4, "active": 2, "adoption_pct": 50.0},
        growth=None,
        split=None,
        active_window_days=60,
    )
    once = build_org_kpi_rows(kpis, DATE)
    twice = build_org_kpi_rows(
        [(r[1], r[2], r[3], r[4]) for r in once],
        DATE,
    )
    assert once == twice


def test_a_failed_write_publishes_nothing(tmp_path):
    """The three tables are one partition. A run that died halfway would
    otherwise leave this month's users table beside last month's workspaces
    table, and the upload step syncs the directory -- a mismatch nothing about
    the files announces."""
    from unittest.mock import patch

    import pytest

    import cometx.cli.admin_growth_csv as mod

    out = tmp_path / "out"
    out.mkdir()
    stale = out / mod.WORKSPACES_FILENAME
    stale.write_text("stale-from-last-run\n", encoding="utf-8")

    real_write = mod._write_csv_rows
    calls = []

    def flaky(fp, header, rows):
        calls.append(fp)
        if len(calls) == 3:
            raise OSError(28, "No space left on device")
        return real_write(fp, header, rows)

    with patch.object(mod, "_write_csv_rows", flaky):
        with pytest.raises(OSError):
            mod.write_growth_csvs([], [], [], str(out), DATE)

    # Nothing new published, the previous partition untouched, no temporaries
    # left behind for the uploader to find.
    assert sorted(p.name for p in out.iterdir()) == [mod.WORKSPACES_FILENAME]
    assert stale.read_text(encoding="utf-8") == "stale-from-last-run\n"


def test_a_successful_write_leaves_no_temporaries(tmp_path):
    from cometx.cli.admin_growth_csv import write_growth_csvs

    out = tmp_path / "out"
    write_growth_csvs([], [], [], str(out), DATE)
    assert not [p.name for p in out.iterdir() if p.name.endswith(".tmp")]


def test_write_replaces_a_previous_runs_files(tmp_path):
    """os.replace over an existing file, not an append or a failure."""
    from cometx.cli.admin_growth_csv import USERS_FILENAME, write_growth_csvs

    out = tmp_path / "out"
    out.mkdir()
    (out / USERS_FILENAME).write_text("old\n", encoding="utf-8")
    write_growth_csvs([], [], [], str(out), DATE)
    assert (out / USERS_FILENAME).read_text(encoding="utf-8").startswith("report_date,")


def test_staging_does_not_use_a_predictable_temp_name(tmp_path):
    """`out_dir` is an operator-supplied path that may be world-writable. A
    fixed `<name>.tmp` lets anyone pre-plant a symlink there and have the
    export truncate whatever it points at; `mkstemp` opens O_CREAT|O_EXCL, so
    the planted name is simply never used."""
    from cometx.cli.admin_growth_csv import USERS_FILENAME, write_growth_csvs

    out = tmp_path / "out"
    out.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("precious\n", encoding="utf-8")
    (out / (USERS_FILENAME + ".tmp")).symlink_to(victim)

    write_growth_csvs([], [], [], str(out), DATE)

    assert victim.read_text(encoding="utf-8") == "precious\n"
    assert (out / USERS_FILENAME).read_text(encoding="utf-8").startswith("report_date,")


def test_concurrent_runs_get_distinct_temp_names(tmp_path):
    """Two runs staging into one directory must not write, or clean up, each
    other's staged data under a shared fixed name."""
    from cometx.cli.admin_growth_csv import USERS_HEADER, _stage_csv

    out = tmp_path / "out"
    out.mkdir()
    a = _stage_csv(str(out), "growth_users.csv", USERS_HEADER, [])
    b = _stage_csv(str(out), "growth_users.csv", USERS_HEADER, [])
    assert a != b
    assert os.path.exists(a) and os.path.exists(b)


def test_a_failed_commit_rolls_back_to_the_previous_generation(tmp_path):
    """A failure partway through the commit must leave the directory on the
    generation it held on entry -- not a mix of old and new files."""
    from unittest.mock import patch

    import pytest

    import cometx.cli.admin_growth_csv as mod

    out = tmp_path / "out"
    out.mkdir()
    previous = {}
    for name in (mod.USERS_FILENAME, mod.WORKSPACES_FILENAME, mod.ORG_KPIS_FILENAME):
        previous[name] = "previous generation of %s\n" % name
        (out / name).write_text(previous[name], encoding="utf-8")

    real_replace = os.replace
    seen = []

    def flaky_replace(src, dst):
        # Fail the workspaces table's publish -- the second of three, so one
        # file has already been replaced and two have not. Only the publish
        # itself lands on the final filename (moving the displaced file aside
        # targets a .bak), so the first hit is the one to fail; the second is
        # the rollback restoring it, which must be allowed through.
        if str(dst).endswith(mod.WORKSPACES_FILENAME):
            seen.append(dst)
            if len(seen) == 1:
                raise OSError(5, "I/O error")
        return real_replace(src, dst)

    with patch.object(mod.os, "replace", flaky_replace):
        with pytest.raises(OSError):
            mod.write_growth_csvs([], [], [], str(out), DATE)

    for name, content in previous.items():
        assert (out / name).read_text(encoding="utf-8") == content, name
    assert sorted(p.name for p in out.iterdir()) == sorted(previous)


def test_published_files_are_readable_not_mkstemp_private(tmp_path):
    """mkstemp creates 0600. These are published data files whose upload step
    may run as another user; narrowing the mode from what a plain open() would
    produce would break such a pipeline silently."""
    import stat

    from cometx.cli.admin_growth_csv import USERS_FILENAME, write_growth_csvs

    out = tmp_path / "out"
    write_growth_csvs([], [], [], str(out), DATE)

    reference = tmp_path / "reference.csv"
    reference.write_text("x\n", encoding="utf-8")

    published = stat.S_IMODE((out / USERS_FILENAME).stat().st_mode)
    assert published == stat.S_IMODE(reference.stat().st_mode)
