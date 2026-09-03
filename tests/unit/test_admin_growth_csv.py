#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit tests for cometx.cli.admin_growth_csv (Glue-ready CSV fact tables)."""

NOW = 1_720_000_000_000  # fixed ms; ~2024-07-03 UTC
DATE = "2026-09-03"


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
    assert alice["data_logged_mb"] == 8320.5
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
    assert rows == [[DATE, "research", 2, 24, 2130, 12422.75]]


def test_org_kpi_rows_are_long_format():
    from cometx.cli.admin_growth_csv import ORG_KPIS_HEADER, build_org_kpi_rows

    assert ORG_KPIS_HEADER == [
        "report_date",
        "metric_name",
        "metric_value",
        "metric_unit",
    ]
    rows = build_org_kpi_rows([("total_users", 6, "count")], DATE)
    assert rows == [[DATE, "total_users", 6, "count"]]


def test_empty_input_yields_no_rows():
    from cometx.cli.admin_growth_csv import (
        build_org_kpi_rows,
        build_users_rows,
        build_workspaces_rows,
    )

    assert build_users_rows([], DATE) == []
    assert build_workspaces_rows([], DATE) == []
    assert build_org_kpi_rows([], DATE) == []
