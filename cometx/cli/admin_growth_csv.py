# -*- coding: utf-8 -*-
"""Glue-ready CSV fact tables derived from the admin chargeback report.

Three flat tables -- users, workspaces, org KPIs -- intended for
S3 -> AWS Glue -> Athena -> QuickSight. Deliberately built from the parsed
`UserRecord`/`WorkspaceRecord` objects rather than from the HTML report's
`report_data`: that payload carries display-formatted strings (thousands
separators via `_num()`, `%`-suffixed rates) which would make a Glue crawler
type the columns as `string` and silently break aggregation.

This module must not import from `admin_growth_report.py` -- that module
imports FROM `admin_growth_users.py`, so importing back would be circular.
Callers pass already-parsed records in.
"""

import datetime

from cometx.cli.admin_growth_users import _looks_like_service_account

USERS_HEADER = [
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

WORKSPACES_HEADER = [
    "report_date",
    "workspace",
    "member_count",
    "num_projects",
    "num_experiments",
    "data_mb",
]

ORG_KPIS_HEADER = [
    "report_date",
    "metric_name",
    "metric_value",
    "metric_unit",
]


def _ms_to_date(ms) -> str:
    """Epoch-ms -> `YYYY-MM-DD` (UTC). Empty string when absent or unparseable
    -- an empty CSV field is what Glue reads as NULL."""
    if ms is None:
        return ""
    try:
        return datetime.datetime.fromtimestamp(
            ms / 1000, tz=datetime.timezone.utc
        ).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError, OverflowError):
        return ""


def _num_or_empty(value):
    """Pass numbers through unformatted; `None` becomes an empty field.

    `None` is preserved as empty rather than coerced to 0 because the two mean
    different things: chargeback omits `opikSpanCount` for deployments without
    Opik, which is not the same as a user with zero spans.
    """
    return "" if value is None else value


def build_users_rows(users, report_date, service_account_names=None):
    """One row per non-deleted licensed user.

    No workspace column by design: chargeback reports these metrics per-user,
    so emitting a row per (user, workspace) would repeat each user's totals and
    make `SUM(experiment_count)` over-count. Per-workspace totals live in the
    workspaces table, where they are exact.

    `service_account_names`, when a set, is the authoritative list from the
    admin `/admin/service-accounts` endpoint (matched on username OR email);
    when `None`, falls back to the same labeled regex heuristic the HTML report
    uses.
    """
    if service_account_names is not None:

        def is_service(user):
            return (
                user.username in service_account_names
                or user.email in service_account_names
            )

    else:
        is_service = _looks_like_service_account

    rows = []
    for user in users:
        if user.deleted_at is not None:
            continue
        rows.append(
            [
                report_date,
                user.username,
                user.email,
                _ms_to_date(user.created_at),
                _ms_to_date(user.last_used_at),
                _ms_to_date(user.em_last_used_at),
                _ms_to_date(user.opik_last_used_at),
                1 if user.suspended else 0,
                1 if is_service(user) else 0,
                _num_or_empty(user.experiment_count),
                _num_or_empty(user.data_logged_mb),
                _num_or_empty(user.opik_span_count),
            ]
        )
    return rows


def build_workspaces_rows(ws_records, report_date):
    """One row per workspace. Exact per-workspace totals, no double-counting."""
    return [
        [
            report_date,
            w.name,
            len(w.members),
            w.num_projects,
            w.num_experiments,
            w.data_mb,
        ]
        for w in ws_records
    ]


def build_org_kpi_rows(kpis, report_date):
    """One row per org-level metric, long format.

    Long rather than wide so new metrics arrive as new ROWS: the Glue schema
    never changes and existing partitions stay readable.
    """
    return [[report_date, name, value, unit] for name, value, unit in kpis]
