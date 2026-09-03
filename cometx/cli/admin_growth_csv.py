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

import csv
import datetime
import os

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


USERS_FILENAME = "growth_users.csv"
WORKSPACES_FILENAME = "growth_workspaces.csv"
ORG_KPIS_FILENAME = "growth_org_kpis.csv"


def collect_org_kpis(users, ws_records, stats, growth, split, active_window_days):
    """Flatten the report's org-level numbers into (name, value, unit) triples.

    Each input block is optional: the HTML report degrades section-by-section
    when the chargeback payload is partial, and the CSV export follows suit --
    a missing block drops its metrics rather than failing the whole run.
    """
    kpis = []

    if stats is not None:
        kpis.append(("total_users", stats.get("total"), "count"))
        kpis.append(
            ("active_users_%dd" % active_window_days, stats.get("active"), "count")
        )
        kpis.append(("active_users_pct", stats.get("adoption_pct"), "percent"))

    if growth is not None:
        kpis.append(("new_users_in_window", growth.get("new_in"), "count"))
        kpis.append(("new_users_in_window_pct", growth.get("pct"), "percent"))

    kpis.append(("total_workspaces", len(ws_records), "count"))
    kpis.append(("total_projects", sum(w.num_projects for w in ws_records), "count"))
    kpis.append(
        ("total_experiments", sum(w.num_experiments for w in ws_records), "count")
    )
    kpis.append(("total_data_mb", sum(w.data_mb for w in ws_records), "megabytes"))

    if split is not None:
        for bucket in ("personal", "service"):
            totals = split.get(bucket) or {}
            kpis.append(("%s_experiments" % bucket, totals.get("experiments"), "count"))
            kpis.append(("%s_data_mb" % bucket, totals.get("data"), "megabytes"))
            kpis.append(("%s_spans" % bucket, totals.get("spans"), "count"))
        # Surface HOW the split was derived: the admin endpoint is optional and
        # silently falls back to a regex heuristic, which the dashboard should
        # be able to distinguish.
        kpis.append(("service_account_source", split.get("source"), "label"))

    return [(name, _num_or_empty(value), unit) for name, value, unit in kpis]


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as fp:
        writer = csv.writer(fp, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def write_growth_csvs(
    users,
    ws_records,
    kpis,
    out_dir,
    report_date,
    service_account_names=None,
):
    """Write the three fact tables into `out_dir`, returning the paths written.

    Files are flat (no Hive partition directories): `report_date` is a column
    on every row, and the caller's upload step chooses the S3 prefix.
    """
    if os.path.exists(out_dir) and not os.path.isdir(out_dir):
        raise NotADirectoryError(
            "--csv-dir %r exists but is not a directory." % out_dir
        )
    os.makedirs(out_dir, exist_ok=True)

    targets = [
        (
            USERS_FILENAME,
            USERS_HEADER,
            build_users_rows(users, report_date, service_account_names),
        ),
        (
            WORKSPACES_FILENAME,
            WORKSPACES_HEADER,
            build_workspaces_rows(ws_records, report_date),
        ),
        (ORG_KPIS_FILENAME, ORG_KPIS_HEADER, build_org_kpi_rows(kpis, report_date)),
    ]

    written = []
    for filename, header, rows in targets:
        path = os.path.join(out_dir, filename)
        _write_csv(path, header, rows)
        written.append(os.path.abspath(path))
    return written
