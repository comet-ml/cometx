#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ****************************************
#                              __
#   _________  ____ ___  ___  / /__  __
#  / ___/ __ \/ __ `__ \/ _ \/ __/ |/_/
# / /__/ /_/ / / / / / /  __/ /__>  <
# \___/\____/_/ /_/ /_/\___/\__/_/|_|
#
#
#  Copyright (c) 2024 Cometx Development
#      Team. All rights reserved.
# ****************************************
"""cometx admin growth-report — cross-platform use-case growth & rates, per
workspace/department (Opik + EM + MPM), rendered as a self-contained HTML page.

Distinct from `admin usage-report` (experiment counts over time, PDF/Streamlit):
growth-report tracks cross-platform use-case creation growth and rates.
"""

from __future__ import annotations

import dataclasses
import datetime
from collections import defaultdict

from cometx.utils import (  # noqa: F401
    format_time_key,
    get_next_time_key,
    parse_time_key,
)


@dataclasses.dataclass(frozen=True)
class CreationEvent:
    """A single use-case creation event (one of the 3 use-case kinds only)."""

    platform: str  # em | opik | mpm
    workspace: str
    use_case: str  # project or monitored-model name
    kind: str  # opik_project | em_project | mpm_model
    created: datetime.datetime


@dataclasses.dataclass(frozen=True)
class UsageMetric:
    """Secondary per-product adoption-depth metric (NOT a use case)."""

    platform: str  # opik | em | mpm
    workspace: str
    metric: str  # SPAN_COUNT | EXPERIMENT_COUNT | REGISTRY_MODELS |
    # REGISTRY_VERSIONS | PREDICTION_VOLUME
    value: float  # total (or snapshot for registry)
    project: str | None = None
    series: list | None = None  # [(time_key, value), ...]; None for snapshot


@dataclasses.dataclass(frozen=True)
class Window:
    start: datetime.datetime
    end: datetime.datetime
    units: str = "month"


KIND_LABELS = {
    "opik_project": "Opik projects",
    "em_project": "EM projects",
    "mpm_model": "Monitored models",
}

PLATFORM_LABELS = {"em": "EM", "opik": "Opik", "mpm": "MPM"}


def _ms_to_utc(ms) -> datetime.datetime:
    """Convert an epoch-milliseconds timestamp to a timezone-aware UTC
    datetime."""
    return datetime.datetime.fromtimestamp(ms / 1000, tz=datetime.timezone.utc)


def bucket_events(events, window, units) -> dict:
    """Count events by creation time-key within the window."""
    counts: dict = defaultdict(int)
    for ev in events:
        if window.start <= ev.created <= window.end:
            counts[format_time_key(ev.created, units)] += 1
    return dict(counts)


def continuous_series(counts: dict, units: str):
    """Earliest->latest keys, zero-filled via get_next_time_key."""
    if not counts:
        return []
    keys = sorted(counts)
    out, k = [], keys[0]
    while True:
        out.append((k, counts.get(k, 0)))
        if k == keys[-1]:
            break
        k = get_next_time_key(k, units)
    return out


def cumulative(series):
    total, out = 0, []
    for k, v in series:
        total += v
        out.append((k, total))
    return out


def growth_stats(events, window, units) -> dict:
    """{total, new_in_window, pct_growth} relative to installed base before window."""
    total = sum(1 for e in events if e.created <= window.end)
    new_in = sum(1 for e in events if window.start <= e.created <= window.end)
    before = sum(1 for e in events if e.created < window.start)
    pct = (new_in / before * 100.0) if before > 0 else 0.0
    return {"total": total, "new_in_window": new_in, "pct_growth": round(pct, 1)}


def generate_growth_report(
    api,
    workspaces,
    *,
    window="7d",
    units="month",
    platforms="em,opik,mpm",
    output="growth_report.html",
    no_open=False,
    limit=None,
):
    reporter = GrowthReporter(
        api, window=window, units=units, platforms=platforms, limit=limit
    )
    report_data = reporter.build(workspaces)  # events + usage -> report_data (C2-C7)
    path = write_growth_html(report_data, output)  # C8
    if not no_open:
        _open(path)
    return path


class GrowthReporter:
    def __init__(self, api, *, window, units, platforms, limit=None):
        self.api = api
        self.window = window
        self.units = units
        self.platforms = platforms
        self.limit = limit

    def build(self, workspaces):
        raise NotImplementedError  # filled in C2-C7

    def _collect_em(self, workspaces):
        """Collect EM `em_project` CreationEvents + EXPERIMENT_COUNT /
        REGISTRY_MODELS / REGISTRY_VERSIONS UsageMetrics.

        EM projects have no creation timestamp, so `created` is resolved via
        a proxy chain (see task-C4-context.md): probe a creation-timestamp
        key on the project json -> earliest experiment
        `start_server_timestamp` -> `lastUpdated`. This is ALL-TIME data;
        `self.window` is not applied here (that happens later for KPIs).
        """
        events: list = []
        usage: list = []

        if self.limit is not None:
            workspaces = list(workspaces)[: self.limit]

        for ws in workspaces:
            try:
                response = self.api._client.get_from_endpoint(
                    "projects", {"workspaceName": ws}
                )
                projects = (response or {}).get("projects", []) or []
            except Exception as exc:
                print(f"Warning: failed to list EM projects for workspace {ws}: {exc}")
                continue

            ws_experiment_total = 0
            ws_counts: dict = defaultdict(int)

            for project in projects:
                proj_name = project.get("projectName")
                try:
                    created = self._em_project_created(ws, proj_name, project)
                    counts = self._em_experiment_counts(ws, proj_name)
                    total = project.get("numberOfExperiments", sum(counts.values()))

                    events.append(
                        CreationEvent(
                            platform="em",
                            workspace=ws,
                            use_case=proj_name,
                            kind="em_project",
                            created=created,
                        )
                    )
                    usage.append(
                        UsageMetric(
                            platform="em",
                            workspace=ws,
                            metric="EXPERIMENT_COUNT",
                            value=total,
                            project=proj_name,
                            series=(
                                continuous_series(counts, self.units) if counts else []
                            ),
                        )
                    )
                    ws_experiment_total += total
                    for key, n in counts.items():
                        ws_counts[key] += n
                except Exception as exc:
                    print(
                        f"Warning: failed to collect EM project "
                        f"{ws}/{proj_name}: {exc}"
                    )
                    continue

            usage.append(
                UsageMetric(
                    platform="em",
                    workspace=ws,
                    metric="EXPERIMENT_COUNT",
                    value=ws_experiment_total,
                    project=None,
                    series=(
                        continuous_series(ws_counts, self.units) if ws_counts else []
                    ),
                )
            )

            try:
                model_names = self.api.get_registry_model_names(ws) or []
                version_total = sum(
                    len(self.api.get_registry_model_versions(ws, name) or [])
                    for name in model_names
                )
                usage.append(
                    UsageMetric(
                        platform="em",
                        workspace=ws,
                        metric="REGISTRY_MODELS",
                        value=len(model_names),
                        project=None,
                        series=None,
                    )
                )
                usage.append(
                    UsageMetric(
                        platform="em",
                        workspace=ws,
                        metric="REGISTRY_VERSIONS",
                        value=version_total,
                        project=None,
                        series=None,
                    )
                )
            except Exception as exc:
                print(f"Warning: failed to collect EM registry stats for {ws}: {exc}")

        return events, usage

    def _em_project_created(self, ws, proj_name, project):
        """Resolve an EM project's creation time via the documented proxy
        chain: creation-timestamp key (future-proof, currently absent) ->
        earliest experiment start_server_timestamp -> lastUpdated."""
        for key in ("createdAt", "creationDate", "creationDateMillis"):
            ms = project.get(key)
            if ms:
                return _ms_to_utc(ms)

        experiments = self.api.get_experiments(ws, proj_name) or []
        starts = [
            exp.start_server_timestamp
            for exp in experiments
            if getattr(exp, "start_server_timestamp", None)
        ]
        if starts:
            return _ms_to_utc(min(starts))

        return _ms_to_utc(project.get("lastUpdated"))

    def _em_experiment_counts(self, ws, proj_name):
        """Bucket this project's experiment start timestamps by `self.units`
        to build the all-time EXPERIMENT_COUNT over-time series."""
        experiments = self.api.get_experiments(ws, proj_name) or []
        counts: dict = defaultdict(int)
        for exp in experiments:
            ms = getattr(exp, "start_server_timestamp", None)
            if ms:
                counts[format_time_key(_ms_to_utc(ms), self.units)] += 1
        return dict(counts)


def write_growth_html(report_data, output):
    raise NotImplementedError  # filled in C8


def _open(path):
    import webbrowser

    webbrowser.open(f"file://{path}")
