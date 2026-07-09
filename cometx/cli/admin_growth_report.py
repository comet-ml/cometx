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
                    # Fetch the project's experiments once and reuse the list
                    # for both the creation proxy and the over-time series.
                    experiments = self.api.get_experiments(ws, proj_name) or []
                    created = self._em_project_created(project, experiments)
                    counts = self._em_experiment_counts(experiments)
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

    def _em_project_created(self, project, experiments):
        """Resolve an EM project's creation time via the documented proxy
        chain: creation-timestamp key (future-proof, currently absent) ->
        earliest experiment start_server_timestamp -> lastUpdated.

        `experiments` is the pre-fetched experiment list for the project
        (fetched once by the caller and shared with `_em_experiment_counts`).
        """
        for key in ("createdAt", "creationDate", "creationDateMillis"):
            ms = project.get(key)
            if ms:
                return _ms_to_utc(ms)

        starts = [
            exp.start_server_timestamp
            for exp in experiments
            if getattr(exp, "start_server_timestamp", None)
        ]
        if starts:
            return _ms_to_utc(min(starts))

        return _ms_to_utc(project.get("lastUpdated"))

    def _collect_opik(self, workspaces):
        """Collect Opik `opik_project` CreationEvents + SPAN_COUNT
        UsageMetrics.

        This is ALL-TIME data (interval_start=project.created_at,
        interval_end=now); `self.window` is not applied here (that happens
        later for KPIs). Degrades gracefully to `([], [])` if `opik` is not
        installed.
        """
        try:
            import opik
        except ImportError:
            print("Warning: opik not installed; skipping Opik collection")
            return [], []

        from cometx.cli.smoke_test import get_opik_config

        events: list = []
        usage: list = []

        if self.limit is not None:
            workspaces = list(workspaces)[: self.limit]

        api_key = self.api.config["comet.api_key"]
        comet_base_url = self.api.config["comet.url_override"].rstrip("/")
        host = get_opik_config(comet_base_url)

        interval = {
            "hour": "HOURLY",
            "day": "DAILY",
            "week": "WEEKLY",
            "month": "WEEKLY",
        }.get(self.units, "WEEKLY")

        now = datetime.datetime.now(datetime.timezone.utc)

        for ws in workspaces:
            try:
                client = opik.Opik(workspace=ws, api_key=api_key, host=host)
            except Exception as exc:
                print(f"Warning: failed to init Opik client for workspace {ws}: {exc}")
                continue

            try:
                projects = []
                page_num = 1
                while True:
                    page = client.rest_client.projects.find_projects(
                        page=page_num, size=100
                    )
                    content = page.content or []
                    projects.extend(content)
                    if not content or len(projects) >= page.total:
                        break
                    page_num += 1
            except Exception as exc:
                print(
                    f"Warning: failed to list Opik projects for workspace {ws}: {exc}"
                )
                continue

            ws_counts: dict = defaultdict(float)

            for project in projects:
                try:
                    if project.created_at is not None:
                        events.append(
                            CreationEvent(
                                platform="opik",
                                workspace=ws,
                                use_case=project.name,
                                kind="opik_project",
                                created=project.created_at,
                            )
                        )

                    interval_start = project.created_at or datetime.datetime(
                        1970, 1, 1, tzinfo=datetime.timezone.utc
                    )
                    resp = client.rest_client.projects.get_project_metrics(
                        project.id,
                        metric_type="SPAN_COUNT",
                        interval=interval,
                        interval_start=interval_start,
                        interval_end=now,
                    )
                    counts: dict = defaultdict(float)
                    for result in resp.results or []:
                        for dp in result.data or []:
                            counts[format_time_key(dp.time, self.units)] += (
                                dp.value or 0
                            )

                    usage.append(
                        UsageMetric(
                            platform="opik",
                            workspace=ws,
                            metric="SPAN_COUNT",
                            value=sum(counts.values()),
                            project=project.name,
                            series=(
                                continuous_series(dict(counts), self.units)
                                if counts
                                else []
                            ),
                        )
                    )
                    for key, val in counts.items():
                        ws_counts[key] += val
                except Exception as exc:
                    print(
                        f"Warning: failed to collect Opik project "
                        f"{ws}/{getattr(project, 'name', '?')}: {exc}"
                    )
                    continue

            usage.append(
                UsageMetric(
                    platform="opik",
                    workspace=ws,
                    metric="SPAN_COUNT",
                    value=sum(ws_counts.values()),
                    project=None,
                    series=(
                        continuous_series(dict(ws_counts), self.units)
                        if ws_counts
                        else []
                    ),
                )
            )

        return events, usage

    def _collect_mpm(self, workspaces):
        """Collect MPM `mpm_model` CreationEvents + PREDICTION_VOLUME
        UsageMetrics.

        MPM models have no reliable creation timestamp, so `created` is
        resolved via the probe -> proxy -> count chain (see
        task-C6-context.md): probe `get_model_details` for a
        creation-timestamp key -> earliest prediction datapoint with y>0
        -> no CreationEvent (model still counted, just not on the
        created-over-time chart). The wide predictions series is fetched
        ONCE per model and reused for both the creation proxy and the
        PREDICTION_VOLUME usage metric. This is ALL-TIME data; `self.window`
        is not applied here. Degrades gracefully to `([], [])` if
        `comet_mpm` is not installed, or if the model-inventory endpoint is
        unavailable (e.g. MPM not provisioned for this account -> 404).
        """
        try:
            import comet_mpm
        except ImportError:
            print("Warning: comet_mpm not installed; skipping MPM collection")
            return [], []

        events: list = []
        usage: list = []

        if self.limit is not None:
            workspaces = list(workspaces)[: self.limit]

        try:
            mpm = comet_mpm.API(api_key=self.api.config["comet.api_key"])
            resp = mpm._client.get("api/mpm/v3/workspaces")
            all_workspaces = (resp or {}).get("workspaces", []) or []
        except Exception as exc:
            print(f"Warning: failed to list MPM workspaces: {exc}")
            return [], []

        interval_type = "HOURLY" if self.units == "hour" else "DAILY"
        start_date = "2015-01-01T00:00:00Z"
        end_date = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        requested = set(workspaces)
        for ws_entry in all_workspaces:
            ws = ws_entry.get("workspaceName")
            if ws not in requested:
                continue

            models = ws_entry.get("models", []) or []
            ws_counts: dict = defaultdict(float)
            ws_total = 0.0

            for model in models:
                model_name = model.get("modelName")
                model_id = model.get("modelId")
                try:
                    points = self._mpm_prediction_points(
                        mpm,
                        model_id,
                        start_date,
                        end_date,
                        interval_type,
                    )
                    created = self._mpm_model_created(mpm, model_id, points)
                    if created is not None:
                        events.append(
                            CreationEvent(
                                platform="mpm",
                                workspace=ws,
                                use_case=model_name,
                                kind="mpm_model",
                                created=created,
                            )
                        )

                    counts = self._mpm_prediction_counts(points)
                    total = sum(counts.values())
                    usage.append(
                        UsageMetric(
                            platform="mpm",
                            workspace=ws,
                            metric="PREDICTION_VOLUME",
                            value=total,
                            project=model_name,
                            series=(
                                continuous_series(counts, self.units) if counts else []
                            ),
                        )
                    )
                    ws_total += total
                    for key, n in counts.items():
                        ws_counts[key] += n
                except Exception as exc:
                    print(
                        f"Warning: failed to collect MPM model "
                        f"{ws}/{model_name}: {exc}"
                    )
                    continue

            usage.append(
                UsageMetric(
                    platform="mpm",
                    workspace=ws,
                    metric="PREDICTION_VOLUME",
                    value=ws_total,
                    project=None,
                    series=(
                        continuous_series(dict(ws_counts), self.units)
                        if ws_counts
                        else []
                    ),
                )
            )

        return events, usage

    def _mpm_prediction_points(self, mpm, model_id, start_date, end_date, interval):
        """Fetch the wide prediction series ONCE and return the raw
        `[{"x": ..., "y": ...}, ...]` points, parsed defensively.

        Reused by both `_mpm_model_created` (creation proxy) and
        `_mpm_prediction_counts` (usage metric) -- never call
        `get_nb_predictions` twice for the same model.
        """
        resp = mpm._client.get_nb_predictions(
            model_id, start_date, end_date, interval, [], None
        )
        if not isinstance(resp, dict):
            return []
        series = resp.get("data") or []
        if not series or not isinstance(series[0], dict):
            return []
        return series[0].get("data") or []

    def _mpm_point_time(self, x):
        """Parse a datapoint's `x` value (epoch-ms int/float or ISO str)."""
        if isinstance(x, (int, float)):
            return _ms_to_utc(x)
        return datetime.datetime.fromisoformat(str(x).replace("Z", "+00:00"))

    def _mpm_model_created(self, mpm, model_id, points):
        """Resolve a model's `created` via probe -> proxy -> None chain."""
        try:
            details = mpm._client.get_model_details(model_id) or {}
        except Exception:
            details = {}

        for key in (
            "createdAt",
            "creationDate",
            "creationTimestamp",
            "created_at",
            "createdAtMillis",
        ):
            val = details.get(key)
            if not val:
                continue
            if isinstance(val, (int, float)):
                return _ms_to_utc(val)
            try:
                return datetime.datetime.fromisoformat(str(val).replace("Z", "+00:00"))
            except ValueError:
                continue

        earliest = None
        for point in points:
            try:
                x, y = point.get("x"), point.get("y")
                if not y:
                    continue
                t = self._mpm_point_time(x)
            except Exception:
                continue
            if earliest is None or t < earliest:
                earliest = t
        return earliest

    def _mpm_prediction_counts(self, points):
        """Bucket the pre-fetched prediction points by `self.units`."""
        counts: dict = defaultdict(float)
        for point in points:
            try:
                x, y = point.get("x"), point.get("y")
                t = self._mpm_point_time(x)
            except Exception:
                continue
            counts[format_time_key(t, self.units)] += y or 0
        return dict(counts)

    def _em_experiment_counts(self, experiments):
        """Bucket the project's pre-fetched experiment start timestamps by
        `self.units` to build the all-time EXPERIMENT_COUNT over-time series."""
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
