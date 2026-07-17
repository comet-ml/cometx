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
import os
import re
from collections import defaultdict
from urllib.parse import urlparse

from tqdm import tqdm

from cometx.cli.admin_growth_render import build_html, write_html
from cometx.cli.admin_growth_users import (
    active_series,
    adoption_stats,
    bottom_users,
    capability_series,
    classify_accounts,
    parse_users,
    top_users,
)
from cometx.utils import fetch_chargeback_report, format_time_key, get_next_time_key

# `build_html` is re-exported here so the growth-report module is the single
# import surface (tests + callers import it from here). Declaring it in
# `__all__` documents the intentional re-export without a per-line noqa.
__all__ = ["generate_growth_report", "GrowthReporter", "build_html", "write_html"]


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


def _num(value):
    """Render a metric value as an int when it has no fractional part, else
    a float -- avoids "3.0" showing up in KPI/table cells for counts that
    are naturally integers but arrive as floats from the collectors."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _extract_service_account_names(payload) -> "set[str] | None":
    """Defensively unwrap the `/admin/service-accounts` response into a
    flat set of account names, tolerating several plausible response
    shapes since the exact schema isn't documented in this codebase: a
    bare list of entries, or a dict wrapping that list under a common
    container key (`serviceAccounts`, `accounts`, `users`). Each entry may
    be a plain string, or a dict carrying the name under `name`,
    `username`, or `email` (mirrors `_extract_licensed_users`'s defensive
    style in `admin_growth_users.py`).

    Returns `None` (not an empty set) when the payload cannot be honestly
    parsed as a service-account container -- either because its shape
    isn't recognized at all, or because a recognized, non-empty container
    yielded zero extractable names. An empty set is returned only for a
    recognized container that is genuinely empty (e.g. `[]` or
    `{"serviceAccounts": []}`), which is a real "zero service accounts"
    answer, not a parse failure. Distinguishing these two lets callers
    (`_fetch_service_accounts`) fall back to the labeled regex heuristic
    on a parse failure instead of silently reporting zero via the
    (misleading) admin_api source."""
    container = payload
    recognized = isinstance(payload, list)
    if isinstance(payload, dict):
        for key in ("serviceAccounts", "accounts", "users"):
            if key in payload:
                container = payload[key]
                recognized = isinstance(container, list)
                break

    if not recognized:
        return None
    if not container:
        return set()

    names = set()
    for entry in container:
        if isinstance(entry, str):
            names.add(entry)
        elif isinstance(entry, dict):
            name = entry.get("name") or entry.get("username") or entry.get("email")
            if name:
                names.add(name)

    return names or None


def _fetch_service_accounts(api) -> "set[str] | None":
    """Fetch the authoritative set of service-account names from the admin
    `/admin/service-accounts` endpoint (same request shape as
    `fetch_chargeback_report` in `cometx.utils`), for `classify_accounts`'s
    admin_api path. Returns `None` on ANY failure -- endpoint disabled,
    403/404, network error, unexpected response shape, or a successful
    response that could not be honestly parsed into any names (see
    `_extract_service_account_names`) -- so callers degrade to the
    labeled regex heuristic instead of crashing or silently reporting a
    zero split under the admin_api label."""
    try:
        parsed = urlparse(api.config["comet.url_override"])
        base = "%s://%s" % (parsed.scheme, parsed.netloc)
        while base.endswith("/"):
            base = base[:-1]
        url = base + "/api/admin/service-accounts"
        response = api._client.get(
            url, headers={"Authorization": api.api_key}, params={}
        )
        payload = response.json()
        return _extract_service_account_names(payload)
    except Exception:
        return None


def _as_float(value):
    """Coerce a collector datapoint value to a float, treating None/blank and
    any non-numeric value (e.g. a stray string from an SDK/REST response) as
    0.0 -- keeps count aggregation from crashing on mixed-type payloads."""
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _all_time_counts(events, units) -> dict:
    """Bucket `events` by creation time-key, with NO window filtering --
    charts always render all-time (Option-A: the window is a shaded band
    drawn on top, not a data filter)."""
    counts: dict = defaultdict(int)
    for ev in events:
        counts[format_time_key(ev.created, units)] += 1
    return dict(counts)


def _stacked_points(events_by_kind, units, kinds) -> list:
    """Build zero-filled `{"key": k, "values": {kind: count, ...}}` points
    for a stackedBars chart, across the union of all-time bucket keys seen
    for ANY kind (so every series shares one continuous, zero-filled key
    sequence even if a given kind has no events in some periods)."""
    per_kind_counts = {
        kind: _all_time_counts(events, units) for kind, events in events_by_kind.items()
    }
    merged: dict = defaultdict(int)
    for counts in per_kind_counts.values():
        for key in counts:
            merged[key] += 0  # ensure key presence without double counting
    if not merged:
        return []
    keys = [k for k, _ in continuous_series(dict(merged), units)]
    return [
        {
            "key": k,
            "values": {kind: per_kind_counts.get(kind, {}).get(k, 0) for kind in kinds},
        }
        for k in keys
    ]


_WINDOW_SPEC_RE = re.compile(r"^(\d+)([dwmy])$")
_WINDOW_DAYS_PER_UNIT = {"d": 1, "w": 7, "m": 30, "y": 365}


def parse_window(spec, now, units="month") -> Window:
    """Parse a relative `--window` spec (e.g. "7d"/"14d"/"30d"/"90d") into a
    `Window(start=now - delta, end=now, units=units)`.

    Format: ``\\d+[dwmy]`` -- ``d``=days, ``w``=weeks (x7 days), ``m``=months
    (approximated as 30 days), ``y``=years (approximated as 365 days). The
    m/y approximations are deliberate: the report only needs a window
    boundary for "installed base before window" comparisons, not calendar-
    exact month/year arithmetic. Defaults to "7d" when `spec` is falsy.
    Raises `ValueError` on a malformed spec.
    """
    spec = "7d" if spec is None else spec.strip()
    match = _WINDOW_SPEC_RE.match(spec)
    if not match:
        raise ValueError(
            f"Invalid --window spec {spec!r}; expected e.g. '7d', '14d', "
            "'30d', '90d' (\\d+[dwmy])"
        )
    amount, unit = int(match.group(1)), match.group(2)
    delta = datetime.timedelta(days=amount * _WINDOW_DAYS_PER_UNIT[unit])
    return Window(start=now - delta, end=now, units=units)


USE_CASE_KINDS = ("opik_project", "em_project", "mpm_model")


def unified_events(events) -> list:
    """All three use-case kinds combined for the unified created-over-time
    series.

    The collectors already emit only the three use-case kinds
    (`opik_project`/`em_project`/`mpm_model`), so this is effectively a
    defensive filter -- any other kind (e.g. a synthetic `workspace` event)
    is excluded. Returns a new list; never mutates `events`.
    """
    return [e for e in events if e.kind in USE_CASE_KINDS]


def use_cases_by_workspace(events) -> dict:
    """Per-workspace use-case roll-up.

    Returns ``{workspace: {"use_cases_total": int, "by_kind": {...},
    "first_created": datetime}}``. `by_kind` always contains all three
    use-case kinds (0 if absent). `first_created` is the workspace-creation
    proxy: the earliest use-case `created` timestamp seen in that workspace.
    Only the three use-case kinds are considered (via `unified_events`).
    """
    result: dict = {}
    for ev in unified_events(events):
        entry = result.setdefault(
            ev.workspace,
            {
                "use_cases_total": 0,
                "by_kind": {kind: 0 for kind in USE_CASE_KINDS},
                "first_created": ev.created,
            },
        )
        entry["use_cases_total"] += 1
        entry["by_kind"][ev.kind] += 1
        if ev.created < entry["first_created"]:
            entry["first_created"] = ev.created
    return result


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
    active_window="60d",
    include_users=True,
    leaderboard_top_n=5,
    exclude_personal=False,
    personal_pattern=None,
):
    reporter = GrowthReporter(
        api,
        window=window,
        units=units,
        platforms=platforms,
        limit=limit,
        active_window=active_window,
        include_users=include_users,
        leaderboard_top_n=leaderboard_top_n,
        exclude_personal=exclude_personal,
        personal_pattern=personal_pattern,
    )
    report_data = reporter.build(workspaces)  # events + usage -> report_data (C2-C7)
    path = write_growth_html(report_data, output)  # C8
    if not no_open:
        _open(path)
    return path


class GrowthReporter:
    def __init__(
        self,
        api,
        *,
        window,
        units,
        platforms,
        limit=None,
        active_window="60d",
        include_users=True,
        leaderboard_top_n=5,
        exclude_personal=False,
        personal_pattern=None,
    ):
        self.api = api
        self.window = window
        self.units = units
        self.platforms = platforms
        self.limit = limit
        self.active_window = active_window
        self.include_users = include_users
        self.leaderboard_top_n = leaderboard_top_n
        self.exclude_personal = exclude_personal
        self.personal_pattern = personal_pattern
        self._warned_no_personal_pattern = False

    def build(self, workspaces):
        """Resolve window/platforms/workspaces, run the selected collectors,
        and assemble `report_data` matching the C8 renderer contract (see
        `.superpowers/sdd/task-C8-report.md`).
        """
        now = self._now()
        window = parse_window(self.window, now, self.units)

        platforms = self._resolve_platforms()
        print("Resolving workspaces...")
        resolved_workspaces = self._resolve_workspaces(workspaces)
        print(
            f"Collecting {', '.join(platforms) or 'no'} data for "
            f"{len(resolved_workspaces)} workspace(s)..."
        )
        events, usage, ran = self._collect_selected(platforms, resolved_workspaces)

        chargeback = None
        if self.include_users and platforms:
            try:
                print("Collecting users/people data (chargeback report)...")
                chargeback = fetch_chargeback_report(self.api)
            except Exception as exc:
                print(f"Warning: failed to fetch chargeback report; skipping people layer: {exc}")
                chargeback = None

        print("Building report...")

        now_ms = int(now.timestamp() * 1000)
        return self._assemble_report_data(
            events, usage, ran, resolved_workspaces, window, chargeback, now_ms
        )

    def _now(self):
        return datetime.datetime.now(datetime.timezone.utc)

    def _resolve_platforms(self):
        """`self.platforms` (csv) intersected with {em,opik,mpm}, in a fixed
        order, dropping opik/mpm if their optional dependency isn't
        importable (EM has no optional dependency)."""
        requested = {p.strip() for p in (self.platforms or "").split(",") if p.strip()}
        resolved = []
        for platform in ("em", "opik", "mpm"):
            if platform not in requested:
                continue
            if platform == "opik":
                try:
                    import opik  # noqa: F401
                except ImportError:
                    print("Note: opik not installed; skipping Opik collection")
                    continue
            elif platform == "mpm":
                try:
                    import comet_mpm  # noqa: F401
                except ImportError:
                    print("Note: comet_mpm not installed; skipping MPM collection")
                    continue
            resolved.append(platform)
        return resolved

    def _resolve_workspaces(self, workspaces):
        """Use the given `WORKSPACE` args if non-empty, else
        `api.get_workspaces()`; apply `self.limit` once here so the final
        workspace list is fixed before any collector runs (the collectors'
        own `self.limit` slicing then becomes a no-op on this already-sliced
        list -- not a double-slice)."""
        resolved = (
            list(workspaces) if workspaces else list(self.api.get_workspaces() or [])
        )
        resolved = self._filter_personal(resolved)
        if self.limit is not None:
            resolved = resolved[: self.limit]
        return resolved

    def _filter_personal(self, workspaces):
        """Drop workspaces matching `self.personal_pattern` when
        `self.exclude_personal` is set (Task 9). A no-op unless BOTH the
        flag is on AND a pattern was supplied -- an on flag with no pattern
        can't identify anything to drop, so it's a no-op too, with a
        once-per-reporter warning since that combination is almost
        certainly a user mistake (they meant to also pass
        `--personal-pattern`). An invalid regex degrades the same way
        (warn once, return unchanged) rather than crashing the whole run.
        """
        if not self.exclude_personal:
            return list(workspaces)
        if not self.personal_pattern:
            if not self._warned_no_personal_pattern:
                print(
                    "Warning: --exclude-personal has no effect without "
                    "--personal-pattern; skipping personal-workspace exclusion"
                )
                self._warned_no_personal_pattern = True
            return list(workspaces)
        try:
            pattern = re.compile(self.personal_pattern)
        except re.error as exc:
            if not self._warned_no_personal_pattern:
                print(
                    f"Warning: invalid --personal-pattern "
                    f"{self.personal_pattern!r} ({exc}); skipping "
                    "personal-workspace exclusion"
                )
                self._warned_no_personal_pattern = True
            return list(workspaces)
        return [ws for ws in workspaces if not pattern.search(ws)]

    _COLLECTOR_METHODS = {
        "em": "_collect_em",
        "opik": "_collect_opik",
        "mpm": "_collect_mpm",
    }

    def _collect_selected(self, platforms, workspaces):
        """Run each selected platform's collector and aggregate all events +
        usage. Returns `(events, usage, ran)` where `ran` is a
        `{"opik": bool, "em": bool, "mpm": bool}` map of which collectors
        actually executed."""
        events: list = []
        usage: list = []
        ran = {"opik": False, "em": False, "mpm": False}
        for platform in platforms:
            print(f"[{PLATFORM_LABELS.get(platform, platform)}] collecting...")
            collector = getattr(self, self._COLLECTOR_METHODS[platform])
            platform_events, platform_usage = collector(workspaces)
            events.extend(platform_events)
            usage.extend(platform_usage)
            ran[platform] = True
        return events, usage, ran

    def _window_label(self, window):
        return (
            f"Analysis window: {window.start.date().isoformat()} "
            f"– {window.end.date().isoformat()} ({self.window or '7d'})"
        )

    def _build_window_block(self, window, count_before):
        return {
            "start": window.start.isoformat(),
            "end": window.end.isoformat(),
            "units": window.units,
            "label": self._window_label(window),
            "count_before": count_before,
            "window_start": format_time_key(window.start, window.units),
            "window_end": format_time_key(window.end, window.units),
        }

    def _series_chart_data(self, events, window):
        """All-time zero-filled `(points, window_start, window_end)` for a
        single-series bar/area chart, per Option-A window rendering."""
        counts = _all_time_counts(events, self.units)
        points = [
            {"key": k, "value": v} for k, v in continuous_series(counts, self.units)
        ]
        window_start = format_time_key(window.start, self.units)
        window_end = format_time_key(window.end, self.units)
        return points, window_start, window_end

    def _growth_kpis(self, stats, count_before, workspaces_count):
        spec = self.window or "7d"
        return [
            {"label": "Workspaces", "value": workspaces_count},
            {"label": "Total", "value": stats["total"]},
            {"label": f"New ({spec})", "value": f"+{stats['new_in_window']}"},
            {
                "label": f"Growth ({spec})",
                "value": f"{stats['pct_growth']}%",
                "sub": f"vs {count_before} before window",
            },
        ]

    def _breakdown_table(self, events, kind_label, workspaces_count):
        """Apply the workspace-vs-use-case breakdown rule: more than one
        workspace -> break down by workspace; a single workspace -> break
        down by use case instead (a by-workspace table would have exactly
        one, uninformative, row)."""
        if workspaces_count > 1:
            by_ws: dict = defaultdict(int)
            for ev in events:
                by_ws[ev.workspace] += 1
            rows = sorted(by_ws.items(), key=lambda kv: -kv[1])
            return {
                "title": f"{kind_label} by workspace",
                "headers": ["Workspace", kind_label],
                "rows": [[ws, count] for ws, count in rows],
            }
        rows = sorted(events, key=lambda ev: ev.created, reverse=True)
        return {
            "title": kind_label,
            "headers": ["Use case", "Created"],
            "rows": [[ev.use_case, ev.created.date().isoformat()] for ev in rows],
        }

    def _unified_table(self, events, workspaces_count):
        if workspaces_count > 1:
            by_ws = use_cases_by_workspace(events)
            rows = sorted(by_ws.items(), key=lambda kv: -kv[1]["use_cases_total"])
            return {
                "title": "By workspace",
                "headers": ["Workspace", "Opik", "EM", "MPM", "Total"],
                "rows": [
                    [
                        ws,
                        entry["by_kind"]["opik_project"],
                        entry["by_kind"]["em_project"],
                        entry["by_kind"]["mpm_model"],
                        entry["use_cases_total"],
                    ]
                    for ws, entry in rows
                ],
            }
        rows = sorted(unified_events(events), key=lambda ev: ev.created, reverse=True)
        return {
            "title": "Use cases",
            "headers": ["Use case", "Kind", "Created"],
            "rows": [
                [
                    ev.use_case,
                    KIND_LABELS.get(ev.kind, ev.kind),
                    ev.created.date().isoformat(),
                ]
                for ev in rows
            ],
        }

    def _unified_stacked_chart(self, events, window):
        events_by_kind = {
            kind: [ev for ev in events if ev.kind == kind] for kind in USE_CASE_KINDS
        }
        points = _stacked_points(events_by_kind, self.units, USE_CASE_KINDS)
        window_start = format_time_key(window.start, self.units) if points else None
        window_end = format_time_key(window.end, self.units) if points else None
        return {
            "id": "chart-unified-created",
            "kind": "stackedBars",
            "title": "Use cases created",
            "hint": f"by kind · {self.units}ly",
            "legend": [
                {"label": "Opik", "color": "--accent"},
                {"label": "EM", "color": "--sdk"},
                {"label": "MPM", "color": "--ok"},
            ],
            "data": {
                "categories": list(USE_CASE_KINDS),
                "labels": {k: KIND_LABELS[k] for k in USE_CASE_KINDS},
                "colors": ["--accent", "--sdk", "--ok"],
                "points": points,
                "window_start": window_start,
                "window_end": window_end,
            },
        }

    def _unified_department_chart(self, unified):
        by_ws = use_cases_by_workspace(unified)
        rows = sorted(
            (
                {"label": ws, "value": entry["use_cases_total"]}
                for ws, entry in by_ws.items()
            ),
            key=lambda r: -r["value"],
        )
        return {
            "id": "chart-unified-by-workspace",
            "kind": "groupedBarsH",
            "title": "Use cases by workspace",
            "hint": "workspaces proxy teams / departments · current totals",
            "data": {"rows": rows},
        }

    def _build_unified_section(self, events, window, workspaces_count):
        unified = unified_events(events)
        count_before = sum(1 for ev in unified if ev.created < window.start)
        stats = growth_stats(unified, window, self.units)
        spec = self.window or "7d"
        return {
            "title": "Use cases across all platforms",
            "window_chip": self._window_label(window),
            "kpis": [
                {
                    "label": "Workspaces",
                    "value": workspaces_count,
                    "sub": "proxy for teams / departments",
                },
                {"label": "Use cases", "value": stats["total"], "tone": "ok"},
                {"label": f"New ({spec})", "value": f"+{stats['new_in_window']}"},
                {
                    "label": f"Growth ({spec})",
                    "value": f"{stats['pct_growth']}%",
                    "sub": f"vs {count_before} before window",
                },
            ],
            "charts": [
                self._unified_stacked_chart(events, window),
                self._unified_department_chart(unified),
            ],
            "table": self._unified_table(events, workspaces_count),
        }

    def _adoption_metric_series(self, metric_name, metrics, window):
        """Combine the per-workspace-total entries (`project is None`) for
        `metric_name` into one all-time zero-filled series + grand total."""
        ws_totals = [
            m for m in metrics if m.metric == metric_name and m.project is None
        ]
        total = sum(m.value for m in ws_totals)
        merged: dict = defaultdict(float)
        for m in ws_totals:
            for key, value in m.series or []:
                merged[key] += value
        points = (
            [
                {"key": k, "value": v}
                for k, v in continuous_series(dict(merged), self.units)
            ]
            if merged
            else []
        )
        window_start = format_time_key(window.start, self.units)
        window_end = format_time_key(window.end, self.units)
        return _num(total), points, window_start, window_end

    def _adoption_table_by_project(self, metric_name, metrics, value_label):
        project_metrics = [
            m for m in metrics if m.metric == metric_name and m.project is not None
        ]
        rows = sorted(project_metrics, key=lambda m: -m.value)
        return {
            "title": f"{value_label} by project",
            "headers": ["Project", value_label],
            "rows": [[m.project, _num(m.value)] for m in rows],
        }

    def _registry_panel(self, usage):
        models = {m.workspace: m.value for m in usage if m.metric == "REGISTRY_MODELS"}
        versions = {
            m.workspace: m.value for m in usage if m.metric == "REGISTRY_VERSIONS"
        }
        workspaces = sorted(set(models) | set(versions))
        return {
            "title": "Model-registry engagement",
            "hint": "snapshot, not over-time",
            "headers": ["Workspace", "Registered models", "Model versions"],
            "rows": [
                [ws, _num(models.get(ws, 0)), _num(versions.get(ws, 0))]
                for ws in workspaces
            ],
        }

    def _series_window_stats(self, points, window_start, window_end):
        """Growth of a per-period value series over the window. Bucket keys
        are zero-padded and lexicographically sortable, so window membership
        is a string comparison. Returns total / new_in_window / before /
        pct_growth (0-guarded), mirroring `growth_stats` for value series."""
        total = new_in = before = 0.0
        for p in points:
            key, value = p.get("key"), p.get("value", 0) or 0
            total += value
            if window_start <= key <= window_end:
                new_in += value
            elif key < window_start:
                before += value
        pct = (new_in / before * 100.0) if before > 0 else 0.0
        return {
            "total": total,
            "new_in_window": new_in,
            "before": before,
            "pct_growth": round(pct, 1),
        }

    def _fastest_growing_project(self, metric_name, metrics, window_start, window_end):
        """The project with the largest in-window increase for `metric_name`
        (absolute new-in-window, which is robust to tiny-base % noise).
        Returns (project, new_in_window) or None if nothing grew in-window."""
        best = None
        for m in metrics:
            if m.metric != metric_name or m.project is None:
                continue
            new_in = sum(
                value
                for key, value in (m.series or [])
                if window_start <= key <= window_end
            )
            if new_in > 0 and (best is None or new_in > best[1]):
                best = (m.project, new_in)
        return best

    def _build_adoption_section(self, platform, usage, window):
        """Per-product adoption/usage section. Registry counts (EM) are
        NEVER merged with MPM's monitored-model metrics -- they live in
        their own `panels` entry with distinct labels."""
        if platform == "opik":
            metric, value_label = "SPAN_COUNT", "Span count"
        elif platform == "em":
            metric, value_label = "EXPERIMENT_COUNT", "Experiment count"
        else:
            metric, value_label = "PREDICTION_VOLUME", "Prediction volume"

        total, points, window_start, window_end = self._adoption_metric_series(
            metric, usage, window
        )
        stats = self._series_window_stats(points, window_start, window_end)
        spec = self.window or "7d"

        # Usage growth KPIs (not just the total), plus the fastest-growing
        # project so a bare "total" section reads as a trend.
        kpis = [
            {"label": value_label, "value": _num(total)},
            {"label": f"New ({spec})", "value": f"+{_num(stats['new_in_window'])}"},
            {
                "label": f"Growth ({spec})",
                "value": f"{stats['pct_growth']}%",
                "sub": f"vs {_num(stats['before'])} before window",
            },
        ]
        # Always surface the fastest-growing project (for every product,
        # incl. EM = most new experiments in the window); "-" when nothing
        # grew in-window so the KPI is consistently present, not missing.
        fastest = self._fastest_growing_project(metric, usage, window_start, window_end)
        if fastest:
            kpis.append(
                {
                    "label": "Fastest-growing project",
                    "value": fastest[0],
                    "sub": f"+{_num(fastest[1])} in window",
                }
            )
        else:
            kpis.append(
                {
                    "label": "Fastest-growing project",
                    "value": "—",
                    "sub": "no activity in window",
                }
            )

        charts = []
        if points:
            charts.append(
                {
                    "id": f"chart-{platform}-adoption",
                    "kind": "bars",
                    "title": f"{value_label} — new per period",
                    "hint": f"by {self.units}",
                    "data": {
                        "points": points,
                        "window_start": window_start,
                        "window_end": window_end,
                    },
                }
            )
            cum_points = [
                {"key": k, "value": v}
                for k, v in cumulative([(p["key"], p["value"]) for p in points])
            ]
            charts.append(
                {
                    "id": f"chart-{platform}-adoption-cumulative",
                    "kind": "area",
                    "title": f"{value_label} — cumulative",
                    "hint": "all-time",
                    "data": {
                        "points": cum_points,
                        "window_start": window_start,
                        "window_end": window_end,
                        "delta": _num(stats["new_in_window"]),
                    },
                }
            )

        section = {
            "title": f"{PLATFORM_LABELS[platform]} — adoption / usage",
            "kpis": kpis,
            "charts": charts,
            "table": self._adoption_table_by_project(metric, usage, value_label),
        }
        if platform == "em":
            registry_panel = self._registry_panel(usage)
            section["panels"] = [registry_panel] if registry_panel["rows"] else []
        return section

    def _build_product_section(
        self, platform, kind, events, usage, window, workspaces_count
    ):
        kind_events = [ev for ev in events if ev.kind == kind]
        stats = growth_stats(kind_events, window, self.units)
        count_before = sum(1 for ev in kind_events if ev.created < window.start)
        workspaces_with_kind = {ev.workspace for ev in kind_events}

        points, window_start, window_end = self._series_chart_data(kind_events, window)
        cum_points = [
            {"key": k, "value": v}
            for k, v in cumulative(
                continuous_series(_all_time_counts(kind_events, self.units), self.units)
            )
        ]

        growth_section = {
            "title": f"{PLATFORM_LABELS[platform]} — growth",
            "window_chip": self._window_label(window),
            "kpis": self._growth_kpis(stats, count_before, len(workspaces_with_kind)),
            "charts": [
                {
                    "id": f"chart-{platform}-bars",
                    "kind": "bars",
                    "title": f"New {KIND_LABELS[kind]}",
                    "hint": f"by {self.units}",
                    "data": {
                        "points": points,
                        "window_start": window_start,
                        "window_end": window_end,
                    },
                },
                {
                    "id": f"chart-{platform}-area",
                    "kind": "area",
                    "title": f"{KIND_LABELS[kind]} — cumulative",
                    "hint": "all-time",
                    "data": {
                        "points": cum_points,
                        "window_start": window_start,
                        "window_end": window_end,
                        "delta": stats["new_in_window"],
                    },
                },
            ],
            "table": self._breakdown_table(
                kind_events, KIND_LABELS[kind], workspaces_count
            ),
        }

        return {
            "label": PLATFORM_LABELS[platform],
            "growth": growth_section,
            "adoption": self._build_adoption_section(platform, usage, window),
        }

    def _active_window_days(self, now):
        """Resolve `self.active_window` (e.g. "60d") into an integer day
        count, via the shared `parse_window` parser (same spec grammar as
        `--window`)."""
        active_window = parse_window(self.active_window, now, self.units)
        return max(1, (active_window.end - active_window.start).days)

    def _build_people_section(self, chargeback, now_ms):
        """Users/adoption section, derived from the chargeback report (Task
        4 helpers). KPIs (Total / Active / Adoption %) plus real over-time
        `lines` charts: active-vs-total (Task 6's `active_series`), and,
        when at least one user has an EM/Opik capability timestamp, a
        per-capability active-users chart (`capability_series`)."""
        now = _ms_to_utc(now_ms)
        active_window_days = self._active_window_days(now)

        users = parse_users(chargeback)
        stats = adoption_stats(users, now_ms, active_window_days)

        kpis = [
            {"label": "Total", "value": stats["total"]},
            {
                "label": f"Active ({self.active_window})",
                "value": stats["active"],
            },
            {"label": "Adoption %", "value": f"{stats['adoption_pct']}%"},
        ]

        active_pts = active_series(users, self.units, now_ms, active_window_days)
        if active_pts:
            window_start = active_pts[0]["key"]
            window_end = active_pts[-1]["key"]
            points = active_pts
        else:
            # No user has a known created_at to bucket from -- degrade to a
            # single current-period point rather than an empty chart.
            bucket_key = format_time_key(now, self.units)
            window_start = window_end = bucket_key
            points = [
                {
                    "key": bucket_key,
                    "values": {"total": stats["total"], "active": stats["active"]},
                }
            ]

        charts = [
            {
                "id": "chart-people-active-total",
                "kind": "lines",
                "title": "Active vs. total users",
                "hint": f"active window {self.active_window} · {self.units}ly",
                "legend": [
                    {"label": "Total", "color": "--sdk"},
                    {"label": "Active", "color": "--ok"},
                ],
                "data": {
                    "categories": ["total", "active"],
                    "labels": {"total": "Total", "active": "Active"},
                    "colors": ["--sdk", "--ok"],
                    "points": points,
                    "window_start": window_start,
                    "window_end": window_end,
                },
            }
        ]

        cap_pts = capability_series(users, self.units, now_ms, active_window_days)
        if cap_pts:
            charts.append(
                {
                    "id": "chart-people-capability",
                    "kind": "lines",
                    "title": "Active users by capability",
                    "hint": f"active window {self.active_window} · {self.units}ly",
                    "legend": [
                        {"label": "EM", "color": "--sdk"},
                        {"label": "Opik", "color": "--accent"},
                    ],
                    "data": {
                        "categories": ["em", "opik"],
                        "labels": {"em": "EM", "opik": "Opik"},
                        "colors": ["--sdk", "--accent"],
                        "points": cap_pts,
                        "window_start": cap_pts[0]["key"],
                        "window_end": cap_pts[-1]["key"],
                    },
                }
            )

        top = top_users(users, "em_score", self.leaderboard_top_n)
        table = {
            "title": f"Top {self.leaderboard_top_n} users by activity",
            "headers": ["User", "Experiments", "Data logged (MB)", "Opik spans"],
            "rows": [
                [
                    u.username,
                    _num(u.experiment_count),
                    _num(u.data_logged_mb),
                    _num(u.opik_span_count) if u.opik_span_count is not None else "-",
                ]
                for u in top
            ],
        }

        return {
            "title": "Users & adoption",
            "kpis": kpis,
            "charts": charts,
            "table": table,
        }

    # Workspace-level leaderboard metrics per platform: (metric name, plain
    # label used in chart titles, e.g. "Top 5 workspaces by {label}").
    _LEADERBOARD_METRICS = {
        "em": [("EXPERIMENT_COUNT", "experiments")],
        "opik": [("SPAN_COUNT", "spans"), ("TRACE_COUNT", "traces")],
        "mpm": [("PREDICTION_VOLUME", "predictions")],
    }

    # User-level leaderboard metrics: (top_users/bottom_users key, plain
    # label, value-extractor matching admin_growth_users._metric_value).
    _USER_LEADERBOARD_METRICS = [
        ("opik_span_count", "Opik spans", lambda u: u.opik_span_count),
        ("em_score", "activity", lambda u: u.experiment_count + u.data_logged_mb),
    ]

    @staticmethod
    def _leaderboard_chart(chart_id, title, rows):
        return {
            "id": chart_id,
            "kind": "groupedBarsH",
            "title": title,
            "data": {"rows": rows},
        }

    def _build_leaderboards_section(self, usage, users, ran):
        """Top-N / bottom-N leaderboards: one per workspace-level metric of
        each platform that actually ran (Task 5's `ran` map), plus user
        leaderboards (by `opik_span_count`, by `em_score`) when `users` is
        non-empty. Bottom-N is active-aware (strictly-positive values only,
        ascending -- mirrors `bottom_users`). Platforms/metrics with no data
        are omitted entirely (no empty panels); returns `None` when there is
        nothing to show at all."""
        n = self.leaderboard_top_n
        charts = []

        for platform, metrics in self._LEADERBOARD_METRICS.items():
            if not ran.get(platform):
                continue
            for metric, label in metrics:
                ws_metrics = [
                    m
                    for m in usage
                    if m.platform == platform
                    and m.metric == metric
                    and m.project is None
                ]
                if not ws_metrics:
                    continue

                ranked_desc = sorted(ws_metrics, key=lambda m: m.value, reverse=True)
                top_rows = [
                    {"label": m.workspace, "value": m.value}
                    for m in ranked_desc[:n]
                ]
                active_asc = sorted(
                    (m for m in ws_metrics if m.value > 0), key=lambda m: m.value
                )
                bottom_rows = [
                    {"label": m.workspace, "value": m.value}
                    for m in active_asc[:n]
                ]

                slug = f"{platform}-{metric.lower()}"
                if top_rows:
                    charts.append(
                        self._leaderboard_chart(
                            f"chart-lb-{slug}-top",
                            f"Top {n} workspaces by {label}",
                            top_rows,
                        )
                    )
                if bottom_rows:
                    charts.append(
                        self._leaderboard_chart(
                            f"chart-lb-{slug}-bottom",
                            f"Bottom {n} workspaces by {label}",
                            bottom_rows,
                        )
                    )

        if users:
            for key, label, value_fn in self._USER_LEADERBOARD_METRICS:
                top = top_users(users, key, n)
                if top:
                    charts.append(
                        self._leaderboard_chart(
                            f"chart-lb-user-{key}-top",
                            f"Top {n} users by {label}",
                            [{"label": u.username, "value": value_fn(u)} for u in top],
                        )
                    )
                bottom = bottom_users(users, key, n)
                if bottom:
                    charts.append(
                        self._leaderboard_chart(
                            f"chart-lb-user-{key}-bottom",
                            f"Bottom {n} users by {label}",
                            [
                                {"label": u.username, "value": value_fn(u)}
                                for u in bottom
                            ],
                        )
                    )

        if not charts:
            return None

        return {"title": "Leaderboards", "charts": charts}

    # Personal-vs-service-account metrics, one groupedBarsH chart each:
    # (classify_accounts totals key, plain label for the chart title).
    _PERSONAL_VS_SERVICE_METRICS = [
        ("experiments", "experiments"),
        ("data", "data logged (MB)"),
        ("spans", "Opik spans"),
    ]

    def _build_personal_vs_service_section(self, users, service_account_names):
        """Personal-vs-service-account split (Task 8): one `groupedBarsH`
        chart per metric (experiments / data / spans), each showing
        Personal vs. Service totals via `classify_accounts`. A metric's
        chart is omitted when both buckets are zero for it (nothing to
        show). Returns `None` when `users` is empty or every metric is
        all-zero (degrade, never render an empty section)."""
        if not users:
            return None

        split = classify_accounts(users, service_account_names)
        personal = split["personal"]
        service = split["service"]
        source = split["source"]

        hint = (
            "Source: service accounts from admin API."
            if source == "admin_api"
            else "Source: heuristic (regex); admin API service-account data unavailable."
        )

        charts = []
        for key, label in self._PERSONAL_VS_SERVICE_METRICS:
            personal_value = personal.get(key, 0)
            service_value = service.get(key, 0)
            if not personal_value and not service_value:
                continue
            charts.append(
                {
                    "id": f"chart-personal-vs-service-{key}",
                    "kind": "groupedBarsH",
                    "title": f"Personal vs. service accounts: {label}",
                    "hint": hint,
                    "data": {
                        "rows": [
                            {"label": "Personal", "value": _num(personal_value)},
                            {"label": "Service", "value": _num(service_value)},
                        ]
                    },
                }
            )

        if not charts:
            return None

        return {
            "title": "Personal vs. service accounts",
            "charts": charts,
            "hint": hint,
        }

    def _assemble_report_data(
        self, events, usage, ran, workspaces, window, chargeback=None, now_ms=None
    ):
        workspaces_count = len(workspaces)
        unified_section = self._build_unified_section(events, window, workspaces_count)
        count_before_unified = sum(
            1 for ev in unified_events(events) if ev.created < window.start
        )

        kind_by_platform = {
            "opik": "opik_project",
            "em": "em_project",
            "mpm": "mpm_model",
        }
        products = {}
        for platform in ("opik", "em", "mpm"):
            if not ran.get(platform):
                continue
            products[platform] = self._build_product_section(
                platform,
                kind_by_platform[platform],
                events,
                usage,
                window,
                workspaces_count,
            )

        sections = {"unified": unified_section, "products": products}
        if chargeback:
            try:
                people_section = self._build_people_section(chargeback, now_ms)
            except Exception as exc:
                print(
                    f"Warning: failed to build people section from chargeback data; "
                    f"skipping people layer: {exc}"
                )
                people_section = None
            if people_section:
                sections["people"] = people_section

        try:
            leaderboard_users = parse_users(chargeback) if chargeback else []
            leaderboards_section = self._build_leaderboards_section(
                usage, leaderboard_users, ran
            )
        except Exception as exc:
            print(
                f"Warning: failed to build leaderboards section; skipping: {exc}"
            )
            leaderboards_section = None
        if leaderboards_section:
            sections["leaderboards"] = leaderboards_section

        try:
            personal_vs_service_section = None
            if leaderboard_users:
                service_account_names = _fetch_service_accounts(self.api)
                personal_vs_service_section = self._build_personal_vs_service_section(
                    leaderboard_users, service_account_names
                )
        except Exception as exc:
            print(
                f"Warning: failed to build personal-vs-service section; skipping: {exc}"
            )
            personal_vs_service_section = None
        if personal_vs_service_section:
            sections["personal_vs_service"] = personal_vs_service_section

        return {
            "meta": {
                "title": "Comet growth report",
                "generated": window.end.isoformat(),
                "source": "Comet Admin API",
            },
            "window": self._build_window_block(window, count_before_unified),
            "collectors": ran,
            "sections": sections,
        }

    def _workspace_usage_metric(self, platform, metric, workspace, value, counts):
        """Build the workspace-level (`project=None`) summary `UsageMetric`
        appended by every collector, so the summary shape lives in one place
        instead of being duplicated across EM/Opik/MPM."""
        return UsageMetric(
            platform=platform,
            workspace=workspace,
            metric=metric,
            value=value,
            project=None,
            series=(continuous_series(dict(counts), self.units) if counts else []),
        )

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

        for ws in tqdm(list(workspaces), desc="EM workspaces", unit="ws"):
            try:
                response = self.api._client.get_from_endpoint(
                    "projects", {"workspaceName": ws}
                )
                projects = (response or {}).get("projects", []) or []
            except Exception as exc:
                print(f"Warning: failed to list EM projects for workspace {ws}: {exc}")
                continue

            # The EM `projects` endpoint returns a FALLBACK set (projects from
            # another workspace) when the API key isn't a member of `ws`,
            # ignoring the requested workspaceName. Keep only projects whose
            # own `workspaceName` actually matches `ws` so those fallback
            # projects aren't mis-attributed here. The workspace is still
            # reported (with a 0 experiment total below) rather than dropped.
            matched = [p for p in projects if p.get("workspaceName") == ws]
            dropped = len(projects) - len(matched)
            if dropped:
                print(
                    f"Warning: EM projects endpoint returned {dropped} project(s) "
                    f"not belonging to workspace {ws} (API key likely lacks EM "
                    f"access there); skipping them and reporting {ws} at 0."
                )
            projects = matched

            ws_experiment_total = 0
            ws_counts: dict = defaultdict(int)

            for project in tqdm(projects, desc=f"EM {ws}", unit="proj", leave=False):
                proj_name = project.get("projectName")
                try:
                    # Fetch the project's experiments once and reuse the list
                    # for both the creation proxy and the over-time series.
                    experiments = self.api.get_experiments(ws, proj_name) or []
                    created = self._em_project_created(project, experiments)
                    counts = self._em_experiment_counts(experiments)
                    # Derive the KPI total from the SAME bucketed counts that
                    # feed the chart series -- NOT the `numberOfExperiments`
                    # metadata, which can disagree with the experiments that
                    # actually carry a `start_server_timestamp`. This keeps the
                    # EM adoption KPI total (and its workspace roll-up) exactly
                    # equal to the cumulative chart sum.
                    total = sum(counts.values())

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
                self._workspace_usage_metric(
                    "em", "EXPERIMENT_COUNT", ws, ws_experiment_total, ws_counts
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

        for ws in tqdm(list(workspaces), desc="Opik workspaces", unit="ws"):
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
            ws_trace_counts: dict = defaultdict(float)

            for project in tqdm(projects, desc=f"Opik {ws}", unit="proj", leave=False):
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
                            counts[format_time_key(dp.time, self.units)] += _as_float(
                                dp.value
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

                    try:
                        trace_resp = client.rest_client.projects.get_project_metrics(
                            project.id,
                            metric_type="TRACE_COUNT",
                            interval=interval,
                            interval_start=interval_start,
                            interval_end=now,
                        )
                        trace_counts: dict = defaultdict(float)
                        for result in trace_resp.results or []:
                            for dp in result.data or []:
                                trace_counts[
                                    format_time_key(dp.time, self.units)
                                ] += _as_float(dp.value)

                        usage.append(
                            UsageMetric(
                                platform="opik",
                                workspace=ws,
                                metric="TRACE_COUNT",
                                value=sum(trace_counts.values()),
                                project=project.name,
                                series=(
                                    continuous_series(dict(trace_counts), self.units)
                                    if trace_counts
                                    else []
                                ),
                            )
                        )
                        for key, val in trace_counts.items():
                            ws_trace_counts[key] += val
                    except Exception as exc:
                        print(
                            f"Warning: failed to collect Opik TRACE_COUNT for "
                            f"{ws}/{getattr(project, 'name', '?')}: {exc}"
                        )
                except Exception as exc:
                    print(
                        f"Warning: failed to collect Opik project "
                        f"{ws}/{getattr(project, 'name', '?')}: {exc}"
                    )
                    continue

            usage.append(
                self._workspace_usage_metric(
                    "opik", "SPAN_COUNT", ws, sum(ws_counts.values()), ws_counts
                )
            )
            usage.append(
                self._workspace_usage_metric(
                    "opik",
                    "TRACE_COUNT",
                    ws,
                    sum(ws_trace_counts.values()),
                    ws_trace_counts,
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
        selected_entries = [
            ws_entry
            for ws_entry in all_workspaces
            # The MPM inventory shape is not verifiable live; guard against
            # malformed elements so one bad entry can't crash the report.
            if isinstance(ws_entry, dict) and ws_entry.get("workspaceName") in requested
        ]
        for ws_entry in tqdm(selected_entries, desc="MPM workspaces", unit="ws"):
            ws = ws_entry.get("workspaceName")

            models = ws_entry.get("models", []) or []
            ws_counts: dict = defaultdict(float)
            ws_total = 0.0

            for model in tqdm(models, desc=f"MPM {ws}", unit="model", leave=False):
                if not isinstance(model, dict):
                    continue
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
                self._workspace_usage_metric(
                    "mpm", "PREDICTION_VOLUME", ws, ws_total, ws_counts
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
            counts[format_time_key(t, self.units)] += _as_float(y)
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
    """Render `report_data` to a self-contained HTML file at `output`.

    Delegates to `cometx.cli.admin_growth_render.write_html` (C8).
    """
    return write_html(report_data, output)


def _open(path):
    import webbrowser

    webbrowser.open("file://" + os.path.abspath(path))
