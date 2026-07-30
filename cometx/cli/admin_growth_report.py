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
"""cometx admin growth-report — org-wide people/usage growth from the admin
chargeback report, rendered as a self-contained HTML page.

Sources exclusively from `/api/admin/chargeback/report` (admin API key
required): organization overview, users, leaderboards, and a
personal-vs-service-account split. Distinct from `admin usage-report`
(experiment counts over time, PDF/Streamlit).
"""

from __future__ import annotations

import dataclasses
import datetime
import os
import re

from cometx.cli.admin_growth_render import build_html, write_html
from cometx.cli.admin_growth_users import (
    _extract_licensed_users,
    active_series,
    adoption_rate_series,
    adoption_stats,
    bottom_users,
    capability_series,
    churn_series,
    classify_accounts,
    em_user_breakdown_series,
    opik_user_breakdown_series,
    parse_users,
    parse_workspaces,
    platform_mix,
    top_users,
    workspace_active_series,
    workspace_active_stats,
    workspace_churn_series,
    workspace_org_totals,
)
from cometx.utils import (
    InvalidServerURLError,
    admin_api_url,
    fetch_chargeback_report,
    format_time_key,
)

# `build_html` is re-exported here so the growth-report module is the single
# import surface (tests + callers import it from here). Declaring it in
# `__all__` documents the intentional re-export without a per-line noqa.
__all__ = [
    "generate_growth_report",
    "GrowthReporter",
    "GrowthReportError",
    "build_html",
    "write_html",
]


class GrowthReportError(Exception):
    """Raised when the report cannot be built -- chiefly when the chargeback
    admin endpoint is unavailable (e.g. a non-admin API key). There is no SDK
    fallback in this build."""


@dataclasses.dataclass(frozen=True)
class Window:
    start: datetime.datetime
    end: datetime.datetime
    units: str = "month"


def _ms_to_utc(ms) -> datetime.datetime:
    """Convert an epoch-milliseconds timestamp to a timezone-aware UTC
    datetime."""
    return datetime.datetime.fromtimestamp(ms / 1000, tz=datetime.timezone.utc)


def _num(value):
    """Render a metric value as an int when it has no fractional part, else
    a float -- avoids "3.0" showing up in KPI/table cells for counts that
    are naturally integers but arrive as floats from the collectors."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _lb_value(value):
    """Normalize a leaderboard bar value the same way workspace rows are
    rendered: round a fractional metric (e.g. `em_score`, which folds in
    `data_logged_mb`) to a clean integer via `_num(round(...))`. Passes
    `None` through unchanged so a metric a user lacks stays absent rather
    than becoming 0."""
    if value is None:
        return None
    if isinstance(value, float):
        return _num(round(value))
    return value


def _window_growth(created_ms, window):
    """Growth over the analysis window from a collection of creation timestamps
    (epoch ms): `new_in` = items created within `[window.start, window.end]`,
    `before` = items created before `window.start`, `pct` = new_in/before*100
    (0-guarded). Shared by the workspace- and user-growth KPIs so the window
    math lives in one place. `None` timestamps are skipped."""
    new_in = before = 0
    for c in created_ms:
        if c is None:
            continue
        dt = _ms_to_utc(c)
        if window.start <= dt <= window.end:
            new_in += 1
        elif dt < window.start:
            before += 1
    pct = round(new_in / before * 100, 1) if before else 0.0
    return {"new_in": new_in, "before": before, "pct": pct}


def _extract_service_account_names(payload) -> "set[str] | None":
    """Defensively unwrap the `/admin/service-accounts` response into a
    flat set of account names, tolerating several plausible response
    shapes since the exact schema isn't documented in this codebase: a
    bare list of entries, or a dict wrapping that list under a
    service-account-specific container key (`serviceAccounts`,
    `accounts`). Each entry may be a plain string, or a dict carrying the
    name under `name`, `username`, or `email` (mirrors
    `_extract_licensed_users`'s defensive style in `admin_growth_users.py`).

    A generic `users` key is deliberately NOT accepted: if the endpoint
    ever returned the full user roster under it, every user would be
    classified as a service account and the report would label that
    inversion authoritative ("admin API"). Better to fail the parse and
    fall back to the labeled heuristic.

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
        for key in ("serviceAccounts", "accounts"):
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
        # Reuse the shared validated URL builder so this endpoint honors the
        # same scheme/host/path-prefix handling as fetch_chargeback_report
        # (the two admin endpoints were previously built inconsistently).
        url = admin_api_url(
            api.config["comet.url_override"], "/api/admin/service-accounts"
        )
        response = api._client.get(
            url, headers={"Authorization": api.api_key}, params={}
        )
        payload = response.json()
        return _extract_service_account_names(payload)
    except Exception:
        return None


def _short_api_error(exc):
    """Condense a verbose SDK/HTTP exception (which may dump full response
    headers, cookies, and CSP) into a single short line: "status: message"
    when parseable, else a truncated one-liner. Keeps per-workspace warnings
    readable instead of pasting a wall of response headers per failure."""
    text = " ".join(str(exc).split())
    status = re.search(r"status_code:\s*(\d+)", text)
    message = re.search(r"'message':\s*'([^']*)'", text)
    if status or message:
        parts = []
        if status:
            parts.append(status.group(1))
        if message:
            parts.append(message.group(1))
        return ": ".join(parts)
    # Fallback (regexes missed): this string can surface in a user-facing
    # GrowthReportError, so drop everything from the first sensitive marker
    # onward -- verbose SDK/HTTP errors dump headers, cookies, body, and CSP.
    lowered = text.lower()
    cut = len(text)
    # NB: markers are matched as substrings anywhere in the text, so each must
    # be specific enough not to fire on an incidental hostname/message word.
    # "content-security-policy" already covers the CSP header; a bare "csp"
    # (3 chars) would truncate errors that merely happen to contain those
    # letters, so it is intentionally not listed.
    for marker in (
        "headers:",
        "header:",
        "cookie",
        "body:",
        "content-security-policy",
        "set-cookie",
    ):
        idx = lowered.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    text = text[:cut].strip() or "unexpected error"
    return text if len(text) <= 160 else text[:157] + "..."


def _chargeback_licensed_users(chargeback):
    """Raw licensed-user records from a chargeback dict. Delegates to the single
    shared unwrapper `admin_growth_users._extract_licensed_users` so the
    `licensedUsers` / `report` / bare-list handling lives in exactly one place
    (used by both `parse_users` and `_scope_chargeback`)."""
    return _extract_licensed_users((chargeback or {}).get("users"))


def _scope_chargeback(chargeback, workspace_names):
    """Narrow an org-wide chargeback dict to `workspace_names`: keep only those
    workspaces and only the users who are members of them. Lets the People /
    leaderboards / personal-vs-service layers match an explicit workspace
    selection instead of always reporting org-wide."""
    wanted = set(workspace_names)
    workspaces = [
        w for w in (chargeback.get("workspaces") or []) if w.get("name") in wanted
    ]
    members = {
        m.get("userName")
        for w in workspaces
        for m in (w.get("members") or [])
        if m.get("userName")
    }
    scoped_users = [
        u
        for u in _chargeback_licensed_users(chargeback)
        if u.get("username") in members
    ]
    return {
        **chargeback,
        "workspaces": workspaces,
        "users": {"licensedUsers": scoped_users},
    }


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


def generate_growth_report(
    api,
    workspaces,
    *,
    window="7d",
    units="month",
    output="growth_report.html",
    no_open=False,
    active_window="60d",
    leaderboard_top_n=5,
    exclude_personal=False,
    personal_pattern=None,
):
    reporter = GrowthReporter(
        api,
        window=window,
        units=units,
        active_window=active_window,
        leaderboard_top_n=leaderboard_top_n,
        exclude_personal=exclude_personal,
        personal_pattern=personal_pattern,
    )
    report_data = reporter.build(workspaces)
    path = write_growth_html(report_data, output)
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
        active_window="60d",
        leaderboard_top_n=5,
        exclude_personal=False,
        personal_pattern=None,
    ):
        self.api = api
        self.window = window
        self.units = units
        self.active_window = active_window
        self.leaderboard_top_n = leaderboard_top_n
        self.exclude_personal = exclude_personal
        self.personal_pattern = personal_pattern
        self._warned_no_personal_pattern = False

    def build(self, workspaces):
        """Fetch the org-wide chargeback report (admin API key required) and
        assemble chargeback-only report_data. Raises GrowthReportError when the
        chargeback endpoint is unavailable -- there is no SDK fallback."""
        now = self._now()
        window = parse_window(self.window, now, self.units)
        print("Fetching chargeback report (admin API)...")
        try:
            chargeback = fetch_chargeback_report(self.api)
        except InvalidServerURLError as exc:
            # A malformed --host / url_override is a configuration problem, not
            # an auth failure. Surface it as-is rather than asserting the API
            # key isn't admin. Caught by its own type, not `ValueError`: a
            # non-JSON 200 (SSO/proxy login page) raises `json.JSONDecodeError`
            # -- also a `ValueError` -- and belongs in the handler below, which
            # reports an unusable endpoint rather than a bad URL.
            raise GrowthReportError(
                f"growth-report could not reach the chargeback endpoint: {exc}"
            ) from exc
        except Exception as exc:
            raise GrowthReportError(
                "growth-report requires an admin API key: the chargeback "
                f"endpoint is unavailable ({_short_api_error(exc)}). This "
                "report is built entirely from the admin chargeback report."
            ) from exc
        chargeback = self._filter_personal_chargeback(chargeback)
        print("Building report...")
        now_ms = int(now.timestamp() * 1000)
        scope = set(workspaces) if workspaces else None
        return self._assemble_report_data(chargeback, window, now_ms, scope=scope)

    def _now(self):
        return datetime.datetime.now(datetime.timezone.utc)

    def _units_adverb(self):
        """Adverb form of the chart bucket unit for hint text, so `day` reads
        `daily` rather than the `{units}ly` -> `dayly` typo."""
        return {
            "hour": "hourly",
            "day": "daily",
            "week": "weekly",
            "month": "monthly",
        }.get(self.units, self.units + "ly")

    def _personal_pattern_compiled(self):
        """Compiled --personal-pattern regex, or None (with a once-per-reporter
        warning) when the flag is off, no pattern was given, or the pattern is
        invalid."""
        if not self.exclude_personal:
            return None
        if not self.personal_pattern:
            if not self._warned_no_personal_pattern:
                print(
                    "Warning: --exclude-personal has no effect without "
                    "--personal-pattern; skipping personal-workspace exclusion"
                )
                self._warned_no_personal_pattern = True
            return None
        try:
            return re.compile(self.personal_pattern)
        except re.error as exc:
            if not self._warned_no_personal_pattern:
                print(
                    f"Warning: invalid --personal-pattern "
                    f"{self.personal_pattern!r} ({exc}); skipping "
                    "personal-workspace exclusion"
                )
                self._warned_no_personal_pattern = True
            return None

    def _filter_personal_chargeback(self, chargeback):
        """Drop personal workspaces (name matches --personal-pattern) from the
        chargeback workspace list when --exclude-personal is set. The user
        roster (users.report / licensedUsers) is left whole; only the
        workspace list and its memberships are trimmed. Returns a shallow copy;
        never mutates the input."""
        pattern = self._personal_pattern_compiled()
        if pattern is None:
            return chargeback
        workspaces = chargeback.get("workspaces") or []
        kept = [w for w in workspaces if not pattern.search(w.get("name") or "")]
        return {**chargeback, "workspaces": kept}

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

    def _workspace_chargeback_table(self, ws_records):
        """Org-wide by-workspace table from chargeback (top 20 by experiments):
        projects, experiments and data logged per workspace."""
        rows = sorted(ws_records, key=lambda w: -w.num_experiments)[:20]
        return {
            "title": "By workspace (org-wide, chargeback)",
            "headers": ["Workspace", "Members", "Projects", "Experiments", "Data (MB)"],
            "rows": [
                [
                    w.name,
                    len(w.members),
                    w.num_projects,
                    _num(w.num_experiments),
                    _num(round(w.data_mb)),
                ]
                for w in rows
            ],
        }

    @staticmethod
    def _workspace_growth_kpi(users, window):
        """Org-wide workspace growth over the analysis window: workspaces first
        seen in-window vs. those existing before it, using each workspace's
        earliest member `created_at` as its creation proxy (mirrors the Users
        section's user-growth KPI). Returns {new_in, before, pct}."""
        ws_created: dict = {}
        for u in users:
            if u.created_at is None:
                continue
            for ws in u.workspaces:
                cur = ws_created.get(ws)
                if cur is None or u.created_at < cur:
                    ws_created[ws] = u.created_at
        return _window_growth(ws_created.values(), window)

    def _build_unified_section(
        self,
        window,
        people_users=None,
        ws_records=None,
        now_ms=None,
        active_window_days=None,
    ):
        """Organization overview (chargeback). Org-wide KPIs (workspaces,
        projects, new-in-window %, active-workspaces %), the platform mix, and
        the workspace total-vs-active / added-vs-deleted charts -- all derived
        from the chargeback report. Every workspace-level number lives here; the
        Users section stays about users."""
        cb_charts = []
        if people_users:
            ws_active_pts = workspace_active_series(
                people_users,
                self.units,
                now_ms,
                active_window_days,
                all_workspaces={w.name for w in (ws_records or [])},
            )
            if ws_active_pts:
                cb_charts.append(
                    {
                        "id": "chart-unified-workspaces-active",
                        "kind": "lines",
                        "title": "Workspaces: total vs. active",
                        "hint": (
                            f"org-wide (chargeback); active window "
                            f"{self.active_window}"
                        ),
                        "legend": [
                            {"label": "Total", "color": "--sdk"},
                            {"label": "Active", "color": "--ok"},
                        ],
                        "data": {
                            "categories": ["total", "active"],
                            "labels": {"total": "Total", "active": "Active"},
                            "colors": ["--sdk", "--ok"],
                            "points": ws_active_pts,
                        },
                    }
                )
            ws_churn = workspace_churn_series(people_users, self.units, now_ms)
            if ws_churn:
                # added/deleted as bars, with an overlaid growth-rate line
                # (added workspaces / surviving total at start of period).
                # The base tracks the *net* surviving total, so it must
                # subtract deletions as well as add creations each bucket;
                # otherwise the denominator drifts high and the rate reads low.
                churn_pts = []
                surviving_total = 0
                for pt in ws_churn:
                    added = pt["values"]["added"]
                    deleted = pt["values"]["deleted"]
                    rate = (
                        round(added / surviving_total * 100, 1)
                        if surviving_total
                        else 0.0
                    )
                    churn_pts.append(
                        {
                            "key": pt["key"],
                            "values": {
                                "added": added,
                                "deleted": deleted,
                                "rate": rate,
                            },
                        }
                    )
                    surviving_total += added - deleted
                cb_charts.append(
                    {
                        "id": "chart-unified-workspace-churn",
                        "kind": "barsLine",
                        "title": "Workspaces added vs. deleted",
                        "hint": (
                            f"org-wide (chargeback) · {self._units_adverb()} · "
                            "deletion is a proxy"
                        ),
                        "legend": [
                            {"label": "Added", "color": "--ok"},
                            {"label": "Deleted", "color": "--warn"},
                            {"label": "Growth rate", "color": "--accent"},
                        ],
                        "data": {
                            "points": churn_pts,
                            "bars": ["added", "deleted"],
                            "line": "rate",
                            "bar_labels": {
                                "added": "Added",
                                "deleted": "Deleted",
                                "rate": "Growth rate",
                            },
                            "bar_colors": ["--ok", "--warn"],
                            "line_label": "Growth rate",
                            "line_color": "--accent",
                            "line_suffix": "%",
                        },
                    }
                )

        # Org-wide platform mix (chargeback): EM-only / Opik-only / both /
        # neither. MPM is not in chargeback, so it's excluded; Opik is a
        # per-user proxy (a member's Opik usage is attributed to all their
        # workspaces).
        if ws_records:
            mix = platform_mix(ws_records, people_users or [])
            mix_rows = [
                {"label": "EM only", "value": mix["em_only"]},
                {"label": "Opik only", "value": mix["opik_only"]},
                {"label": "EM + Opik", "value": mix["both"]},
                {"label": "Neither", "value": mix["neither"]},
            ]
            cb_charts.append(
                {
                    "id": "chart-unified-platform-mix",
                    "kind": "groupedBarsH",
                    "title": "Workspace platform mix",
                    "hint": (
                        "org-wide (chargeback); Opik is a per-user proxy; "
                        "MPM not represented in chargeback"
                    ),
                    "data": {"rows": mix_rows},
                }
            )

        # ORGANIZATION OVERVIEW (chargeback): org-wide headline numbers +
        # chargeback-derived charts. Degrades to zeros if ws_records is empty.
        ws_records = ws_records or []
        org = workspace_org_totals(ws_records)
        wa = workspace_active_stats(
            people_users or [],
            now_ms,
            active_window_days,
            all_workspaces={w.name for w in ws_records},
        )
        wg = self._workspace_growth_kpi(people_users or [], window)
        kpis = [
            {"label": "Total workspaces", "value": org["workspaces"]},
            {
                "label": "Total projects",
                "value": org["projects"],
                "sub": "EM projects",
                "tone": "ok",
            },
            {
                "label": f"New in {self.window or '7d'} (% of base)",
                "value": f"{wg['pct']}%",
                "sub": f"+{wg['new_in']} new (est. from earliest member)",
            },
            {
                "label": "Active workspaces %",
                "value": f"{wa['active_pct']}%",
                "sub": f"{wa['active']}/{wa['total']} active",
            },
        ]
        return {
            "title": "Organization overview (chargeback)",
            "window_chip": self._window_label(window),
            "kpis": kpis,
            "charts": [c for c in cb_charts if c],
            "table": self._workspace_chargeback_table(ws_records),
        }

    def _active_window_days(self, now):
        """Resolve `self.active_window` (e.g. "60d") into an integer day
        count, via the shared `parse_window` parser (same spec grammar as
        `--window`)."""
        active_window = parse_window(self.active_window, now, self.units)
        return max(1, (active_window.end - active_window.start).days)

    def _build_people_section(self, users, now_ms, window=None):
        """User-level section (workspace metrics live in the Organization
        overview). KPIs (Total / Active / Active % / new-in-window % of base)
        plus over-time
        `lines` charts: active-vs-total, adoption rates, per-capability active
        users, EM/Opik user breakdowns, and users added-vs-deleted. `users` is
        the pre-parsed (and, when scoped, already-filtered) chargeback user
        list. Returns `None` when there are no users."""
        if not users:
            return None
        now = _ms_to_utc(now_ms)
        active_window_days = self._active_window_days(now)

        stats = adoption_stats(users, now_ms, active_window_days)

        kpis = [
            {"label": "Total users", "value": stats["total"]},
            {
                "label": f"Active users ({self.active_window})",
                "value": stats["active"],
            },
            {"label": "Active users %", "value": f"{stats['adoption_pct']}%"},
        ]

        # User growth over the analysis window: new accounts in window /
        # accounts before window (shares _window_growth with the workspace KPI).
        if window is not None:
            growth = _window_growth((u.created_at for u in users), window)
            kpis.append(
                {
                    "label": f"New in {self.window or '7d'} (% of base)",
                    "value": f"{growth['pct']}%",
                    "sub": f"+{growth['new_in']} new",
                }
            )

        active_pts = active_series(users, self.units, now_ms, active_window_days)
        if not active_pts:
            # No user has a known created_at to bucket from -- degrade to a
            # single current-period point rather than an empty chart.
            active_pts = [
                {
                    "key": format_time_key(now, self.units),
                    "values": {"total": stats["total"], "active": stats["active"]},
                }
            ]

        charts = [
            {
                "id": "chart-people-active-total",
                "kind": "lines",
                "title": "Active vs. total users",
                "hint": f"active window {self.active_window} · {self._units_adverb()}",
                "legend": [
                    {"label": "Total", "color": "--sdk"},
                    {"label": "Active", "color": "--ok"},
                ],
                "data": {
                    "categories": ["total", "active"],
                    "labels": {"total": "Total", "active": "Active"},
                    "colors": ["--sdk", "--ok"],
                    "points": active_pts,
                },
            }
        ]

        rate_pts = adoption_rate_series(users, self.units, now_ms, active_window_days)
        if rate_pts:
            # adoption_rate_series omits em/opik when that capability has no
            # signal, so build the chart categories/legend from the keys that
            # are actually present (overall is always there).
            present = rate_pts[0]["values"].keys()
            rate_spec = [
                ("overall", "Overall", "--ok"),
                ("em", "Experimentation", "--sdk"),
                ("opik", "Opik", "--accent"),
            ]
            rate_spec = [s for s in rate_spec if s[0] in present]
            charts.append(
                {
                    "id": "chart-people-adoption-rate",
                    "kind": "lines",
                    "title": "Adoption rates",
                    "hint": (
                        f"active users / total; active window "
                        f"{self.active_window} · {self._units_adverb()}"
                    ),
                    "legend": [
                        {"label": lbl, "color": col} for _, lbl, col in rate_spec
                    ],
                    "data": {
                        "categories": [k for k, _, _ in rate_spec],
                        "labels": {k: lbl for k, lbl, _ in rate_spec},
                        "colors": [col for _, _, col in rate_spec],
                        "points": rate_pts,
                    },
                }
            )

        cap_pts = capability_series(users, self.units, now_ms, active_window_days)
        if cap_pts:
            charts.append(
                {
                    "id": "chart-people-capability",
                    "kind": "lines",
                    "title": "Active users by capability",
                    "hint": (
                        f"active window {self.active_window} · "
                        f"{self._units_adverb()}"
                    ),
                    "legend": [
                        {"label": "EM", "color": "--sdk"},
                        {"label": "Opik", "color": "--accent"},
                    ],
                    "data": {
                        "categories": ["em", "opik"],
                        "labels": {"em": "EM", "opik": "Opik"},
                        "colors": ["--sdk", "--accent"],
                        "points": cap_pts,
                    },
                }
            )

        window_hint = f"active window {self.active_window} · {self._units_adverb()}"

        em_pts = em_user_breakdown_series(users, self.units, now_ms, active_window_days)
        if em_pts:
            charts.append(
                {
                    "id": "chart-people-em-breakdown",
                    "kind": "lines",
                    "title": "EM users: active, experimenters, data pushers",
                    "hint": window_hint + "; experimenters/data-pushers from totals",
                    "legend": [
                        {"label": "Active", "color": "--sdk"},
                        {"label": "Experimenters", "color": "--accent"},
                        {"label": "Data pushers", "color": "--warn"},
                    ],
                    "data": {
                        "categories": ["active", "experimenters", "data_pushers"],
                        "labels": {
                            "active": "Active",
                            "experimenters": "Experimenters",
                            "data_pushers": "Data pushers",
                        },
                        "colors": ["--sdk", "--accent", "--warn"],
                        "points": em_pts,
                    },
                }
            )

        opik_pts = opik_user_breakdown_series(
            users, self.units, now_ms, active_window_days
        )
        if opik_pts:
            charts.append(
                {
                    "id": "chart-people-opik-breakdown",
                    "kind": "lines",
                    "title": "Opik users: active, span producers",
                    "hint": window_hint + "; span-producers from totals",
                    "legend": [
                        {"label": "Active", "color": "--accent"},
                        {"label": "Span producers", "color": "--ok"},
                    ],
                    "data": {
                        "categories": ["active", "span_producers"],
                        "labels": {
                            "active": "Active",
                            "span_producers": "Span producers",
                        },
                        "colors": ["--accent", "--ok"],
                        "points": opik_pts,
                    },
                }
            )

        user_churn = churn_series(users, self.units, now_ms)
        if user_churn:
            charts.append(
                {
                    "id": "chart-people-user-churn",
                    "kind": "lines",
                    "title": "Users added vs. deleted",
                    "hint": (
                        f"per period · {self._units_adverb()}; deletions reflect "
                        "soft-deletes still present in the snapshot"
                    ),
                    "legend": [
                        {"label": "Added", "color": "--ok"},
                        {"label": "Deleted", "color": "--warn"},
                    ],
                    "data": {
                        "categories": ["added", "deleted"],
                        "labels": {"added": "Added", "deleted": "Deleted"},
                        "colors": ["--ok", "--warn"],
                        "points": user_churn,
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
                    _num(round(u.data_logged_mb)),
                    _num(u.opik_span_count) if u.opik_span_count is not None else "-",
                ]
                for u in top
            ],
        }

        return {
            "title": "Users",
            "kpis": kpis,
            "charts": charts,
            "table": table,
        }

    # User-level leaderboard metrics: (top_users/bottom_users key, plain
    # label, value-extractor matching admin_growth_users._metric_value).
    _USER_LEADERBOARD_METRICS = [
        (
            "opik_span_count",
            "Opik spans",
            lambda u: u.opik_span_count,
            "org-wide (chargeback)",
        ),
        (
            "em_score",
            "EM activity",
            lambda u: (u.experiment_count or 0) + (u.data_logged_mb or 0),
            "EM activity = experiments + data logged (MB); org-wide (chargeback)",
        ),
    ]

    @staticmethod
    def _leaderboard_chart(chart_id, title, rows, hint=None):
        chart = {
            "id": chart_id,
            "kind": "groupedBarsH",
            "title": title,
            "data": {"rows": rows},
        }
        if hint:
            chart["hint"] = hint
        return chart

    @staticmethod
    def _workspace_rollup(users, value_fn):
        """Sum a per-user chargeback metric across each user's workspaces.
        Approximate: chargeback carries per-user totals (not per-workspace), so
        a user in multiple workspaces contributes their full total to each."""
        totals = {}
        for u in users:
            value = value_fn(u) or 0
            if value <= 0:
                continue
            for ws in u.workspaces:
                totals[ws] = totals.get(ws, 0) + value
        return totals

    def _build_leaderboards_section(self, users, ws_records):
        """Top-N / bottom-N workspace and user leaderboards, all org-wide from
        chargeback. Experiments and projects are ranked EXACTLY from the
        chargeback per-workspace numbers; Opik spans from the chargeback
        per-user rollup (a proxy: a member's spans are attributed to each of
        their workspaces). Bottom-N is active-aware (strictly-positive,
        ascending). Empty metrics are omitted; returns `None` when there is
        nothing to show."""
        n = self.leaderboard_top_n
        charts = []

        def emit_ws(slug, label, totals, hint):
            if not totals:
                return
            ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
            top_rows = [{"label": ws, "value": _num(v)} for ws, v in ranked[:n]]
            active_asc = sorted(
                ((ws, v) for ws, v in totals.items() if v > 0), key=lambda kv: kv[1]
            )
            bottom_rows = [{"label": ws, "value": _num(v)} for ws, v in active_asc[:n]]
            if top_rows:
                charts.append(
                    self._leaderboard_chart(
                        f"chart-lb-{slug}-top",
                        f"Top {n} workspaces by {label}",
                        top_rows,
                        hint,
                    )
                )
            if bottom_rows:
                charts.append(
                    self._leaderboard_chart(
                        f"chart-lb-{slug}-bottom",
                        f"Bottom {n} workspaces by {label}",
                        bottom_rows,
                        hint,
                    )
                )

        org_exact_hint = "org-wide, exact (chargeback per-workspace)"
        org_proxy_hint = (
            "org-wide from chargeback; per-user spans attributed to each of "
            "the user's workspaces (proxy)"
        )

        # Experiments + projects: exact, org-wide from chargeback per-workspace.
        if ws_records:
            emit_ws(
                "ws-experiments",
                "experiments",
                {w.name: w.num_experiments for w in ws_records if w.num_experiments},
                org_exact_hint,
            )
            emit_ws(
                "ws-projects",
                "projects",
                {w.name: w.num_projects for w in ws_records if w.num_projects},
                org_exact_hint,
            )

        # Opik spans: org-wide per-user rollup proxy.
        if users:
            emit_ws(
                "ws-spans",
                "Opik spans",
                self._workspace_rollup(users, lambda u: u.opik_span_count or 0),
                org_proxy_hint,
            )

        if users:
            for key, label, value_fn, hint in self._USER_LEADERBOARD_METRICS:
                top = top_users(users, key, n)
                if top:
                    charts.append(
                        self._leaderboard_chart(
                            f"chart-lb-user-{key}-top",
                            f"Top {n} users by {label}",
                            [
                                {"label": u.username, "value": _lb_value(value_fn(u))}
                                for u in top
                            ],
                            hint,
                        )
                    )
                bottom = bottom_users(users, key, n)
                if bottom:
                    charts.append(
                        self._leaderboard_chart(
                            f"chart-lb-user-{key}-bottom",
                            f"Bottom {n} users by {label}",
                            [
                                {"label": u.username, "value": _lb_value(value_fn(u))}
                                for u in bottom
                            ],
                            hint,
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

        # `source == "heuristic"` means the admin service-accounts fetch was
        # unavailable / failed / returned an unrecognized shape (a genuine empty
        # admin response is honored as admin_api). Say so, rather than implying
        # the admin API responded with zero accounts.
        hint = (
            "Source: service accounts from admin API."
            if source == "admin_api"
            else "Source: heuristic (regex); admin service-accounts API unavailable."
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
        }

    @staticmethod
    def _scope_label(scope, org_workspaces, org_users, scoped_count=None):
        """One-line scope descriptor for the report header. When scoped, the
        count reflects the workspaces actually present after scoping/filtering
        (`scoped_count`), not the raw requested arg list, so the badge matches
        the rendered sections."""
        if scope is not None:
            n = scoped_count if scoped_count is not None else len(scope)
            return (
                f"Scoped to {n} selected workspace(s) "
                "(per-user totals remain org-wide)"
            )
        if org_workspaces is not None:
            return (
                f"Org-wide: {org_workspaces} workspaces, {org_users} users "
                "(chargeback)"
            )
        return "Org-wide (chargeback)"

    def _assemble_report_data(self, chargeback, window, now_ms, scope=None):
        # Org totals for the header, from the FULL (unscoped) chargeback.
        org_users = org_workspaces = None
        try:
            org_users = sum(1 for u in parse_users(chargeback) if not u.suspended)
            org_workspaces = len(parse_workspaces(chargeback))
        except Exception:
            org_users = org_workspaces = None

        scoped = (
            _scope_chargeback(chargeback, scope) if scope is not None else chargeback
        )

        active_window_days = self._active_window_days(_ms_to_utc(now_ms))
        try:
            people_users = parse_users(scoped)
            ws_records = parse_workspaces(scoped)
        except Exception as exc:
            print(
                f"Warning: failed to parse chargeback report: "
                f"{_short_api_error(exc)}"
            )
            people_users = []
            ws_records = []

        sections = {
            "unified": self._build_unified_section(
                window,
                people_users=people_users,
                ws_records=ws_records,
                now_ms=now_ms,
                active_window_days=active_window_days,
            )
        }

        if people_users:
            try:
                people_section = self._build_people_section(
                    people_users, now_ms, window=window
                )
            except Exception as exc:
                print(
                    f"Warning: failed to build users section: "
                    f"{_short_api_error(exc)}"
                )
                people_section = None
            if people_section:
                sections["people"] = people_section

        try:
            leaderboards_section = self._build_leaderboards_section(
                people_users, ws_records
            )
        except Exception as exc:
            print(f"Warning: failed to build leaderboards section; skipping: {exc}")
            leaderboards_section = None
        if leaderboards_section:
            sections["leaderboards"] = leaderboards_section

        try:
            personal_vs_service_section = None
            if people_users:
                service_account_names = _fetch_service_accounts(self.api)
                # `_fetch_service_accounts` already returns None on any failure
                # (endpoint unavailable / unrecognized shape) and a set on
                # success. An empty set is the authoritative "admin API returned
                # zero service accounts" answer, which classify_accounts honors
                # as the admin_api source -- do NOT coerce it to None (that would
                # discard the authoritative answer and fall back to the regex
                # heuristic).
                personal_vs_service_section = self._build_personal_vs_service_section(
                    people_users, service_account_names
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
                "source": "Comet Admin API (chargeback)",
                "scope": self._scope_label(
                    scope, org_workspaces, org_users, scoped_count=len(ws_records)
                ),
            },
            "window": self._build_window_block(window, 0),
            "sections": sections,
        }


def write_growth_html(report_data, output):
    """Render `report_data` to a self-contained HTML file at `output`.

    Delegates to `cometx.cli.admin_growth_render.write_html` (C8).
    """
    return write_html(report_data, output)


def _open(path):
    import webbrowser

    webbrowser.open("file://" + os.path.abspath(path))
