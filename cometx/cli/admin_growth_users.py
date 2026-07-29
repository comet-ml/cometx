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
"""People/usage layer for `cometx admin growth-report`, derived from the
chargeback report.

Pure parse + derivation functions only: no HTTP calls, no rendering, no
CLI wiring. Later tasks (fetch, render, CLI) import from this module.
"""

from __future__ import annotations

import dataclasses
import datetime
import logging
import re

from cometx.utils import format_time_key, get_next_time_key, parse_time_key

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class UserRecord:
    """A single licensed user, as reported by the chargeback report, with
    workspace membership reverse-mapped in."""

    username: str
    email: str
    created_at: int
    deleted_at: "int | None"
    suspended: bool
    last_used_at: int
    experiment_count: float
    data_logged_mb: float
    opik_span_count: "float | None" = None
    em_last_used_at: "int | None" = None
    opik_last_used_at: "int | None" = None
    workspaces: "list" = dataclasses.field(default_factory=list)


def _extract_licensed_users(users_field) -> list:
    """Defensively unwrap the `chargeback["users"]` container, which may be:
    a dict with a `report` key, a dict with a `licensedUsers` key, or a bare
    list of user records."""
    if isinstance(users_field, dict):
        if "licensedUsers" in users_field:
            return users_field.get("licensedUsers") or []
        if "report" in users_field:
            report = users_field.get("report")
            return _extract_licensed_users(report) if report is not None else []
        return []
    if isinstance(users_field, list):
        return users_field
    return []


def parse_users(chargeback: dict) -> "list[UserRecord]":
    """Parse `chargeback["users"]` into `UserRecord`s, reverse-mapping
    `workspaces[].members[].userName` into each user's `workspaces` list."""
    raw_users = _extract_licensed_users((chargeback or {}).get("users"))

    workspaces_by_username: "dict[str, list[str]]" = {}
    for ws in (chargeback or {}).get("workspaces") or []:
        ws_name = ws.get("name")
        for member in ws.get("members") or []:
            user_name = member.get("userName")
            if user_name is None:
                continue
            workspaces_by_username.setdefault(user_name, []).append(ws_name)

    records = []
    for raw in raw_users:
        username = raw.get("username")
        records.append(
            UserRecord(
                username=username,
                email=raw.get("email"),
                created_at=raw.get("createdAt"),
                deleted_at=raw.get("deletedAt"),
                suspended=raw.get("suspended", False),
                last_used_at=raw.get("lastUsedAt"),
                experiment_count=raw.get("experimentCount", 0),
                data_logged_mb=raw.get("dataLoggedMb", 0.0),
                opik_span_count=raw.get("opikSpanCount"),
                em_last_used_at=raw.get("emLastUsedAt"),
                opik_last_used_at=raw.get("opikLastUsedAt"),
                workspaces=list(workspaces_by_username.get(username, [])),
            )
        )
    return records


def _within_window(ts: "int | None", window_end_ms: int, window_ms: int) -> bool:
    """True if timestamp `ts` falls within a rolling window of `window_ms`
    size ending at `window_end_ms` (inclusive of both bounds). False if
    `ts` is None (field absent for this user/capability)."""
    if ts is None:
        return False
    return (window_end_ms - window_ms) <= ts <= window_end_ms


def is_active(user: UserRecord, now_ms: int, active_window_days: int) -> bool:
    """True if `user.last_used_at` falls within the active window ending at
    `now_ms` (inclusive of both bounds)."""
    window_ms = active_window_days * 86400 * 1000
    return _within_window(user.last_used_at, now_ms, window_ms)


def adoption_stats(users, now_ms: int, active_window_days: int) -> dict:
    """Aggregate adoption stats. `total` excludes suspended users;
    `adoption_pct` is 0-guarded when there are no non-suspended users."""
    non_suspended = [u for u in users if not u.suspended]
    total = len(non_suspended)
    active = sum(1 for u in non_suspended if is_active(u, now_ms, active_window_days))
    adoption_pct = round(active / total * 100, 1) if total else 0.0
    return {"total": total, "active": active, "adoption_pct": adoption_pct}


def workspace_active_stats(
    users, now_ms: int, active_window_days: int, all_workspaces=None
) -> dict:
    """Workspace-level analogue of `adoption_stats`. A workspace is `active`
    when it has a non-suspended member active within the window (same rule as
    active users). `total` is the size of `all_workspaces` when provided (the
    authoritative chargeback workspace list, so the denominator matches the
    "Total workspaces" count even for zero-member/inactive workspaces);
    otherwise it falls back to the set of workspaces seen via membership.
    `active_pct` is 0-guarded."""
    active_ws = set()
    member_ws = set()
    for u in users:
        if u.suspended:
            continue
        member_active = is_active(u, now_ms, active_window_days)
        for ws in u.workspaces:
            member_ws.add(ws)
            if member_active:
                active_ws.add(ws)
    total = len(all_workspaces) if all_workspaces is not None else len(member_ws)
    active = len(active_ws)
    active_pct = round(active / total * 100, 1) if total else 0.0
    return {"total": total, "active": active, "active_pct": active_pct}


def _metric_value(user: UserRecord, key: str):
    """Return the raw metric value for `key`, or None if the metric is
    absent for this user (only possible for opik_span_count)."""
    if key == "em_score":
        return (user.experiment_count or 0) + (user.data_logged_mb or 0)
    if key == "opik_span_count":
        return user.opik_span_count
    raise ValueError("Unsupported key: {}".format(key))


def _rankable_users(users):
    """Users eligible for the activity leaderboards: exclude deleted and
    suspended accounts, matching the exclusions the time-series builders and
    `classify_accounts` apply, so the leaderboard doesn't list a
    deleted/suspended account as a current top user (which would contradict the
    active/total and capability charts in the same report)."""
    return [u for u in users if u.deleted_at is None and not u.suspended]


def top_users(users, key: str, n: int) -> "list[UserRecord]":
    """Top `n` users by `key` (descending), over non-deleted/non-suspended
    users, skipping those for whom the metric is absent (None)."""
    scored = [(u, _metric_value(u, key)) for u in _rankable_users(users)]
    scored = [(u, v) for u, v in scored if v is not None]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [u for u, _ in scored[:n]]


def bottom_users(users, key: str, n: int) -> "list[UserRecord]":
    """Bottom `n` users by `key` (ascending), over non-deleted/non-suspended
    users, active-aware: only considers users whose metric value is strictly
    positive."""
    scored = [(u, _metric_value(u, key)) for u in _rankable_users(users)]
    scored = [(u, v) for u, v in scored if v is not None and v > 0]
    scored.sort(key=lambda pair: pair[1])
    return [u for u, _ in scored[:n]]


# Labeled regex fallback used by `classify_accounts` when the admin
# `/admin/service-accounts` endpoint is unavailable (disabled, 403/404, or
# any other fetch failure). Anchored to name-segment/domain boundaries so a
# real user like "lisa-brown" or "abbot-jones" (which merely CONTAIN "sa-"
# / "bot-" mid-string) is never mislabeled as a service account.

# Prefix-style tokens: match only at the start of the username, or right
# after a `.`/`_`/`-` separator (i.e. as a whole name-segment prefix).
_SERVICE_ACCOUNT_PREFIX_PATTERN = re.compile(
    r"(?:^|[._-])(?:svc|sa|bot|pipeline)-", re.IGNORECASE
)
# Keyword token: the client notebook flags any username containing "automated"
# as a service account (substring match). Kept as a plain substring for parity.
_SERVICE_ACCOUNT_KEYWORD_PATTERN = re.compile(r"automated", re.IGNORECASE)
# Segment token: matched as-is against the username (already has a leading
# hyphen boundary, so no separate anchoring is needed). Bounded at the tail
# to end-of-string or the next separator so "jane-service-accountant" is not
# mislabeled as a service account.
_SERVICE_ACCOUNT_SEGMENT_PATTERN = re.compile(
    r"-service-account(?:$|[._-])", re.IGNORECASE
)
# Domain token: matched against the email's domain only, anchored to the
# END of the domain, so "sagemaker-integration.com" matches but a lookalike
# domain like "sagemaker-integration.com.evil.com" does not.
_SERVICE_ACCOUNT_DOMAIN_PATTERN = re.compile(
    r"(?:^|\.)sagemaker-integration\.com$", re.IGNORECASE
)


def _looks_like_service_account(user: UserRecord) -> bool:
    """Heuristic-only check (used when no authoritative service-account set
    is available): does the username match one of the known service-account
    naming conventions, or does the email's domain match a known
    service-account domain."""
    username = user.username or ""
    if _SERVICE_ACCOUNT_PREFIX_PATTERN.search(username):
        return True
    if _SERVICE_ACCOUNT_SEGMENT_PATTERN.search(username):
        return True
    if _SERVICE_ACCOUNT_KEYWORD_PATTERN.search(username):
        return True
    email = user.email or ""
    domain = email.rsplit("@", 1)[-1] if "@" in email else ""
    return bool(domain and _SERVICE_ACCOUNT_DOMAIN_PATTERN.search(domain))


def classify_accounts(users, service_account_names=None) -> dict:
    """Split all NON-DELETED users into "personal" vs "service" buckets and
    sum experiments / data / spans across each bucket (suspended users are
    still non-deleted, so they are included in whichever bucket they land
    in; their own metrics are typically 0 either way).

    `service_account_names`, when a set, is the authoritative list fetched
    from the admin `/admin/service-accounts` endpoint: membership in that
    set decides the split, and `source` is reported as "admin_api".
    Because the endpoint may identify accounts by either username or
    email, a user matches if *either* their `username` or their `email` is
    in the set (avoids the mirror failure where entries carry only email
    but users are matched on username, silently classifying everyone as
    personal). When it is `None` (the endpoint failed, was disabled, or
    isn't present in this deployment), falls back to a labeled regex
    heuristic (`_looks_like_service_account`) and reports `source` as
    "heuristic" so callers can surface which method produced the split.
    """
    if service_account_names is not None:
        source = "admin_api"

        def is_service(user: UserRecord) -> bool:
            return (
                user.username in service_account_names
                or user.email in service_account_names
            )

    else:
        source = "heuristic"
        is_service = _looks_like_service_account

    totals = {
        "personal": {"experiments": 0, "data": 0, "spans": 0},
        "service": {"experiments": 0, "data": 0, "spans": 0},
    }

    for user in users:
        if user.deleted_at is not None:
            continue
        bucket = totals["service"] if is_service(user) else totals["personal"]
        bucket["experiments"] += user.experiment_count or 0
        bucket["data"] += user.data_logged_mb or 0
        bucket["spans"] += user.opik_span_count or 0

    return {
        "personal": totals["personal"],
        "service": totals["service"],
        "source": source,
    }


def _ms_to_dt(ms: int) -> "datetime.datetime":
    """Epoch-ms -> tz-aware UTC datetime (matches the `_ms_to_utc` helper
    in admin_growth_report.py; duplicated here to avoid a circular import,
    since that module imports FROM this one)."""
    return datetime.datetime.fromtimestamp(ms / 1000, tz=datetime.timezone.utc)


def _dt_to_ms(dt: "datetime.datetime") -> int:
    """Datetime -> epoch-ms. `parse_time_key` (from `cometx.utils`) returns
    naive datetimes; treat those as UTC (consistent with `_ms_to_dt` above)
    rather than the platform-local timezone that `.timestamp()` would
    otherwise assume."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return int(dt.timestamp() * 1000)


def _bucket_keys(earliest_ms: int, now_ms: int, units: str) -> "list[str]":
    """Zero-filled, ordered bucket keys from the bucket containing
    `earliest_ms` through the bucket containing `now_ms`, inclusive.

    Note: `continuous_series` (mentioned in the task brief) actually lives
    in `admin_growth_report.py`, not `cometx.utils` -- and that module
    imports FROM this one, so importing it back here would be circular.
    This is the same zero-fill approach (via `get_next_time_key`), built
    locally from the primitives that do live in `cometx.utils`
    (`format_time_key`, `get_next_time_key`, `parse_time_key`)."""
    start_key = format_time_key(_ms_to_dt(earliest_ms), units)
    end_key = format_time_key(_ms_to_dt(now_ms), units)
    keys = [start_key]
    key = start_key
    guard = 0
    guard_max = 100_000
    while key != end_key and guard < guard_max:
        key = get_next_time_key(key, units)
        keys.append(key)
        guard += 1
    if key != end_key:
        # Span too large for these units (e.g. hourly buckets over years):
        # the series is truncated at the guard, so warn rather than silently
        # returning a chart that stops short of `now`.
        LOGGER.warning(
            "Growth series truncated at %d %s buckets (spanning %s..%s); "
            "the chart stops before the current period. Use coarser units.",
            guard_max,
            units,
            start_key,
            end_key,
        )
    return keys


def _iter_buckets(users: "list[UserRecord]", units: str, now_ms: int):
    """Yield `(key, bucket_end_ms, existing)` for each zero-filled bucket
    from the earliest `created_at` through `now_ms`. `existing` is the
    list of non-suspended users that had already been created (and not
    yet deleted) as of `bucket_end_ms`. `bucket_end_ms` is capped at
    `now_ms` so the current/last bucket reflects "as of now" rather than
    the theoretical end of an in-progress period."""
    # Sort the eligible (non-suspended, dated) users by `created_at` once so
    # each bucket can *admit* newly-created users via a sweep pointer instead
    # of rescanning the whole list. This turns the created-side filter from
    # O(buckets x users) into a single pass; the only per-bucket work left is
    # dropping users whose deletion has passed (cheap unless there are many
    # deletions). Buckets are yielded in increasing time order, and
    # `created_at` only ever admits (never removes) as time advances, so the
    # sweep is safe.
    dated = sorted(
        (u for u in users if not u.suspended and u.created_at is not None),
        key=lambda u: u.created_at,
    )
    if not dated:
        return
    earliest_ms = dated[0].created_at
    if earliest_ms > now_ms:
        earliest_ms = now_ms

    admitted: "list[UserRecord]" = []
    ptr = 0
    n = len(dated)
    for key in _bucket_keys(earliest_ms, now_ms, units):
        next_key = get_next_time_key(key, units)
        next_start_ms = _dt_to_ms(parse_time_key(next_key, units))
        bucket_end_ms = min(next_start_ms - 1, now_ms)
        while ptr < n and dated[ptr].created_at <= bucket_end_ms:
            admitted.append(dated[ptr])
            ptr += 1
        existing = [
            u for u in admitted if u.deleted_at is None or u.deleted_at >= bucket_end_ms
        ]
        yield key, bucket_end_ms, existing


def active_series(
    users: "list[UserRecord]", units: str, now_ms: int, active_window_days: int
) -> "list[dict]":
    """One point per time bucket from the earliest `created_at` to `now_ms`:
    `total` = non-suspended users existing (created and not-yet-deleted) as
    of that bucket's end; `active` = the subset of those whose
    `last_used_at` falls within a rolling `active_window_days` window
    ending at that bucket's end. Returns `[]` if there are no users with a
    known `created_at` (degrade, never crash)."""
    window_ms = active_window_days * 86400 * 1000
    points = []
    for key, bucket_end_ms, existing in _iter_buckets(users, units, now_ms):
        total = len(existing)
        active = sum(
            1
            for u in existing
            if _within_window(u.last_used_at, bucket_end_ms, window_ms)
        )
        points.append({"key": key, "values": {"total": total, "active": active}})
    return points


def capability_series(
    users: "list[UserRecord]", units: str, now_ms: int, active_window_days: int
) -> "list[dict] | None":
    """Like `active_series`, but per-capability: `em` = users active on EM
    (`em_last_used_at` within the rolling window as of bucket-end); `opik`
    = same for `opik_last_used_at`. Returns `None` when NO user in `users`
    has either capability timestamp set (nothing to chart)."""
    if not any(
        u.em_last_used_at is not None or u.opik_last_used_at is not None for u in users
    ):
        return None

    window_ms = active_window_days * 86400 * 1000
    points = []
    for key, bucket_end_ms, existing in _iter_buckets(users, units, now_ms):
        em_active = sum(
            1
            for u in existing
            if _within_window(u.em_last_used_at, bucket_end_ms, window_ms)
        )
        opik_active = sum(
            1
            for u in existing
            if _within_window(u.opik_last_used_at, bucket_end_ms, window_ms)
        )
        points.append({"key": key, "values": {"em": em_active, "opik": opik_active}})
    return points


def adoption_rate_series(
    users: "list[UserRecord]", units: str, now_ms: int, active_window_days: int
) -> "list[dict]":
    """Per-bucket adoption RATES (percentages), mirroring the client
    notebook's "Adoption Rates" panel. For each bucket: `total` =
    non-suspended users existing as of bucket-end; each rate is the share of
    that total active within the rolling `active_window_days` window ending at
    bucket-end -- `overall` from `last_used_at` (active on ANY platform), `em`
    from `em_last_used_at`, `opik` from `opik_last_used_at`. Rates are
    0-guarded when a bucket has no users.

    `overall` is always present; `em`/`opik` are OMITTED entirely when no user
    carries that capability timestamp (mirrors `capability_series`), so an
    absent capability is not charted as a misleading flat 0%. Returns `[]` when
    no user has a known `created_at` (degrade, never crash)."""
    has_em = any(u.em_last_used_at is not None for u in users)
    has_opik = any(u.opik_last_used_at is not None for u in users)
    window_ms = active_window_days * 86400 * 1000
    points = []
    for key, bucket_end_ms, existing in _iter_buckets(users, units, now_ms):
        total = len(existing)

        def rate(getter):
            if not total:
                return 0.0
            hits = sum(
                1
                for u in existing
                if _within_window(getter(u), bucket_end_ms, window_ms)
            )
            return round(hits / total * 100, 1)

        values = {"overall": rate(lambda u: u.last_used_at)}
        if has_em:
            values["em"] = rate(lambda u: u.em_last_used_at)
        if has_opik:
            values["opik"] = rate(lambda u: u.opik_last_used_at)
        points.append({"key": key, "values": values})
    return points


def workspace_active_series(
    users: "list[UserRecord]",
    units: str,
    now_ms: int,
    active_window_days: int,
    all_workspaces=None,
) -> "list[dict] | None":
    """Per-bucket total vs active WORKSPACES. A workspace is active in a bucket
    when at least one non-suspended member existing as of bucket-end was active
    within the rolling `active_window_days` window (the same activity rule used
    for active users). Membership comes from the chargeback reverse-map
    (`UserRecord.workspaces`).

    The per-bucket `total` grows over time: a workspace counts from the bucket
    in which its earliest member existed (using the same created/not-yet-deleted
    membership as `_iter_buckets`). This keeps the total line honest about
    history instead of flat, while still reaching the authoritative count in the
    final bucket.

    When `all_workspaces` (the authoritative chargeback workspace-name set) is
    given, only its *undatable* members -- workspaces that never appear in the
    membership reverse-map, e.g. zero-member workspaces the chargeback snapshot
    lists but whose members carry no `created_at` -- are seeded into every
    bucket. Those can't be placed on the timeline, so attributing them to all of
    history is the least-wrong choice and keeps the final bucket matching
    `workspace_active_stats(..., all_workspaces=...)` and the KPI. Returns `None`
    only when there is nothing to show (no membership AND no `all_workspaces`)."""
    all_ws = set(all_workspaces) if all_workspaces else set()
    datable_ws = {ws for u in users for ws in u.workspaces}
    # Workspaces we can never place on the timeline: seed them into every bucket.
    seed_ws = all_ws - datable_ws
    if not all_ws and not datable_ws:
        return None
    window_ms = active_window_days * 86400 * 1000
    points = []
    for key, bucket_end_ms, existing in _iter_buckets(users, units, now_ms):
        total_ws, active_ws = set(seed_ws), set()
        for u in existing:
            is_active_user = _within_window(u.last_used_at, bucket_end_ms, window_ms)
            for ws in u.workspaces:
                total_ws.add(ws)
                if is_active_user:
                    active_ws.add(ws)
        points.append(
            {"key": key, "values": {"total": len(total_ws), "active": len(active_ws)}}
        )
    return points


def churn_series(
    users: "list[UserRecord]", units: str, now_ms: int
) -> "list[dict] | None":
    """Per-bucket user churn: `added` = accounts created in the bucket,
    `deleted` = accounts whose deletion date falls in the bucket. Both are
    read straight from `created_at`/`deleted_at` (no snapshot set-diffing, so
    no reconstructed-membership bug). Caveat: only accounts still present in
    the chargeback snapshot are visible, so hard-deleted accounts are not
    counted -- `deleted` reflects soft-deletes only. Returns `None` when no
    user has a known `created_at`."""
    created = [u.created_at for u in users if u.created_at is not None]
    if not created:
        return None
    earliest_ms = min(created)
    if earliest_ms > now_ms:
        earliest_ms = now_ms

    added_counts: dict = {}
    deleted_counts: dict = {}
    for u in users:
        if u.created_at is not None:
            k = format_time_key(_ms_to_dt(u.created_at), units)
            added_counts[k] = added_counts.get(k, 0) + 1
        if u.deleted_at is not None:
            k = format_time_key(_ms_to_dt(u.deleted_at), units)
            deleted_counts[k] = deleted_counts.get(k, 0) + 1

    return [
        {
            "key": key,
            "values": {
                "added": added_counts.get(key, 0),
                "deleted": deleted_counts.get(key, 0),
            },
        }
        for key in _bucket_keys(earliest_ms, now_ms, units)
    ]


def workspace_churn_series(
    users: "list[UserRecord]", units: str, now_ms: int
) -> "list[dict] | None":
    """Per-bucket workspace churn (proxy, since chargeback has no direct
    workspace lifecycle). A workspace's `added` bucket is its earliest member
    `created_at`; its `deleted` bucket is the latest member `deleted_at`, but
    only when EVERY member has been deleted (best-effort). Returns `None` when
    no workspace membership is present."""
    ws_members: dict = {}
    for u in users:
        for ws in u.workspaces:
            ws_members.setdefault(ws, []).append(u)
    if not ws_members:
        return None

    added_counts: dict = {}
    deleted_counts: dict = {}
    earliest_ms = None
    for members in ws_members.values():
        created = [m.created_at for m in members if m.created_at]
        if not created:
            continue
        first_ms = min(created)
        earliest_ms = first_ms if earliest_ms is None else min(earliest_ms, first_ms)
        k = format_time_key(_ms_to_dt(first_ms), units)
        added_counts[k] = added_counts.get(k, 0) + 1
        deleted = [m.deleted_at for m in members]
        if deleted and all(d is not None for d in deleted):
            dk = format_time_key(_ms_to_dt(max(deleted)), units)
            deleted_counts[dk] = deleted_counts.get(dk, 0) + 1

    if earliest_ms is None:
        return None
    if earliest_ms > now_ms:
        earliest_ms = now_ms

    return [
        {
            "key": key,
            "values": {
                "added": added_counts.get(key, 0),
                "deleted": deleted_counts.get(key, 0),
            },
        }
        for key in _bucket_keys(earliest_ms, now_ms, units)
    ]


def _user_breakdown_series(
    users, units, now_ms, active_window_days, last_used_getter, counters
):
    """Shared per-bucket user-breakdown builder for the EM/Opik capability
    breakdowns. `active` counts existing users whose `last_used_getter(u)`
    falls within the rolling `active_window_days` window as of bucket-end; each
    `(name, predicate)` in `counters` adds a per-bucket count of existing users
    matching that predicate (snapshot cumulative, so a user is counted from the
    bucket they first exist). Returns `None` when no user shows any signal
    (a non-null activity timestamp or any counter predicate true)."""

    def has_signal(u):
        return last_used_getter(u) is not None or any(pred(u) for _, pred in counters)

    if not any(has_signal(u) for u in users):
        return None
    window_ms = active_window_days * 86400 * 1000
    points = []
    for key, bucket_end_ms, existing in _iter_buckets(users, units, now_ms):
        values = {
            "active": sum(
                1
                for u in existing
                if _within_window(last_used_getter(u), bucket_end_ms, window_ms)
            )
        }
        for name, pred in counters:
            values[name] = sum(1 for u in existing if pred(u))
        points.append({"key": key, "values": values})
    return points


def em_user_breakdown_series(
    users: "list[UserRecord]", units: str, now_ms: int, active_window_days: int
) -> "list[dict] | None":
    """Per-bucket EM user breakdown: `active` (`em_last_used_at` within the
    rolling window), `experimenters` (`experiment_count` > 0), `data_pushers`
    (`data_logged_mb` > 0). Experimenter / data-pusher counts use the snapshot
    cumulative totals, so a user is counted from the bucket they first exist
    (an approximation that matches the client notebook). Returns `None` when
    there is no EM signal at all."""
    return _user_breakdown_series(
        users,
        units,
        now_ms,
        active_window_days,
        last_used_getter=lambda u: u.em_last_used_at,
        counters=[
            ("experimenters", lambda u: (u.experiment_count or 0) > 0),
            ("data_pushers", lambda u: (u.data_logged_mb or 0) > 0),
        ],
    )


def opik_user_breakdown_series(
    users: "list[UserRecord]", units: str, now_ms: int, active_window_days: int
) -> "list[dict] | None":
    """Per-bucket Opik user breakdown: `active` (`opik_last_used_at` within the
    rolling window) and `span_producers` (`opik_span_count` > 0, snapshot
    cumulative). Returns `None` when there is no Opik signal at all."""
    return _user_breakdown_series(
        users,
        units,
        now_ms,
        active_window_days,
        last_used_getter=lambda u: u.opik_last_used_at,
        counters=[("span_producers", lambda u: (u.opik_span_count or 0) > 0)],
    )


@dataclasses.dataclass(frozen=True)
class WorkspaceRecord:
    """A workspace as reported by the chargeback report's `workspaces[]`."""

    name: str
    num_experiments: float
    data_mb: float
    num_projects: int
    members: "tuple"  # member usernames (immutable, matching frozen=True)


def parse_workspaces(chargeback: dict) -> "list[WorkspaceRecord]":
    """Parse `chargeback["workspaces"]` into `WorkspaceRecord`s. `projects` may
    be a list (its length is the project count) or already a number; both are
    tolerated. These are EM/experiment projects -- chargeback does not carry
    Opik projects or MPM models."""
    out = []
    for w in (chargeback or {}).get("workspaces") or []:
        projects = w.get("projects")
        if isinstance(projects, list):
            num_projects = len(projects)
        elif isinstance(projects, (int, float)):
            num_projects = int(projects)
        else:
            num_projects = 0
        members = tuple(
            m.get("userName") for m in (w.get("members") or []) if m.get("userName")
        )
        out.append(
            WorkspaceRecord(
                name=w.get("name"),
                num_experiments=w.get("numberOfExperiments") or 0,
                data_mb=w.get("totalSizeInMb") or 0.0,
                num_projects=num_projects,
                members=members,
            )
        )
    return out


def workspace_org_totals(workspaces: "list[WorkspaceRecord]") -> dict:
    """Org-wide totals across all workspaces (from chargeback)."""
    return {
        "workspaces": len(workspaces),
        "projects": sum(w.num_projects for w in workspaces),
        "experiments": sum(w.num_experiments for w in workspaces),
        "data_mb": sum(w.data_mb for w in workspaces),
    }


def platform_mix(
    workspaces: "list[WorkspaceRecord]", users: "list[UserRecord]"
) -> dict:
    """Classify each workspace by platform usage into EM-only / Opik-only /
    both / neither.

    EM = the workspace has experiments or projects (chargeback, reliable).
    Opik = a member has `opik_span_count > 0`. This is a PROXY: chargeback
    carries Opik usage only per-user, so a user active in several workspaces
    has their Opik usage attributed to all of them (possible false positives).
    MPM is absent from chargeback and is NOT classified here."""
    opik_ws = set()
    for u in users:
        if (u.opik_span_count or 0) > 0:
            opik_ws.update(u.workspaces)
    counts = {"em_only": 0, "opik_only": 0, "both": 0, "neither": 0}
    for w in workspaces:
        em = (w.num_experiments or 0) > 0 or w.num_projects > 0
        op = w.name in opik_ws
        if em and op:
            counts["both"] += 1
        elif em:
            counts["em_only"] += 1
        elif op:
            counts["opik_only"] += 1
        else:
            counts["neither"] += 1
    return counts
