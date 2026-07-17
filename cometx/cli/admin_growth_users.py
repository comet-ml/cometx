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
import re

from cometx.utils import format_time_key, get_next_time_key, parse_time_key


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


def _metric_value(user: UserRecord, key: str):
    """Return the raw metric value for `key`, or None if the metric is
    absent for this user (only possible for opik_span_count)."""
    if key == "em_score":
        return user.experiment_count + user.data_logged_mb
    if key == "opik_span_count":
        return user.opik_span_count
    raise ValueError("Unsupported key: {}".format(key))


def top_users(users, key: str, n: int) -> "list[UserRecord]":
    """Top `n` users by `key` (descending), skipping users for whom the
    metric is absent (None)."""
    scored = [(u, _metric_value(u, key)) for u in users]
    scored = [(u, v) for u, v in scored if v is not None]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [u for u, _ in scored[:n]]


def bottom_users(users, key: str, n: int) -> "list[UserRecord]":
    """Bottom `n` users by `key` (ascending), active-aware: only considers
    users whose metric value is strictly positive."""
    scored = [(u, _metric_value(u, key)) for u in users]
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
    r"(?:^|[._-])(?:svc|sa|bot)-", re.IGNORECASE
)
# Segment token: matched as-is against the username (already has a leading
# hyphen boundary, so no separate anchoring is needed).
_SERVICE_ACCOUNT_SEGMENT_PATTERN = re.compile(r"-service-account", re.IGNORECASE)
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
    set (by `username`) decides the split, and `source` is reported as
    "admin_api". When it is `None` (the endpoint failed, was disabled, or
    isn't present in this deployment), falls back to a labeled regex
    heuristic (`_looks_like_service_account`) and reports `source` as
    "heuristic" so callers can surface which method produced the split.
    """
    if service_account_names is not None:
        source = "admin_api"

        def is_service(user: UserRecord) -> bool:
            return user.username in service_account_names

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

    return {"personal": totals["personal"], "service": totals["service"], "source": source}


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
    while key != end_key and guard < 100_000:
        key = get_next_time_key(key, units)
        keys.append(key)
        guard += 1
    return keys


def _iter_buckets(users: "list[UserRecord]", units: str, now_ms: int):
    """Yield `(key, bucket_end_ms, existing)` for each zero-filled bucket
    from the earliest `created_at` through `now_ms`. `existing` is the
    list of non-suspended users that had already been created (and not
    yet deleted) as of `bucket_end_ms`. `bucket_end_ms` is capped at
    `now_ms` so the current/last bucket reflects "as of now" rather than
    the theoretical end of an in-progress period."""
    created_times = [u.created_at for u in users if u.created_at is not None]
    if not created_times:
        return
    earliest_ms = min(created_times)
    if earliest_ms > now_ms:
        earliest_ms = now_ms

    for key in _bucket_keys(earliest_ms, now_ms, units):
        next_key = get_next_time_key(key, units)
        next_start_ms = _dt_to_ms(parse_time_key(next_key, units))
        bucket_end_ms = min(next_start_ms - 1, now_ms)
        existing = [
            u
            for u in users
            if not u.suspended
            and u.created_at is not None
            and u.created_at <= bucket_end_ms
            and (u.deleted_at is None or u.deleted_at >= bucket_end_ms)
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
            1 for u in existing if _within_window(u.last_used_at, bucket_end_ms, window_ms)
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
