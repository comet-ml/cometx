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


def is_active(user: UserRecord, now_ms: int, active_window_days: int) -> bool:
    """True if `user.last_used_at` falls within the active window ending at
    `now_ms` (inclusive of both bounds)."""
    window_ms = active_window_days * 86400 * 1000
    return (now_ms - window_ms) <= user.last_used_at <= now_ms


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
