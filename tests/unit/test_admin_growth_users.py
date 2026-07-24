#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit tests for cometx.cli.admin_growth_users (people layer parse +
adoption/leaderboard derivations)."""

NOW = 1_720_000_000_000  # fixed ms; ~2024-07, tests pass now_ms explicitly


def _cb():
    return {
        "workspaces": [
            {"name": "team-a", "members": [{"userName": "alice"}, {"userName": "bob"}]},
            {"name": "team-b", "members": [{"userName": "alice"}]},
        ],
        "users": {
            "licensedUsers": [
                {
                    "username": "alice",
                    "email": "a@x.com",
                    "createdAt": NOW - 100,
                    "lastUsedAt": NOW - 10,
                    "experimentCount": 40,
                    "dataLoggedMb": 100.0,
                    "opikSpanCount": 5000,
                    "emLastUsedAt": NOW - 10,
                    "opikLastUsedAt": NOW - 10,
                    "suspended": False,
                    "deletedAt": None,
                },
                {
                    "username": "bob",
                    "email": "b@x.com",
                    "createdAt": NOW - 100,
                    "lastUsedAt": NOW - (60 * 86400 * 1000),
                    "experimentCount": 1,
                    "dataLoggedMb": 2.0,
                    "opikSpanCount": 0,
                    "suspended": False,
                    "deletedAt": None,
                },
                {
                    "username": "carol",
                    "email": "c@x.com",
                    "createdAt": NOW - 100,
                    "lastUsedAt": NOW - 5,
                    "experimentCount": 0,
                    "dataLoggedMb": 0.0,
                    "opikSpanCount": 0,
                    "suspended": True,
                    "deletedAt": None,
                },
            ]
        },
    }


def test_parse_users_maps_workspaces_and_fields():
    from cometx.cli.admin_growth_users import parse_users

    users = {u.username: u for u in parse_users(_cb())}
    assert set(users["alice"].workspaces) == {"team-a", "team-b"}
    assert users["alice"].opik_span_count == 5000
    assert users["bob"].opik_last_used_at is None  # absent field -> None


def test_adoption_excludes_suspended_and_respects_window():
    from cometx.cli.admin_growth_users import adoption_stats, parse_users

    users = parse_users(_cb())
    stats = adoption_stats(users, now_ms=NOW, active_window_days=30)
    # total = alice + bob (carol suspended, excluded); active = alice only (bob 60d ago)
    assert stats["total"] == 2
    assert stats["active"] == 1
    assert stats["adoption_pct"] == 50.0


def test_top_and_bottom_users_by_em_score():
    from cometx.cli.admin_growth_users import bottom_users, parse_users, top_users

    users = parse_users(_cb())
    top = top_users(users, key="em_score", n=1)
    assert top[0].username == "alice"
    # bottom = active-aware: only users with em_score > 0, ascending -> bob before alice
    bot = bottom_users(users, key="em_score", n=1)
    assert bot[0].username == "bob"


def test_top_users_absent_span_field_excluded():
    from cometx.cli.admin_growth_users import parse_users, top_users

    # a report with NO opikSpanCount anywhere
    cb = {
        "workspaces": [],
        "users": {
            "licensedUsers": [
                {
                    "username": "x",
                    "email": "x@x",
                    "lastUsedAt": NOW,
                    "experimentCount": 1,
                }
            ]
        },
    }
    users = parse_users(cb)
    assert users[0].opik_span_count is None
    assert top_users(users, key="opik_span_count", n=5) == []  # nothing has the metric


def test_active_series_buckets_total_and_active():
    from cometx.cli.admin_growth_users import active_series, parse_users

    users = parse_users(_cb())
    pts = active_series(users, units="month", now_ms=NOW, active_window_days=30)
    assert pts and all("total" in p["values"] and "active" in p["values"] for p in pts)
    # last bucket total excludes suspended carol
    assert pts[-1]["values"]["total"] == 2


def test_capability_series_none_when_no_capability_fields():
    from cometx.cli.admin_growth_users import capability_series, parse_users

    cb = {
        "workspaces": [],
        "users": {
            "licensedUsers": [
                {"username": "x", "email": "x", "lastUsedAt": NOW, "createdAt": NOW - 1}
            ]
        },
    }
    assert (
        capability_series(
            parse_users(cb), units="month", now_ms=NOW, active_window_days=30
        )
        is None
    )


def test_classify_accounts_uses_service_account_set():
    from cometx.cli.admin_growth_users import classify_accounts, parse_users

    users = parse_users(_cb())
    out = classify_accounts(users, service_account_names={"bob"})
    assert out["service"]["experiments"] == 1  # bob
    assert (
        out["personal"]["experiments"] == 40
    )  # alice (carol suspended still counts data? no: include all non-deleted)


def test_classify_accounts_regex_fallback_labeled():
    from cometx.cli.admin_growth_users import classify_accounts, parse_users

    cb = {
        "workspaces": [],
        "users": {
            "licensedUsers": [
                {
                    "username": "svc-pipeline",
                    "email": "p@x",
                    "experimentCount": 3,
                    "lastUsedAt": 1,
                },
                {
                    "username": "dana",
                    "email": "d@x",
                    "experimentCount": 7,
                    "lastUsedAt": 1,
                },
            ]
        },
    }
    users = parse_users(cb)
    out = classify_accounts(users, service_account_names=None)
    assert out["service"]["experiments"] == 3
    assert out["personal"]["experiments"] == 7
    assert out["source"] == "heuristic"


def _classify_username(username, email=None):
    """Helper: classify a single synthetic user (heuristic path) and return
    "personal" or "service"."""
    from cometx.cli.admin_growth_users import classify_accounts, parse_users

    cb = {
        "workspaces": [],
        "users": {
            "licensedUsers": [
                {
                    "username": username,
                    "email": email or (username + "@x.com"),
                    "experimentCount": 1,
                    "lastUsedAt": 1,
                }
            ]
        },
    }
    users = parse_users(cb)
    out = classify_accounts(users, service_account_names=None)
    return "service" if out["service"]["experiments"] == 1 else "personal"


def test_classify_accounts_regex_does_not_mislabel_substring_matches():
    # "lisa-brown" contains "sa-"; "abbot-jones" contains "bot-". Neither is
    # a service-account naming convention -- both are plain personal users.
    assert _classify_username("lisa-brown") == "personal"
    assert _classify_username("abbot-jones") == "personal"


def test_classify_accounts_regex_segment_boundary():
    # "jane-service-accountant" merely CONTAINS the "-service-account"
    # segment mid-word (as a prefix of "accountant") -- must not be
    # mislabeled as a service account. "foo-service-account" is the real
    # naming convention and must still classify as service.
    assert _classify_username("jane-service-accountant") == "personal"
    assert _classify_username("foo-service-account") == "service"


def test_classify_accounts_regex_matches_anchored_prefix_tokens():
    assert _classify_username("sa-pipeline") == "service"
    assert _classify_username("svc-x") == "service"
    assert _classify_username("bot-runner") == "service"


def test_classify_accounts_regex_matches_sagemaker_domain_anchored():
    assert (
        _classify_username("dana", email="dana@sagemaker-integration.com") == "service"
    )
    assert (
        _classify_username("dana2", email="dana2@sagemaker-integration.com.evil.com")
        == "personal"
    )


def test_classify_accounts_regex_matches_pipeline_and_automated_tokens():
    # Added to match the client notebook's classifier: "pipeline-" prefix and
    # the "automated" substring both indicate a service account.
    assert _classify_username("pipeline-nightly") == "service"
    assert _classify_username("data-automated-job") == "service"
    # "automation" does NOT contain "automated" -- must stay personal (guards
    # against over-matching the new keyword token).
    assert _classify_username("automation") == "personal"


def test_adoption_rate_series_percentages():
    from cometx.cli.admin_growth_users import adoption_rate_series, parse_users

    pts = adoption_rate_series(
        parse_users(_cb()), units="month", now_ms=NOW, active_window_days=30
    )
    assert pts
    # Last bucket: non-suspended existing = alice + bob (carol suspended); of
    # those, only alice is active within 30d overall / on EM / on Opik.
    last = pts[-1]["values"]
    assert last["overall"] == 50.0
    assert last["em"] == 50.0
    assert last["opik"] == 50.0


def test_adoption_rate_series_empty_when_no_created_at():
    from cometx.cli.admin_growth_users import adoption_rate_series, parse_users

    cb = {
        "workspaces": [],
        "users": {
            "licensedUsers": [{"username": "x", "email": "x", "lastUsedAt": NOW}]
        },
    }
    assert (
        adoption_rate_series(
            parse_users(cb), units="month", now_ms=NOW, active_window_days=30
        )
        == []
    )


def test_workspace_active_series_total_and_active():
    from cometx.cli.admin_growth_users import parse_users, workspace_active_series

    pts = workspace_active_series(
        parse_users(_cb()), units="month", now_ms=NOW, active_window_days=30
    )
    assert pts
    last = pts[-1]["values"]
    assert last["total"] == 2  # team-a, team-b (from non-suspended alice/bob)
    assert last["active"] == 2  # active alice belongs to both -> both active
    # no membership anywhere -> None
    cb = {
        "workspaces": [],
        "users": {
            "licensedUsers": [
                {"username": "x", "email": "x", "createdAt": NOW - 1, "lastUsedAt": NOW}
            ]
        },
    }
    assert (
        workspace_active_series(
            parse_users(cb), units="month", now_ms=NOW, active_window_days=30
        )
        is None
    )


def test_churn_series_added_and_deleted():
    from cometx.cli.admin_growth_users import churn_series, parse_users

    month = 30 * 86400 * 1000
    cb = {
        "workspaces": [],
        "users": {
            "licensedUsers": [
                {
                    "username": "a",
                    "email": "a",
                    "createdAt": NOW - 3 * month,
                    "lastUsedAt": NOW,
                },
                {
                    "username": "b",
                    "email": "b",
                    "createdAt": NOW - 2 * month,
                    "lastUsedAt": NOW,
                    "deletedAt": NOW - month,
                },
            ]
        },
    }
    pts = churn_series(parse_users(cb), units="month", now_ms=NOW)
    assert pts
    assert sum(p["values"]["added"] for p in pts) == 2
    assert sum(p["values"]["deleted"] for p in pts) == 1


def test_em_user_breakdown_series():
    from cometx.cli.admin_growth_users import em_user_breakdown_series, parse_users

    last = em_user_breakdown_series(
        parse_users(_cb()), units="month", now_ms=NOW, active_window_days=30
    )[-1]["values"]
    assert last["experimenters"] == 2  # alice(40) + bob(1)
    assert last["data_pushers"] == 2  # alice(100) + bob(2)
    assert last["active"] == 1  # only alice has emLastUsedAt in window


def test_opik_user_breakdown_series():
    from cometx.cli.admin_growth_users import opik_user_breakdown_series, parse_users

    last = opik_user_breakdown_series(
        parse_users(_cb()), units="month", now_ms=NOW, active_window_days=30
    )[-1]["values"]
    assert last["span_producers"] == 1  # only alice has opikSpanCount > 0
    assert last["active"] == 1  # only alice has opikLastUsedAt in window


def test_workspace_churn_series_added():
    from cometx.cli.admin_growth_users import parse_users, workspace_churn_series

    pts = workspace_churn_series(parse_users(_cb()), units="month", now_ms=NOW)
    assert pts
    assert sum(p["values"]["added"] for p in pts) == 2  # team-a, team-b
    assert sum(p["values"]["deleted"] for p in pts) == 0  # no fully-deleted workspace


def test_workspace_active_stats():
    from cometx.cli.admin_growth_users import parse_users, workspace_active_stats

    stats = workspace_active_stats(
        parse_users(_cb()), now_ms=NOW, active_window_days=30
    )
    # team-a (alice/bob) + team-b (alice); carol suspended contributes nothing
    assert stats["total"] == 2
    # alice active covers both workspaces; bob inactive but alice keeps both active
    assert stats["active"] == 2
    assert stats["active_pct"] == 100.0


def test_parse_workspaces_org_totals_and_platform_mix():
    from cometx.cli.admin_growth_users import (
        parse_users,
        parse_workspaces,
        platform_mix,
        workspace_org_totals,
    )

    cb = {
        "workspaces": [
            {
                "name": "em-ws",
                "numberOfExperiments": 10,
                "totalSizeInMb": 5.0,
                "projects": [{}, {}],
                "members": [{"userName": "a"}],
            },
            {
                "name": "opik-ws",
                "numberOfExperiments": 0,
                "projects": [],
                "members": [{"userName": "b"}],
            },
            {
                "name": "both-ws",
                "numberOfExperiments": 3,
                "projects": [{}],
                "members": [{"userName": "c"}],
            },
            {
                "name": "empty-ws",
                "numberOfExperiments": 0,
                "projects": [],
                "members": [],
            },
        ],
        "users": {
            "licensedUsers": [
                {"username": "a", "email": "a", "opikSpanCount": 0, "lastUsedAt": 1},
                {"username": "b", "email": "b", "opikSpanCount": 500, "lastUsedAt": 1},
                {"username": "c", "email": "c", "opikSpanCount": 50, "lastUsedAt": 1},
            ]
        },
    }
    ws = parse_workspaces(cb)
    assert workspace_org_totals(ws) == {
        "workspaces": 4,
        "projects": 3,
        "experiments": 13,
        "data_mb": 5.0,
    }
    # em-ws=EM only; opik-ws=Opik only (b has spans); both-ws=both; empty-ws=neither
    assert platform_mix(ws, parse_users(cb)) == {
        "em_only": 1,
        "opik_only": 1,
        "both": 1,
        "neither": 1,
    }


def test_workspace_active_series_seeds_all_workspaces_incl_zero_member():
    # A workspace with zero datable members never appears in the membership
    # reverse-map, but seeding from all_workspaces keeps `total` aligned with
    # the KPI (workspace_active_stats(..., all_workspaces=...)).
    from cometx.cli.admin_growth_users import parse_users, workspace_active_series

    now = 1_720_000_000_000
    cb = {
        "workspaces": [
            {"name": "ws-a", "members": [{"userName": "alice"}]},
            {"name": "ws-empty", "members": []},  # zero members -> not in reverse-map
        ],
        "users": {
            "report": [
                {
                    "username": "alice",
                    "email": "a",
                    "createdAt": now - 10**10,
                    "lastUsedAt": now,
                },
            ]
        },
    }
    users = parse_users(cb)
    pts = workspace_active_series(
        users, "month", now, 60, all_workspaces={"ws-a", "ws-empty"}
    )
    assert pts is not None
    # every bucket's total includes the zero-member workspace (seeded = 2)
    assert all(p["values"]["total"] == 2 for p in pts)
    # without the seed, ws-empty would be missing -> total 1
    pts_noseed = workspace_active_series(users, "month", now, 60)
    assert all(p["values"]["total"] == 1 for p in pts_noseed)
