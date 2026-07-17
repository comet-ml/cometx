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
