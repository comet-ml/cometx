# -*- coding: utf-8 -*-
# ****************************************
#                              __
#   _________  ____ ___  ___  / /__  __
#  / ___/ __ \/ __ `__ \/ _ \/ __/ |/_/
# / /__/ /_/ / / / / / /  __/ /__>  <
# \___/\____/_/ /_/ /_/\___/\__/_/|_|
#
#
#  Copyright (c) 2022 Cometx Development
#      Team. All rights reserved.
# ****************************************

import argparse
import base64
import json
import unittest
from unittest.mock import MagicMock, call, patch

from cometx.cli.migrate_users import (
    COMET_CLOUD_URL,
    _add_member,
    _get_existing_workspaces,
    _resolve_server_url,
    migrate_users,
)


def _make_new_style_key(base_url):
    payload = json.dumps({"baseUrl": base_url}).encode()
    encoded = base64.b64encode(payload).decode().rstrip("=")
    return f"abc123*{encoded}"


class TestResolveServerUrl:
    def test_explicit_url_takes_priority(self):
        key = _make_new_style_key("https://encoded.example.com")
        assert _resolve_server_url(key, "https://explicit.example.com") == "https://explicit.example.com"

    def test_explicit_url_trailing_slash_stripped(self):
        assert _resolve_server_url("anykey", "https://example.com/") == "https://example.com"

    def test_new_style_key_decoded(self):
        key = _make_new_style_key("https://self-hosted.example.com")
        assert _resolve_server_url(key) == "https://self-hosted.example.com"

    def test_new_style_key_trailing_slash_stripped(self):
        key = _make_new_style_key("https://self-hosted.example.com/")
        assert _resolve_server_url(key) == "https://self-hosted.example.com"

    def test_old_style_key_defaults_to_cloud(self):
        assert _resolve_server_url("oldstylekey") == COMET_CLOUD_URL

    def test_malformed_new_style_key_defaults_to_cloud(self):
        assert _resolve_server_url("abc*notvalidbase64!!!") == COMET_CLOUD_URL

    def test_new_style_key_missing_base_url_defaults_to_cloud(self):
        payload = json.dumps({"other": "value"}).encode()
        encoded = base64.b64encode(payload).decode()
        key = f"abc*{encoded}"
        assert _resolve_server_url(key) == COMET_CLOUD_URL


class TestGetExistingWorkspaces:
    @patch("cometx.cli.migrate_users.requests.get")
    def test_flat_list_response(self, mock_get):
        mock_get.return_value.json.return_value = ["ws1", "ws2"]
        mock_get.return_value.raise_for_status = MagicMock()
        result = _get_existing_workspaces("https://example.com", {})
        assert result == {"ws1", "ws2"}

    @patch("cometx.cli.migrate_users.requests.get")
    def test_dict_list_response(self, mock_get):
        mock_get.return_value.json.return_value = [{"name": "ws1"}, {"name": "ws2"}]
        mock_get.return_value.raise_for_status = MagicMock()
        result = _get_existing_workspaces("https://example.com", {})
        assert result == {"ws1", "ws2"}


class TestAddMember:
    @patch("cometx.cli.migrate_users.requests.post")
    def test_success(self, mock_post):
        mock_post.return_value.status_code = 200
        status, error = _add_member("http://example.com/add", {}, "user@example.com", "ws1")
        assert status == "added"
        assert error is None

    @patch("cometx.cli.migrate_users.requests.post")
    def test_already_member(self, mock_post):
        mock_post.return_value.status_code = 400
        mock_post.return_value.json.return_value = {"msg": "already member of ws1"}
        status, error = _add_member("http://example.com/add", {}, "user@example.com", "ws1")
        assert status == "already_member"
        assert error is None

    @patch("cometx.cli.migrate_users.requests.post")
    def test_failure(self, mock_post):
        mock_post.return_value.status_code = 403
        mock_post.return_value.json.return_value = {"msg": "forbidden"}
        status, error = _add_member("http://example.com/add", {}, "user@example.com", "ws1")
        assert status == "failed"
        assert error["status"] == 403

    @patch("cometx.cli.migrate_users.requests.post")
    def test_request_exception(self, mock_post):
        import requests as req
        mock_post.side_effect = req.exceptions.ConnectionError("timeout")
        status, error = _add_member("http://example.com/add", {}, "user@example.com", "ws1")
        assert status == "failed"
        assert error["status"] == "exception"


class TestMigrateUsersDryRun:
    def _make_args(self, **kwargs):
        defaults = dict(
            api_key="dest-key",
            url=None,
            source_api_key="source-key",
            source_url=None,
            chargeback_report=None,
            create_workspaces=False,
            dry_run=True,
            failures_output="failures.json",
        )
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    @patch("cometx.cli.migrate_users._resolve_server_url", return_value="https://dest.example.com")
    @patch("cometx.cli.migrate_users._fetch_chargeback_report")
    @patch("builtins.print")
    def test_dry_run_no_api_calls(self, mock_print, mock_fetch, mock_resolve):
        mock_fetch.return_value = {
            "workspaces": [
                {
                    "name": "ws1",
                    "members": [
                        {"email": "alice@example.com", "userName": "alice"},
                        {"email": "bob@example.com", "userName": "bob"},
                    ],
                }
            ]
        }
        with patch("cometx.cli.migrate_users.requests.post") as mock_post:
            migrate_users(self._make_args())
            mock_post.assert_not_called()

    @patch("cometx.cli.migrate_users._resolve_server_url", return_value="https://dest.example.com")
    @patch("cometx.cli.migrate_users._fetch_chargeback_report")
    @patch("builtins.print")
    def test_dry_run_prints_per_user(self, mock_print, mock_fetch, mock_resolve):
        mock_fetch.return_value = {
            "workspaces": [
                {
                    "name": "ws1",
                    "members": [
                        {"email": "alice@example.com", "userName": "alice"},
                    ],
                }
            ]
        }
        migrate_users(self._make_args())
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "alice@example.com" in printed
        assert "ws1" in printed

    @patch("cometx.cli.migrate_users._resolve_server_url", return_value="https://dest.example.com")
    @patch("cometx.cli.migrate_users._fetch_chargeback_report")
    @patch("builtins.print")
    def test_dry_run_skips_no_email(self, mock_print, mock_fetch, mock_resolve):
        mock_fetch.return_value = {
            "workspaces": [
                {
                    "name": "ws1",
                    "members": [
                        {"email": None, "userName": "noemail"},
                    ],
                }
            ]
        }
        migrate_users(self._make_args())
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "noemail" in printed
        assert "no email" in printed

    def test_no_source_key_without_chargeback_report_exits(self):
        args = self._make_args(source_api_key=None, chargeback_report=None)
        with self.assertRaises(SystemExit):
            migrate_users(args)

    def assertRaises(self, exc):
        import contextlib

        @contextlib.contextmanager
        def ctx():
            try:
                yield
                raise AssertionError(f"{exc} was not raised")
            except exc:
                pass

        return ctx()


if __name__ == "__main__":
    unittest.main()
