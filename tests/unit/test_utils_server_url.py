"""Unit tests for cometx.utils.validate_server_base -- the single shared
server-URL rule used by --url/--source-url, the migrate-users request
boundary, and admin_api_url."""

import pytest

from cometx.utils import (
    InvalidServerURLError,
    redact_url_userinfo,
    validate_server_base,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        # credentials replaced, host/path kept
        ("https://admin:s3cr3t@comet.internal", "https://***@comet.internal"),
        ("http://user@host/clientlib", "http://***@host/clientlib"),
        # quoted mid-string, as SDK/HTTP exceptions do
        (
            "failed: GET https://u:p@host/api/x returned 403",
            "failed: GET https://***@host/api/x returned 403",
        ),
        # scheme-less base
        ("admin:s3cr3t@comet.internal", "***@comet.internal"),
        # an @ in a path is not credentials
        ("https://comet.internal/a@b", "https://comet.internal/a@b"),
        # untouched when there is nothing to redact
        ("https://comet.internal/api", "https://comet.internal/api"),
        ("", ""),
    ],
)
def test_redact_url_userinfo(raw, expected):
    assert redact_url_userinfo(raw) == expected


def test_redact_url_userinfo_passes_non_strings_through():
    assert redact_url_userinfo(None) is None


def test_validation_error_redacts_credentials():
    # The error is printed to the terminal / logs, so it must not echo a
    # password back from the operator-supplied base.
    with pytest.raises(InvalidServerURLError) as exc_info:
        validate_server_base("ftp://admin:s3cr3t@comet.internal")
    assert "s3cr3t" not in str(exc_info.value)
    assert "***@comet.internal" in str(exc_info.value)


@pytest.mark.parametrize(
    "url",
    [
        "https://comet.example.com",
        "http://comet.internal.corp",  # on-prem plain http
        "https://comet.x.com/clientlib/",  # path prefix
        "http://localhost:8080",
    ],
)
def test_accepts_http_and_https_with_a_host(url):
    assert validate_server_base(url).netloc


@pytest.mark.parametrize(
    "url",
    [
        "ftp://comet.example.com",  # non-http(s) scheme
        "comet.example.com",  # no scheme/netloc
        "https://",  # empty netloc
        "",  # empty
    ],
)
def test_rejects_malformed(url):
    with pytest.raises(InvalidServerURLError):
        validate_server_base(url)


def test_label_names_the_offending_input():
    with pytest.raises(InvalidServerURLError) as exc_info:
        validate_server_base("nope", label="--url/--source-url")
    assert "--url/--source-url" in str(exc_info.value)


def test_all_three_call_sites_share_the_rule():
    # The point of the shared validator: one base can't be accepted by one
    # entry point and rejected by another.
    from cometx.cli.migrate_users import _RequestsClient, _resolve_server_url
    from cometx.utils import admin_api_url

    on_prem = "http://comet.internal.corp"
    assert _resolve_server_url("anykey", on_prem) == on_prem
    assert admin_api_url(on_prem, "/api/admin/chargeback/report").startswith(on_prem)

    with pytest.raises(InvalidServerURLError):
        admin_api_url("ftp://x.com", "/api/admin/chargeback/report")
    with pytest.raises(InvalidServerURLError):
        _RequestsClient().get("ftp://x.com")
    with pytest.raises(SystemExit):  # the CLI layer converts it to exit(1)
        _resolve_server_url("anykey", "ftp://x.com")
