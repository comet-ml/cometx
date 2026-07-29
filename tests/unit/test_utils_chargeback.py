from unittest.mock import MagicMock

import pytest


def test_fetch_chargeback_builds_url_and_returns_json():
    from cometx.utils import fetch_chargeback_report

    api = MagicMock()
    api.config = {"comet.url_override": "https://comet.example.com/"}
    api.api_key = "KEY"
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"workspaces": [], "users": {}}
    api._client.get.return_value = resp

    out = fetch_chargeback_report(api)

    assert out == {"workspaces": [], "users": {}}
    called_url = api._client.get.call_args[0][0]
    assert called_url == "https://comet.example.com/api/admin/chargeback/report"


def test_fetch_chargeback_appends_report_month():
    from cometx.utils import fetch_chargeback_report

    api = MagicMock()
    api.config = {"comet.url_override": "https://comet.example.com"}
    api.api_key = "KEY"
    resp = MagicMock(status_code=200)
    resp.json.return_value = {}
    api._client.get.return_value = resp

    fetch_chargeback_report(api, report_month="2026-06")

    called_url = api._client.get.call_args[0][0]
    assert called_url == "https://comet.example.com/api/admin/chargeback/report"
    # reportMonth is now passed as a query param (URL-encoded by the client),
    # not interpolated into the URL string.
    assert api._client.get.call_args.kwargs["params"] == {"reportMonth": "2026-06"}


def test_fetch_chargeback_allows_http_on_prem_base():
    # On-prem Comet servers are reached over plain http (documented in
    # MIGRATIONS.md / README.md); http must not be rejected.
    from cometx.utils import fetch_chargeback_report

    api = MagicMock()
    api.config = {"comet.url_override": "http://comet.internal.corp"}
    api.api_key = "KEY"
    resp = MagicMock(status_code=200)
    resp.json.return_value = {}
    api._client.get.return_value = resp

    fetch_chargeback_report(api)

    called_url = api._client.get.call_args[0][0]
    assert called_url == "http://comet.internal.corp/api/admin/chargeback/report"


def test_fetch_chargeback_preserves_path_prefix():
    # A base with a path prefix (e.g. /clientlib) must keep that prefix
    # rather than silently dropping it.
    from cometx.utils import fetch_chargeback_report

    api = MagicMock()
    api.config = {"comet.url_override": "https://comet.x.com/clientlib/"}
    api.api_key = "KEY"
    resp = MagicMock(status_code=200)
    resp.json.return_value = {}
    api._client.get.return_value = resp

    fetch_chargeback_report(api)

    called_url = api._client.get.call_args[0][0]
    assert called_url == ("https://comet.x.com/clientlib/api/admin/chargeback/report")


@pytest.mark.parametrize(
    "base",
    [
        "ftp://comet.example.com",  # non-http(s) scheme
        "comet.example.com",  # no scheme/netloc
        "https://",  # empty netloc
        "",  # empty
    ],
)
def test_fetch_chargeback_rejects_malformed_base(base):
    from cometx.utils import fetch_chargeback_report

    api = MagicMock()
    api.config = {"comet.url_override": base}
    api.api_key = "KEY"

    with pytest.raises(ValueError):
        fetch_chargeback_report(api)
    api._client.get.assert_not_called()


def test_fetch_chargeback_rejects_malformed_host_override():
    from cometx.utils import fetch_chargeback_report

    api = MagicMock()
    api.config = {"comet.url_override": "https://comet.example.com"}
    api.api_key = "KEY"

    with pytest.raises(ValueError):
        fetch_chargeback_report(api, host="not-a-url")
    api._client.get.assert_not_called()
