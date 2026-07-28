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
    assert (
        called_url
        == "https://comet.example.com/api/admin/chargeback/report?reportMonth=2026-06"
    )


@pytest.mark.parametrize(
    "base",
    [
        "http://comet.example.com",  # not https
        "ftp://comet.example.com",  # wrong scheme
        "comet.example.com",  # no scheme/netloc
        "https://",  # empty netloc
        "",  # empty
    ],
)
def test_fetch_chargeback_rejects_non_https_base(base):
    from cometx.utils import fetch_chargeback_report

    api = MagicMock()
    api.config = {"comet.url_override": base}
    api.api_key = "KEY"

    with pytest.raises(ValueError):
        fetch_chargeback_report(api)
    api._client.get.assert_not_called()


def test_fetch_chargeback_rejects_non_https_host_override():
    from cometx.utils import fetch_chargeback_report

    api = MagicMock()
    api.config = {"comet.url_override": "https://comet.example.com"}
    api.api_key = "KEY"

    with pytest.raises(ValueError):
        fetch_chargeback_report(api, host="http://127.0.0.1")
    api._client.get.assert_not_called()
