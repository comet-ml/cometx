from types import SimpleNamespace
from unittest import mock

import comet_ml
import pytest
from comet_ml.exceptions import CometRestApiException

from cometx.api import API
from cometx.views import Section, build_experiment_view, build_view, line_panel


def _api():
    """A cometx API with a mock client, made without network access."""
    api = object.__new__(API)
    api._client = mock.MagicMock()
    api.get_project = mock.MagicMock(return_value={"projectId": "proj-1"})
    return api


def _rest_error(message):
    response = mock.MagicMock(status_code=400, url="http://x")
    response.json.return_value = {"msg": message, "code": 400}
    return CometRestApiException("POST", response)


def _project_view():
    return build_view("Dash", [Section("S", [line_panel("loss")])])


def _experiment_view():
    return build_experiment_view("Run", [Section("S", [line_panel("loss")])])


def test_create_view__experiment_layout__raises_before_saving():
    api = _api()

    with pytest.raises(ValueError, match="build_view"):
        api.create_view("ws", "proj", _experiment_view())


def test_create_view__name_taken__value_error_points_to_update_view():
    api = _api()
    with mock.patch.object(
        comet_ml.API, "create_view", side_effect=_rest_error("View name already used.")
    ):
        with pytest.raises(ValueError, match="update_view"):
            api.create_view("ws", "proj", _project_view())


def test_create_view__other_rest_error__reraised():
    api = _api()
    with mock.patch.object(
        comet_ml.API, "create_view", side_effect=_rest_error("boom")
    ):
        with pytest.raises(CometRestApiException):
            api.create_view("ws", "proj", _project_view())


def test_update_view__experiment_layout__raises():
    view = _experiment_view()
    view.template_id = "tmpl"

    with pytest.raises(ValueError, match="build_view"):
        _api().update_view("ws", "proj", view)


def test_create_experiment_view__project_layout__raises():
    experiment = mock.MagicMock()

    with pytest.raises(ValueError, match="build_experiment_view"):
        _api().create_experiment_view(experiment, _project_view())
    experiment.create_view.assert_not_called()


def test_create_experiment_view__name_taken__value_error():
    experiment = mock.MagicMock()
    experiment.create_view.side_effect = _rest_error("View name already used.")

    with pytest.raises(ValueError, match="include_workspace_project_views"):
        _api().create_experiment_view(experiment, _experiment_view())


def test_update_experiment_view__older_comet_ml__upserts_with_template_id():
    api = _api()
    api._client.upsert_chart_template_view.return_value = {"templateName": "Run"}
    experiment = SimpleNamespace(project_id="proj-1")  # no update_view
    view = _experiment_view()
    view.template_id = "tmpl"

    api.update_experiment_view(experiment, view)

    project_id, upserted = api._client.upsert_chart_template_view.call_args[0]
    assert project_id == "proj-1"
    assert upserted.template_id == "tmpl"


def test_update_experiment_view__no_template_id__raises():
    with pytest.raises(ValueError, match="template_id"):
        _api().update_experiment_view(
            SimpleNamespace(project_id="p"), _experiment_view()
        )


def test_create_dashboard__deprecated__warns_and_still_creates():
    api = _api()
    api._get_project_id = mock.MagicMock(return_value="proj-1")

    with pytest.warns(DeprecationWarning, match="create_view"):
        api.create_dashboard("ws", "proj", "Dash", panels=["t1"])

    endpoint = api._client.post_from_endpoint.call_args[0][0]
    payload = api._client.post_from_endpoint.call_args[1]["payload"]
    assert endpoint == "write/dashboard-template/create"
    assert payload["codePanelTemplateIds"] == ["t1"]


def test_update_dashboard__deprecated__warns_and_still_posts():
    api = _api()
    api._get_project_id = mock.MagicMock(return_value="proj-1")

    with pytest.warns(DeprecationWarning, match="update_view"):
        api.update_dashboard("ws", "proj", "tmpl", template_name="New")

    payload = api._client.post_from_endpoint.call_args[1]["payload"]
    assert payload["template_id"] == "tmpl"
    assert payload["template_name"] == "New"
