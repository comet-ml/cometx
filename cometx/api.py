# -*- coding: utf-8 -*-
# ****************************************
#                              __
#   _________  ____ ___  ___  / /__  __
#  / ___/ __ \/ __ `__ \/ _ \/ __/ |/_/
# / /__/ /_/ / / / / / /  __/ /__>  <
# \___/\____/_/ /_/ /_/\___/\__/_/|_|
#
#
#  Copyright (c) 2023 Cometx Development
#      Team. All rights reserved.
# ****************************************

import os
import tempfile
import warnings
from typing import Any, Dict, List
from urllib.parse import urlparse

import requests
from comet_ml import API
from comet_ml.api_objects.view import View
from comet_ml.exceptions import CometRestApiException

from .panel_utils import create_panel_zip

_BUILDERS = {
    "project": "cometx.views.build_view",
    "experiment": "cometx.views.build_experiment_view",
}


def _layout_scope(view: View) -> Any:
    """Which scope's envelope holds the view's sections: "project" (v3),
    "experiment" (v2["panels"]), or None when it has no sections."""
    if isinstance(view.v3, dict) and view.v3.get("sections"):
        return "project"
    panels = view.v2.get("panels") if isinstance(view.v2, dict) else None
    if isinstance(panels, dict) and panels.get("sections"):
        return "experiment"
    return None


def _check_scope(view: View, expected: str) -> None:
    """Raise if the view's sections are in the other scope's envelope, which
    Comet would save but show with no panels."""
    actual = _layout_scope(view)
    if actual is not None and actual != expected:
        raise ValueError(
            "view %r has %s-scope sections, which show no panels in %s scope;"
            " build it with %s() instead"
            % (view.name, actual, expected, _BUILDERS[expected])
        )


class API(API):
    def get_panels(self, workspace: str) -> List[Dict[str, Any]]:
        """
        Get the metadata for all panels in a workspace.

        Args:
            workspace (str): the name of the workspace

        Returns: a list of dictionaries representing the panel
            metadata.

        Example:
        ```python linenums="1"
        from cometx import API

        api = API()
        panels = api.get_panels("my-workspace-name")
        ```
        The structure of a panel metadata:
        ```json
        {
         'templateId': '1234',
         'owner': 'OWNER',
         'teamId': 'OWNER-default',
         'templateName': 'PANEL NAME',
         'queryBuilderId': '',
         'scopeType': 'PRIVATE',
         'revisionId': 1721739799515,
         'createdAt': 1721739799581
        }
        ```
        """
        results = self._client.get_from_endpoint(
            "code-panel/get-all", {"workspace": workspace}
        )
        return results["codePanelTemplateRows"]

    def get_panel(self, panel_id: str) -> Dict[str, Any]:
        """
        Get the panel data given the panel's ID.

        Args:
            panel_id (str): the panel id (also called templateId)

        Returns: a dictionary of panel data

        Example:
        ```python linenums="1"
        from cometx import API

        api = API()
        panel = api.get_panel("1234")
        ```
        """
        results = self._client.get_from_endpoint(
            "code-panel/download", {"templateId": panel_id}
        )
        return results

    def get_panel_code(self, panel_id: str) -> str:
        """
        Given a panel ID, return the active code (may be JavaScript or Python).

        Args:
            panel_id (str): the panel id (also called templateId)

        Example:
        ```python linenums="1"
        from cometx import API

        api = API()
        panels = api.get_panel_code("1234")
        ```
        """
        metadata = self.get_panel(panel_id)
        if metadata["code"]["type"] == "py":
            return metadata["code"]["pyCode"]
        else:
            return metadata["code"]["code"]

    def download_panel_zip(self, panel_id, filename=None):
        """
        Given a panel ID, download the associationed panel as a zip file.

        Args:
            panel_id (str): the panel ID
            filename (str): optional, the path/filename of where to save
                the panel

        Example:
        ```python linenums="1"
        from cometx import API

        api = API()
        filename = api.download_panel_zip("1234")
        ```
        """
        results = self._client.get_from_endpoint(
            f"template/{panel_id}/download", {}, return_type="binary"
        )
        filename = filename if filename else f"panel-{panel_id}.zip"
        with open(filename, "wb") as fp:
            fp.write(results)
        return filename

    def upload_panel_name(self, name):
        # Upload well-known panel names
        workspace = self.get_default_workspace()
        url = f"https://raw.githubusercontent.com/comet-ml/comet-examples/master/panels/{name}/{name}.py"
        return self.upload_panel_url(workspace, url)

    def upload_panel(self, workspace=None, name=None):
        # Upload well-known panel names
        if name is None:
            raise Exception("Missing panel name to api.upload_panel(name='Name')")
        workspace = workspace if workspace is not None else self.get_default_workspace()
        url = f"https://raw.githubusercontent.com/comet-ml/comet-examples/master/panels/{name}/{name}.py"
        return self.upload_panel_url(workspace, url)

    def upload_panel_url(self, workspace, item):
        """
        Upload a panel from a URL.
        """
        # TODO:
        # https://github.com/comet-ml/comet-examples/blob/master/panels/TensorboardProfileViewer.py
        # turns into:
        # https://raw.githubusercontent.com/comet-ml/comet-examples/master/panels/TensorboardProfileViewer.py
        # Does work with private repo (raw) with token::
        # https://raw.githubusercontent.com/comet-ml/snakebite-custom-solutions/main/panels/AverageTwoMetrics.py?token=XXXX
        print("Downloading %r..." % item)
        response = requests.get(item)
        parsed_url = urlparse(item)
        if parsed_url.path.endswith(".py"):
            code = response.content.decode()
            print("   Creating zipped code...")
            nice_name = os.path.splitext(os.path.basename(parsed_url.path))[0]
            filename = create_panel_zip(nice_name, code)
            print("   Uploading panel %r..." % nice_name)
            self.upload_panel_zip(workspace, filename)
        elif parsed_url.path.endswith(".zip"):
            zipcontents = response.content
            print("    Saving zip file...")
            with tempfile.NamedTemporaryFile(suffix=".zip") as fp:
                fp.write(zipcontents)
                print("    Uploading panel...")
                self.upload_panel_zip(workspace, fp.name)
        else:
            raise Exception("I don't know what to do with %r" % parsed_url.path)

    def upload_panel_code(
        self,
        workspace: str,
        panel_name: str,
        code: str,
        template_id: str = None,
    ) -> Dict[str, Any]:
        """
        Upload Python code as a panel in a workspace.

        Args:
            workspace (str): the workspace to place the panel into
            panel_name (str): the name of the panel
            code (str): the code to turn into a panel
            template_id (str, optional): if provided, overwrites the existing
                panel with this ID instead of creating a new one

        Example:
        ```python linenums="1"
        from cometx.api import API

        api = API()

        code = '''
        from comet_ml import ui
        ui.display("Hello, world, from a script!")
        '''
        api.upload_panel_code("my-workspace", "My Python Script", code)
        ```
        """
        filename = create_panel_zip(panel_name, code)
        return self.upload_panel_zip(workspace, filename, template_id=template_id)

    def upload_panel_zip(
        self, workspace: str, filename: str, template_id: str = None
    ) -> Dict[str, str]:
        """
        Upload a panel zip file to a workspace.

        Args:
            workspace (str): the workspace to place the panel into
            filename (str): the name of the panel zip to upload
            template_id (str, optional): if provided, overwrites the existing
                panel with this ID instead of creating a new one

        Returns: dictionary of results

        Example:
        ```python linenums="1"
        from cometx import API

        api = API()
        panels = api.upload_panel_zip("my-workspace", "panel-1234.zip")
        ```
        """
        params = {"teamName": workspace}
        if template_id is not None:
            params["templateId"] = template_id
        payload = {}
        with open(filename, "rb") as fp:
            files = {"file": (filename, fp)}
            results = self._client.post_from_endpoint(
                "write/template/upload",
                payload=payload,
                params=params,
                files=files,
            )
        return results.json()

    def _get_project_id(self, workspace: str, project_name: str) -> str:
        projects = self._client.get_from_endpoint(
            "projects", params={"workspaceName": workspace}
        )
        project = next(
            (p for p in projects["projects"] if p["projectName"] == project_name),
            None,
        )
        if project is None:
            raise Exception(
                f"Project {project_name!r} not found in workspace {workspace!r}"
            )
        return project["projectId"]

    def create_dashboard(
        self,
        workspace: str,
        project_name: str,
        template_name: str,
        template_id: str = None,
        panels: List[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Deprecated: use `create_view` with a layout from
        `cometx.views.build_view`, which places panels in the view.

        Creates a view in a project. Panels listed in `panels` are not shown
        on it: the new view opens with no panels.

        Args:
            workspace (str): the workspace name the project belongs to
            project_name (str): the project name to create the view in
            template_name (str): name for the new view
            template_id (str, optional): source view ID to clone code
                panel associations from
            panels (list, optional): panel template IDs
            **kwargs: additional fields for the view

        Returns: dictionary representing the created view
        """
        warnings.warn(
            "create_dashboard() is deprecated; use create_view() with "
            "cometx.views.build_view()",
            DeprecationWarning,
            stacklevel=2,
        )
        project_id = self._get_project_id(workspace, project_name)

        body = {"project_id": project_id, "template_name": template_name, **kwargs}
        if template_id is not None:
            body["template_id"] = template_id
        if panels is not None:
            body["codePanelTemplateIds"] = panels
        results = self._client.post_from_endpoint(
            "write/dashboard-template/create",
            payload=body,
        )
        return results.json()

    def update_dashboard(
        self,
        workspace: str,
        project_name: str,
        template_id: str,
        template_name: str = None,
        panels: List[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Deprecated: use `update_view`, which saves a view's layout.

        Calls an endpoint that only Comet's web app can use; with an API key
        the server answers 404.

        Args:
            workspace (str): the workspace name the project belongs to
            project_name (str): the project name the view belongs to
            template_id (str): the ID of the view to update
            template_name (str, optional): new name for the view
            panels (list, optional): panel template IDs
            **kwargs: additional fields for the view

        Returns: dictionary representing the updated view
        """
        warnings.warn(
            "update_dashboard() is deprecated; use update_view()",
            DeprecationWarning,
            stacklevel=2,
        )
        project_id = self._get_project_id(workspace, project_name)

        body = {"project_id": project_id, "template_id": template_id, **kwargs}
        if template_name is not None:
            body["template_name"] = template_name
        if panels is not None:
            body["codePanelTemplateIds"] = panels
        results = self._client.post_from_endpoint(
            "dashboard-templates/project/upsert",
            payload=body,
        )
        return results.json()

    def create_view(self, workspace: str, project_name: str, view: View) -> Any:
        """
        Create a project-level view.

        Like comet_ml's `create_view`, on any comet_ml version: a view built
        with `build_experiment_view` raises a ValueError (it would show no
        panels), as does a name the project already uses.

        Args:
            workspace (str): the workspace name
            project_name (str): the project name
            view (View): from `cometx.views.build_view`

        Returns: the created View, or None on failure
        """
        _check_scope(view, "project")
        try:
            return super().create_view(workspace, project_name, view)
        except CometRestApiException as error:
            if "View name already used" not in str(error):
                raise
            raise ValueError(
                "Project %s/%s already has a view named %r. Use another name, "
                "or set that view's template_id on this one and call "
                "update_view() to replace it." % (workspace, project_name, view.name)
            ) from error

    def update_view(self, workspace: str, project_name: str, view: View) -> Any:
        """
        Overwrite an existing project-level view in place.

        Unlike `create_view`, this keeps the view's `template_id`, so the saved
        view is replaced rather than a new one added. Uses comet_ml's own
        `update_view` where the installed version has it.

        Args:
            workspace (str): the workspace name
            project_name (str): the project name
            view (View): a View with `template_id` set, typically one returned
                by `get_views()` and then modified

        Returns: the updated View, or None on failure
        """
        _check_scope(view, "project")
        base_update = getattr(super(), "update_view", None)
        if base_update is not None:
            return base_update(workspace, project_name, view)

        if not view.template_id:
            raise ValueError(
                "update_view requires a view with template_id set; "
                "use create_view to add a new view"
            )
        project = self.get_project(workspace, project_name)
        if not project:
            raise ValueError("unknown project %s/%s" % (workspace, project_name))
        result = self._client.upsert_view(project["projectId"], view)
        return View.from_payload_dict(result) if result else None

    def create_experiment_view(self, experiment, view: View) -> Any:
        """
        Create an experiment view, shared by every experiment in its project.

        Like `APIExperiment.create_view`, with one difference on any comet_ml
        version: a name already taken raises a ValueError that says why.
        Experiment view names must be unique across all your projects, though
        `get_views()` lists only the experiment's own project.

        Args:
            experiment (APIExperiment): any experiment in the project
            view (View): from `cometx.views.build_experiment_view`

        Returns: the created View, or None on failure
        """
        _check_scope(view, "experiment")
        try:
            return experiment.create_view(view)
        except CometRestApiException as error:
            if "View name already used" not in str(error):
                raise
            raise ValueError(
                "An experiment view named %r already exists. Experiment view "
                "names must be unique across all your projects, so it may be "
                "in another project: get_views(include_workspace_project_views="
                "True) lists it. Use another name, or update_experiment_view() "
                "to replace one in this project." % view.name
            ) from error

    def update_experiment_view(self, experiment, view: View) -> Any:
        """
        Overwrite an existing experiment view in place, keeping its
        `template_id`. Uses comet_ml's own `APIExperiment.update_view` where the
        installed version has one.

        Args:
            experiment (APIExperiment): any experiment in the view's project
            view (View): a View with `template_id` set, typically from
                `experiment.get_views()`, with a new layout

        Returns: the updated View, or None on failure
        """
        _check_scope(view, "experiment")
        base_update = getattr(experiment, "update_view", None)
        if base_update is not None:
            return base_update(view)
        if not view.template_id:
            raise ValueError(
                "update_experiment_view requires a view with template_id set; "
                "use create_experiment_view to add a new view"
            )
        result = self._client.upsert_chart_template_view(experiment.project_id, view)
        return View.from_payload_dict(result) if result else None

    def log_pr_curves(
        self, experiment, y_true, y_predicted, labels=None, overwrite=False, step=None
    ):
        """
        Log a Precision/Recall curve for each class/column to the given experiment.

        Args:
            experiment: the experiment to log the curves to
            y_true: a list of lists of binary truth (where 1 means that column's class is present)
            y_predicted: a list of list of probabilities of predictions (0 to 1) for each output
            labels (optional): a list of strings (class names)
            overwrite (optional): whether to overwrite previously-logged curves
            step: (optional, by highly encouraged) the step in the training process
        """
        try:
            import numpy as np
            from sklearn.metrics import precision_recall_curve
        except ImportError:
            raise Exception(
                "Experiment.log_pr_curve() requires numpy and sklearn"
            ) from None

        y_true = np.array(y_true)
        y_predicted = np.array(y_predicted)

        if labels is None:
            labels = [("Class %s" % i) for i in range(len(y_true[0]))]

        results = []
        for i in range(len(labels)):
            y, x, _ = precision_recall_curve(y_true[:, i], y_predicted[:, i])

            result = experiment.log_curve(
                name=labels[i],
                x=x,
                y=y,
                overwrite=overwrite,
                step=step,
            )
            results.append(result)
        return results
