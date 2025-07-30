# -*- coding: utf-8 -*-
# ****************************************
#                               __
#    _________  ____ ___  ___  / /__  __
#   / ___/ __ \/ __ `__ \/ _ \/ __/ |/ /
#  / /__/ /_/ / / / / / /  __/ /_ >   <
#  \___/\____/_/ /_/ /_/\___/\__//_/|_/
#
#     Copyright (c) 2023-2024 Cometx
#  Development Team. All rights reserved.
# ****************************************

"""
This module provides an interface for users to export data from
Comet.
"""

import io
import json
import logging
import os
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional

try:
    from tqdm import tqdm as ProgressBar
except ImportError:
    from cometx.utils import ProgressBar

from comet_ml.api import APIExperiment
from comet_ml.artifact_helpers.artifact_getter import get_artifact
from comet_ml.config import get_config
from comet_ml.summary import Summary

from ..._version import __version__
from ...api import API
from ...utils import _input_user_yn, get_query_experiments

LOGGER = logging.getLogger(__name__)

CLONE_TEXT = """
To restore the original git:

```
git clone {origin}
cd {directory}
```
"""

REPRODUCE_CLONE_TEXT = """
git clone {origin}
cd {directory}
"""

README_TEMPLATE = """
Reproduce git commands
---------------------------
{clone_text}
To return to git branch and restore work in progress:

```
git checkout {branch}
git checkout {parent}
{patch_text}
```
"""

REPRODUCE_TEMPLATE = """
{clone_text}
git checkout {branch}
git checkout {parent}
{patch_text}
"""


def is_same(name1: Any, name2: Any) -> bool:
    """
    Check two versions/stages/alias to see if
    they match; case insensitive.
    """
    if name1 is None or name2 is None:
        return False
    return name1.lower() == name2.lower()


def sanitize_filename(filename):
    # type: (str) -> str
    """
    Sanitize filenames so that it can't cause any security
    problems (like overwriting system files).
    """
    filename = "/" + filename
    filename = filename.replace("/../", "/").replace(":", "-")
    while filename.startswith("/"):
        filename = filename[1:]
    return filename


def clean_comet_path(path):
    # type: (str) -> str
    """
    Make sure Comet path is in standard format.
    """
    if not path:
        return path
    while path.endswith("/"):
        path = path[:-1]
    while path.startswith("/"):
        path = path[1:]
    path = path.replace("//", "/")
    return path


def flatten(list: List[List[str]]) -> List[str]:
    """
    Flatten a list of lists into a single list.
    """
    return [item for sublist in list for item in sublist]


class DownloadManager:
    """
    Class for holding all of the download functions.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        The DownloadManager constructor. Optionally takes a Comet API key.
        """
        # Experiment resources:
        self.DEFAULT_RESOURCES = [
            "system",
            "run",  # short for code, git, output, and requirements
            "others",
            "parameters",
            "metadata",
            "metrics",
            "assets",
            "html",
            "project",  # short for project_metadata, project_notes
        ]
        self.META_RESOURCES = {
            "run": [
                "code",
                "requirements",
                "git",
                "output",
                "graph",
            ],
            "project": [
                "project_metadata",
                "project_notes",
            ],
        }
        self.RESOURCE_FUNCTIONS = {
            "system": "download_system_details",
            "code": "download_code",
            "requirements": "download_requirements",
            "output": "download_output",
            "others": "download_others",
            "parameters": "download_parameters",
            "metadata": "download_metadata",
            "metrics": "download_metrics",
            "output": "download_output",
            "assets": "download_assets",
            "git": "download_git",
            "graph": "download_graph",
            "project_metadata": None,  # Project item
            "project_notes": None,  # Project item
            "html": "download_html",
        }
        self.ALL_RESOURCES = sorted(
            list(self.RESOURCE_FUNCTIONS.keys()) + list(self.META_RESOURCES.keys())
        )
        self.SUB_RESOURCES = flatten(
            [self.META_RESOURCES[resource] for resource in self.META_RESOURCES]
        )
        self.api = API(api_key)
        self.config = get_config()

    def list(
        self,
        comet_path: Optional[str] = None,
        use_name: Optional[bool] = False,
    ) -> None:
        """
        The method to list resources.

        Args:
            comet_path: (str, optional) the Comet path to the experiment, artifact, panel, or model-registry
            use_name: (bool, optional) if True, use the experiment name for folder name; else
                use the experiment ID for folder name
        """
        self.download(
            comet_path=comet_path,
            use_name=use_name,
            list_items=True,
        )

    def download(
        self,
        comet_path=None,
        include=None,
        ignore=None,
        output=None,
        use_name=False,
        list_items=False,
        flat=False,
        ask=False,
        filename=None,
        asset_type=None,
        sync="all",
        debug=False,
        query=None,
        max_workers=1,
    ):
        # type: (Optional[str], Optional[List[str]], Optional[List[str]], Optional[str], Optional[bool], Optional[bool], Optional[bool], Optional[bool], Optional[str], Optional[str], Optional[bool], Optional[str]) -> None
        """
        The top-level method to download resources.

        Args:
            comet_path: (str, optional) the Comet path to the experiment, artifact, panel, or model-registry
            query: (str, option) a Comet query string. See:
                https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/API/#apiquery
            include: (list of str, optional) experiment resources to include in download
            ignore: (list of str, optional) experiment resources to ignore
            output: (str, optional) output path to download to; default is current folder
            use_name: (bool, optional) if True, use the experiment name for folder name; else
                use the experiment ID for folder name
            list_items: (bool, optional) if True, just list out the items on command line;
                otherwise, download them
            flat: (bool, optional) if True, do not use folder hierarchy, but just put
                into output folder. For experiment download only.
            ask: (bool, optional) if True, do not ask user for permission; else
                ask user to download
            asset_type:  (str, optional) if given, only match assets with this type
            filename: (str, optional) if given, only download files ending with this
            sync: the level to check if previously downloaded; can be "all" (default),
                "experiment", "project", or "workspace"
        """
        if max_workers > 1:
            self.queue = ThreadPoolExecutor(max_workers=max_workers)
        else:
            self.queue = None
        self.include = set(include if include else self.DEFAULT_RESOURCES[:])
        self.ignore = ignore if ignore else []
        # Remove top-level resources before expanding:
        for resource in self.ignore:
            if resource in self.include:
                self.include.remove(resource)
        # Expand any meta resources:
        for resource in list(self.include):
            if resource in self.META_RESOURCES:
                for new_resource in self.META_RESOURCES[resource]:
                    if new_resource not in self.ignore:
                        self.include.add(new_resource)
                self.include.remove(resource)
        for resource in self.include:
            if resource not in self.ALL_RESOURCES:
                print(
                    "{resource} is not a supported experiment resource; aborting. Should be one of: {supported_resources}.".format(
                        resource=resource,
                        supported_resources=", ".join(self.ALL_RESOURCES),
                    )
                )
                return

        self.root = output if output is not None else os.getcwd()
        self.debug = debug
        self.use_name = use_name
        self.flat = flat
        self.ask = ask
        self.filename = filename
        self.asset_type = asset_type
        self.sync = sync
        self.summary = {key: 0 for key in self.RESOURCE_FUNCTIONS.keys()}
        self.summary["artifacts"] = 0
        self.summary["model-registry"] = 0
        self.summary["panels"] = 0

        comet_path = clean_comet_path(comet_path)
        args = comet_path.split("/") if comet_path is not None else []
        artifact = len(args) > 1 and args[1] == "artifacts"
        model_registry = len(args) > 1 and args[1] == "model-registry"
        panel = len(args) > 1 and args[1] == "panels"

        # Downloads can be one of: experiment, panel, model-registry, or artifact
        if artifact is True:
            if list_items:
                if len(args) == 2:
                    self.list_artifacts(args[0])
                elif len(args) == 3:
                    self.list_artifacts(args[0], args[2])
                else:
                    raise ValueError("use `workspace/artifacts[/name]`")
            else:
                if len(args) == 4:
                    self.download_artifact(args[0], args[2], args[3])
                elif len(args) == 3:
                    self.download_artifact(args[0], args[2])
                else:
                    raise ValueError(
                        "use `workspace/artifacts/name[/version_or_alias]`"
                    )

        elif model_registry is True:
            if list_items:
                if len(args) == 2:
                    self.list_models(args[0])
                elif len(args) == 3:
                    self.list_models(args[0], args[2])
                else:
                    raise ValueError("use `workspace/model-registry[/name]`")
            else:
                if len(args) == 4:
                    self.download_model(args[0], args[2], args[3])
                elif len(args) == 3:
                    self.download_model(args[0], args[2])
                else:
                    raise ValueError(
                        "use `workspace/model-registry/name[/version_or_stage]`"
                    )
        elif panel is True:
            if list_items:
                if len(args) == 2:
                    self.list_panels(args[0])
                elif len(args) == 3:
                    self.list_panels(args[0], args[2])
                else:
                    raise ValueError("use `workspace/panel[/name_or_id]`")
            else:
                if len(args) == 2:
                    self.verify_workspace(args[0])
                    all_panels = self.api.get_panels(args[0])
                    for panel in all_panels:
                        self.download_panel(args[0], panel["templateId"])
                elif len(args) == 3:
                    self.download_panel(args[0], args[2])
                else:
                    raise ValueError("use `workspace/panel/name_or_id`")
        else:
            # Experiment
            if len(self.include) == 0:
                print("Warning: no experiment resources given")
                return

            if len(args) == 0:
                # no comet_path given, do all workspaces for user
                if not list_items:
                    print(
                        "Use `comet download WORKSPACE` where WORKSPACE is one of the following:"
                    )
                self.list_workspaces()
            elif len(args) == 1:
                # Download "workspace"
                # First, see if args[0] is a workspace name:
                if args[0] in self.api.get_workspaces():
                    # Let's list all projects, artifacts, and models
                    if list_items:
                        self.list_workspace(args[0])
                        self.list_artifacts(args[0])
                        self.list_models(args[0])
                    else:
                        self.download_workspace(args[0])
                else:
                    experiment = self.api.get_experiment_by_key(args[0])
                    if experiment:
                        if list_items:
                            self.list_experiment(experiment)
                        else:
                            workspace = experiment.workspace
                            project_name = experiment.project_name
                            self.download_experiment(experiment)
                    else:
                        # assume a valid workspace
                        if list_items:
                            self.list_workspace(args[0])
                            self.list_artifacts(args[0])
                            self.list_models(args[0])
                        else:
                            self.download_workspace(args[0])

            elif len(args) == 2:
                # Download "workspace/project"
                workspace = args[0]
                project_name = args[1]
                if list_items:
                    self.list_project(workspace, project_name, query=query)
                else:
                    self.download_project(workspace, project_name, query=query)
            elif len(args) == 3:
                # "workspace/project/experiment"
                experiment = self.api.get(comet_path)
                if experiment:
                    workspace = experiment.workspace
                    project_name = experiment.project_name

                    if list_items:
                        self.list_experiment(experiment)
                    else:
                        self.download_experiment(experiment)
                else:
                    raise ValueError("no such experiment: %r" % comet_path)
            else:
                print("Invalid Comet path: %r" % comet_path)
                return
        if any(self.summary.values()):
            self.display_summary()

    def display_summary(self):
        # type: () -> None
        """
        Display a summary of downloaded resources.
        """
        print("=" * 33)
        print("Comet Download Summary")
        print("=" * 33)
        print("%-17s: %14s" % ("Resource", "Download Count"))
        print("%-17s: %14s" % ("-" * 17, "-" * 14))
        for key in sorted(self.summary.keys()):
            if self.summary[key] > 0:
                print("%-17s: %14s" % (key, self.summary[key]))
        print("%-17s: %14s" % ("-" * 17, "-" * 14))
        print("%-17s: %14s" % ("Total", sum(self.summary.values())))
        print("=" * 33)

    def list_models(self, workspace: str, name: Optional[str] = None) -> None:
        """
        List the models, one per line.

        Args:
            workspace: (str) name of workspace
            name: (str, optional) name of model (may end with /version or /stage)
        """
        self.verify_workspace(workspace)
        if name:
            self.list_model_versions(workspace, name)
        else:
            model_names = self.api.get_registry_model_names(workspace)
            for name in model_names:
                self.list_model_versions(workspace, name)

    def list_panels(self, workspace, name_or_id=None):
        # type: (str, Optional[str]) -> None
        """
        List the panels, one per line.

        Args:
            workspace: (str) name of workspace
            name_or_id: (str, optional) name or id of panel
        """
        self.verify_workspace(workspace)
        all_panels = self.api.get_panels(workspace)
        if name_or_id is not None:
            selected = [
                panel
                for panel in all_panels
                if (
                    panel["templateId"] == name_or_id
                    or panel["templateName"] == name_or_id
                )
            ]
            if selected:
                panel = selected[0]
                print(
                    "%s/panel/%s (%s)"
                    % (
                        workspace,
                        panel["templateName"],
                        panel["templateId"],
                    )
                )
                for key, value in panel.items():
                    print("    %s:" % key, repr(value))
        else:
            for panel in all_panels:
                print(
                    "%s/panel/%s (%s)"
                    % (
                        workspace,
                        panel["templateName"],
                        panel["templateId"],
                    )
                )

    def list_model_versions(self, workspace, name):
        # type: (str, str) -> None
        """
        List the models and versions, one per line.

        Args:
            workspace: (str) name of workspace
            name: (str) name of model (may end with /version or /stage)
        """
        self.verify_workspace(workspace)
        details = self.api.get_registry_model_details(workspace, name)
        for version in details["versions"]:
            if version["stages"]:
                print(
                    "%s/model-registry/%s/%s (%s)"
                    % (
                        workspace,
                        name,
                        version["version"],
                        ", ".join(version["stages"]),
                    )
                )
            else:
                print("%s/model-registry/%s/%s" % (workspace, name, version["version"]))

    def verify_workspace(self, workspace):
        # type: (str) -> None
        """
        Verify that the workspace is valid.

        Args:
            workspace: (str) name of workspace
        """
        pass
        # if workspace not in self.api.get_workspaces():
        #    raise ValueError("Invalid workspace name: %r" % workspace)

    def list_artifacts(self, workspace, name=None):
        # type: (str, Optional[str]) -> None
        """
        List the artifacts, one per line.

        Args:
            workspace: (str) name of workspace
            name: (str, optional) name of artifact (may end with /version or /alias)
        """
        self.verify_workspace(workspace)
        ajson_list = self.api.get_artifact_list(workspace)
        if name:
            self.list_artifact_details(workspace, name)
        else:
            for ajson in ajson_list["artifacts"]:
                self.list_artifact_details(workspace, ajson["name"])

    def list_artifact_details(self, workspace, name):
        # type: (str, str) -> None
        """
        List the artifact details, one per line.

        Args:
            workspace: (str) name of workspace
            name: (str) name of artifact (may end with /version or /alias)
        """
        self.verify_workspace(workspace)
        details = self.api.get_artifact_details(workspace, name)
        for version in details["versions"]:
            if version["alias"]:
                print(
                    "%s/artifacts/%s/%s (%s)"
                    % (
                        workspace,
                        name,
                        version["version"],
                        ", ".join(version["alias"]),
                    )
                )
            else:
                print("%s/artifacts/%s/%s" % (workspace, name, version["version"]))

    def list_workspaces(self):
        # type: () -> None
        """
        List the user's workspaces, one per line.
        """
        workspaces = self.api.get_workspaces()
        for workspace in sorted(workspaces):
            print(workspace)

    def list_workspace(self, workspace):
        # type: (str) -> None
        """
        List the workspace's projects, one per line.

        Args:
            workspace: (str) name of workspace
        """
        self.verify_workspace(workspace)
        projects = self.api.get_projects(workspace)
        for project_name in sorted(projects):
            print("%s/%s" % (workspace, project_name))

    def get_experiments(self, workspace, project_name, query):
        """
        Return the experiments, possibly matching a query string.
        """
        if query:
            experiments = get_query_experiments(
                self.api, query, workspace, project_name
            )
        else:
            experiments = self.api.get_experiments(workspace, project_name)
        return experiments

    def list_project(self, workspace, project_name, query=None):
        # type: (str, str) -> None
        """
        List the project's experiments, one per line.

        Args:
            workspace: (str) name of workspace
            project_name: (str) name of project
        """
        self.verify_workspace(workspace)
        experiments = self.get_experiments(workspace, project_name, query)
        for experiment in experiments:
            if self.use_name:
                print(
                    "%s/%s/%s"
                    % (
                        workspace,
                        project_name,
                        experiment.name or experiment.id,
                    )
                )
            else:
                print(
                    "%s/%s/%s"
                    % (
                        workspace,
                        project_name,
                        experiment.id,
                    )
                )

    def list_experiment(self, experiment: APIExperiment) -> None:
        """
        List the experiment's Comet path.

        Args:
            experiment: (APIExperiment) the experiment
        """
        if self.use_name:
            print(
                "%s/%s/%s"
                % (
                    experiment.workspace,
                    experiment.project_name,
                    experiment.name or experiment.id,
                )
            )
        else:
            print(
                "%s/%s/%s"
                % (
                    experiment.workspace,
                    experiment.project_name,
                    experiment.id,
                )
            )

    def get_experiment_path(self, experiment: APIExperiment, *subdirs) -> str:
        # type: (str, List[str]) -> str
        """
        Given an APIExperiment, return the Comet path.

        Args:
            experiment: (APIExperiment) the experiment
            subdirs: (strings, optional) additional folders
        """
        if self.use_name:
            name = experiment.name or experiment.key
        else:
            name = experiment.key

        workspace = experiment.workspace
        project_name = experiment.project_name

        return os.path.join(self.root, workspace, project_name, name, *subdirs)

    def _should_write(self, filepath):
        if self.filename:
            retval = re.search(self.filename, filepath)
            if self.debug:
                if not retval:
                    print(
                        "    skipping %r, does not match filename %r"
                        % (filepath, self.filename)
                    )
                else:
                    print("    writing matched %r" % filepath)
            return retval
        else:
            if self.debug:
                print("    writing %r" % filepath)
            return True

    def download_graph(self, experiment: APIExperiment) -> None:
        """
        Given an APIExperiment, download the included resources.

        Args:
            experiment: (APIExperiment) the experiment
        """
        if self.flat:
            path = self.root
        else:
            path = self.get_experiment_path(experiment, "run")

        filepath = os.path.join(path, "graph_definition.txt")
        if self._should_write(filepath):
            graph = experiment.get_model_graph()
            if graph:
                self.summary["graph"] += 1
                os.makedirs(path, exist_ok=True)
                with open(filepath, "w") as f:
                    f.write(graph)

    def download_panel(self, workspace, name_or_id):
        # type: (APIExperiment, str, str) -> None
        """
        Download a panel given its name or ID.

        Args:
            workspace: (str) name of workspace
            name_or_id: (str) name or ID of panel
        """
        self.verify_workspace(workspace)
        if self.flat:
            path = self.root
        else:
            path = os.path.join(self.root, workspace, "panels")

        all_panels = self.api.get_panels(workspace)
        selected = [
            panel
            for panel in all_panels
            if (
                panel["templateId"] == name_or_id or panel["templateName"] == name_or_id
            )
        ]
        if selected:
            panel_id = selected[0]["templateId"]
        else:
            raise Exception("No such panel: %s" % name_or_id)

        os.makedirs(path, exist_ok=True)
        results = self.api.download_panel_zip(
            panel_id, os.path.join(path, "panel-%s.zip" % panel_id)
        )
        if results:
            print("Downloaded %s" % results)
            self.summary["panels"] += 1

    def download_model(self, workspace, name, version_or_stage=None):
        # type: (APIExperiment, str, str, Optional[str]) -> None
        """
        Download a model given its name and optionally version or stage.

        Args:
            workspace: (str) name of workspace
            name: (str) name of model
            version_or_stage: (str, optional) the version or stage
        """
        self.verify_workspace(workspace)
        if self.flat:
            path = self.root
        else:
            path = os.path.join(self.root, workspace, "model-registry", name)

        version = None
        stage = None
        if version_or_stage:
            details = self.api.get_registry_model_details(workspace, name)
            done = False
            for version in details["versions"]:
                if is_same(version_or_stage, version["version"]):
                    version = version["version"]
                    stage = None
                    done = True
                    break
                for stage in version["stages"]:
                    if is_same(version_or_stage, stage):
                        version = None
                        stage = stage
                        done = True
                        break

            if not done:
                raise ValueError("cannot find version or stage: %r" % version_or_stage)

        results = self.api.download_registry_model(
            workspace,
            name,
            expand=True,
            output_path=path,
            stage=stage,
            version=version,
        )
        if results:
            self.summary["model-registry"] += 1

    def download_artifact(self, workspace, name, version_or_alias=None):
        # type: (str, str, Optional[str]) -> None
        """
        Download an artifact given its name and optionally version or alias.

        Args:
            workspace: (str) name of workspace
            name: (str) name of model
            version_or_alias: (str, optional) the version or alias
        """
        if self.flat:
            path = self.root
        else:
            path = os.path.join(self.root, workspace, "artifacts", name)

        # Download artifact decoupled from any Experiment
        artifact_details = self.api._client.get_artifact_details(
            workspace=workspace,
            name=name,
        )
        version = (
            version_or_alias if version_or_alias else artifact_details["latestVersion"]
        )
        params = {
            "artifact_id": artifact_details["artifactId"],
            "version_or_alias": version,
        }
        artifact = get_artifact(
            self.api._client, params, None, Summary("DownloadManager"), self.config
        )
        result = artifact.download(path, "OVERWRITE")
        if result:
            self.summary["artifacts"] += 1

    def download_metadata(self, experiment):
        # type: (APIExperiment) -> None
        """
        Given an APIExperiment, download the metadata.

        Args:
            experiment: (APIExperiment) the experiment
        """
        if self.flat:
            path = self.root
        else:
            path = self.get_experiment_path(experiment)

        filepath = os.path.join(path, "metadata.json")
        if self._should_write(filepath):
            metadata = experiment.get_metadata()
            metadata["tags"] = experiment.get_tags()
            metadata["cometDownloadVersion"] = __version__
            self.summary["metadata"] += 1
            os.makedirs(path, exist_ok=True)
            with open(filepath, "w") as f:
                f.write(json.dumps(metadata))

    def download_html(self, experiment):
        # type: (APIExperiment) -> None
        """
        Given an APIExperiment, download the html.

        Args:
            experiment: (APIExperiment) the experiment
        """
        if self.flat:
            path = self.root
        else:
            path = self.get_experiment_path(experiment, "assets", "html")

        filepath = os.path.join(path, "experiment.html")
        if self._should_write(filepath):
            html = experiment.get_html()
            if html:
                self.summary["html"] += 1
                os.makedirs(path, exist_ok=True)
                with open(filepath, "w") as f:
                    f.write(html)

    def download_metrics(self, experiment):
        # type: (APIExperiment) -> None
        """
        Given an APIExperiment, download the metrics.

        Args:
            experiment: (APIExperiment) the experiment
        """
        if self.flat:
            path = self.root
        else:
            path = self.get_experiment_path(experiment)

        filepath = os.path.join(path, "metrics.jsonl")
        if self._should_write(filepath):
            metrics = experiment.get_metrics()
            if metrics:
                self.summary["metrics"] += 1
                os.makedirs(path, exist_ok=True)
                with open(filepath, "w") as f:
                    for metric in metrics:
                        f.write(json.dumps(metric))
                        f.write("\n")

    def download_requirements(self, experiment):
        # type: (APIExperiment) -> None
        """
        Given an APIExperiment, download the requirements.txt.

        Args:
            experiment: (APIExperiment) the experiment
        """
        if self.flat:
            path = self.root
        else:
            path = self.get_experiment_path(experiment, "run")

        filepath = os.path.join(path, "requirements.txt")
        if self._should_write(filepath):
            details = experiment.get_system_details()
            os_packages = None
            if "installedPackages" in details:
                os_packages = details["installedPackages"]
            if not os_packages and "osPackages" in details:
                os_packages = details["osPackages"]
            if os_packages:
                self.summary["requirements"] += 1
                os.makedirs(path, exist_ok=True)
                with open(filepath, "w") as f:
                    f.write("\n".join(os_packages))

    def download_system_details(self, experiment):
        # type: (APIExperiment) -> None
        """
        Given an APIExperiment, download the system details.

        Args:
            experiment: (APIExperiment) the experiment
        """
        if self.flat:
            path = self.root
        else:
            path = self.get_experiment_path(experiment)

        filepath = os.path.join(path, "system_details.json")
        if self._should_write(filepath):
            details = experiment.get_system_details()
            if "osPackages" in details:
                del details["osPackages"]
            if "installedPackages" in details:
                del details["installedPackages"]
            self.summary["system"] += 1
            os.makedirs(path, exist_ok=True)
            with open(filepath, "w") as f:
                f.write(json.dumps(details))

    def download_others(self, experiment):
        # type: (APIExperiment) -> None
        """
        Given an APIExperiment, download the logged others.

        Args:
            experiment: (APIExperiment) the experiment
        """
        if self.flat:
            path = self.root
        else:
            path = self.get_experiment_path(experiment)

        filepath = os.path.join(path, "others.jsonl")
        if self._should_write(filepath):
            others = experiment.get_others_summary()
            self.summary["others"] += 1
            os.makedirs(path, exist_ok=True)
            with open(filepath, "w") as f:
                for other in others:
                    f.write(json.dumps(other))
                    f.write("\n")

    def download_parameters(self, experiment):
        # type: (APIExperiment) -> None
        """
        Given an APIExperiment, download the hyperparameters.

        Args:
            experiment: (APIExperiment) the experiment
        """
        if self.flat:
            path = self.root
        else:
            path = self.get_experiment_path(experiment)

        filepath = os.path.join(path, "parameters.json")
        if self._should_write(filepath):
            params = experiment.get_parameters_summary()
            self.summary["parameters"] += 1
            os.makedirs(path, exist_ok=True)
            with open(filepath, "w") as f:
                f.write(json.dumps(params))

    def download_git(self, experiment):
        # type: (APIExperiment) -> None
        """
        Given an APIExperiment, download the git data.

        Args:
            experiment: (APIExperiment) the experiment
        """
        if self.flat:
            path = self.root
        else:
            path = self.get_experiment_path(experiment, "run")

        git_meta = None
        git_patch = None
        git_meta_loaded = False
        git_patch_loaded = False

        filepath = os.path.join(path, "git_metadata.json")
        # First, save the git metadata:
        if self._should_write(filepath):
            try:
                git_meta = experiment.get_git_metadata()
            except Exception:
                print("Not allowed to get git metadata for experiment")
                git_meta = {}

            git_meta_loaded = True
            if git_meta and any(git_meta.values()):
                self.summary["git"] += 1
                os.makedirs(path, exist_ok=True)
                with open(filepath, "w") as f:
                    f.write(json.dumps(git_meta))

        filepath = os.path.join(path, "git_diff.patch")
        # Next, save the git patch
        if self._should_write(filepath):
            try:
                git_patch = experiment.get_git_patch()
            except Exception:
                print("Not allowed to get git patch for experiment")
                git_patch = None

            git_patch_loaded = True
            if git_patch:
                try:
                    zip_patch = io.BytesIO(git_patch)
                    archive = zipfile.ZipFile(zip_patch)
                    patch_contents = archive.read("git_diff.patch")
                except Exception:
                    LOGGER.info("assuming zip patch is uncompressed")
                    # Early days, wasn't zip encoded
                    patch_contents = bytes(git_patch, encoding="utf8")
                self.summary["git"] += 1
                os.makedirs(path, exist_ok=True)
                with open(filepath, "wb") as f:
                    f.write(patch_contents)

        filepath = os.path.join(path, "README.md")
        if self._should_write(filepath):
            # Make a README to restore git:
            if not git_meta_loaded:
                try:
                    git_meta = experiment.get_git_metadata()
                except Exception:
                    print("Not allowed to get git metadata for experiment")
                    git_meta = {}

            if not git_patch_loaded:
                try:
                    git_patch = experiment.get_git_patch()
                except Exception:
                    print("Not allowed to get git patch for experiment")
                    git_patch = None

            if git_meta.get("origin"):
                origin = git_meta["origin"]
                directory = git_meta["origin"].split("/")[-1].split(".")[0]
                clone_text = CLONE_TEXT.format(origin=origin, directory=directory)
                if git_patch:
                    patch_text = "git apply git_diff.patch"
                else:
                    patch_text = ""
                if git_meta.get("branch"):
                    git_meta["branch"] = git_meta["branch"].split("/")[-1]

                if git_meta:
                    template = README_TEMPLATE.format(
                        clone_text=clone_text,
                        patch_text=patch_text,
                        branch=git_meta.get("branch"),
                        parent=git_meta.get("parent"),
                    )
                    self.summary["git"] += 1
                else:
                    template = "No git information was available."

                with open(filepath, "w") as f:
                    f.write(template)

    def get_git_text(self, experiment):
        git_meta = experiment.get_git_metadata()
        git_patch = experiment.get_git_patch()

        if git_meta["origin"]:
            origin = git_meta["origin"]
            directory = git_meta["origin"].split("/")[-1].split(".")[0]
            clone_text = REPRODUCE_CLONE_TEXT.format(origin=origin, directory=directory)
        else:
            clone_text = ""

        if git_patch:
            patch_text = "git apply ../git_diff.patch"
        else:
            patch_text = ""

        if git_meta["branch"]:
            git_meta["branch"] = git_meta["branch"].split("/")[-1]

        template = REPRODUCE_TEMPLATE.format(
            clone_text=clone_text,
            patch_text=patch_text,
            branch=git_meta["branch"],
            parent=git_meta["parent"],
        )
        return template

    def download_code(self, experiment):
        # type: (APIExperiment) -> None
        """
        Given an APIExperiment, download the code.

        Args:
            experiment: (APIExperiment) the experiment
        """
        if self.flat:
            path = self.root
        else:
            path = self.get_experiment_path(experiment, "run")

        # FIXME: now in run/code/*.py, as original filename
        filepath = os.path.join(path, "script.py")
        if self._should_write(filepath):
            code = experiment.get_code()
            if code:
                self.summary["code"] += 1
                os.makedirs(path, exist_ok=True)
                with open(filepath, "w") as f:
                    f.write(code)

    def download_output(self, experiment):
        # type: (APIExperiment) -> None
        """
        Given an APIExperiment, download the output.

        Args:
            experiment: (APIExperiment) the experiment
        """
        if self.flat:
            path = self.root
        else:
            path = self.get_experiment_path(experiment, "run")

        filepath = os.path.join(path, "output.txt")
        if self._should_write(filepath):
            output = experiment.get_output()
            if output:
                self.summary["output"] += 1
                os.makedirs(path, exist_ok=True)
                with open(filepath, "w") as f:
                    f.write(output)

    def submit_task(self, file_path, experiment, method_name, args, kwargs):
        def task():
            method = getattr(experiment, method_name)
            results = method(*args, **kwargs)
            if results:
                with open(file_path, "wb+") as f:
                    f.write(results)

        if self.queue is None:
            # Do it now:
            task()
        else:
            # add to queue
            self.queue.submit(task)

    def end(self):
        if self.queue is not None:
            self.queue.shutdown(wait=True)

    def download_assets(self, experiment):
        # type: (APIExperiment) -> None
        """
        Given an APIExperiment, download the assets.

        Args:
            experiment: (APIExperiment) the experiment
        """
        if self.flat:
            assets_path = self.root
        else:
            assets_path = self.get_experiment_path(experiment, "assets")

        assets = experiment.get_asset_list(
            self.asset_type if self.asset_type else "all"
        )
        if len(assets) > 0:
            filename = "assets_metadata.jsonl"
            filepath = os.path.join(assets_path, filename)
            if self._should_write(filepath):
                self.summary["assets"] += 1
                os.makedirs(assets_path, exist_ok=True)
                with open(filepath, "w") as f:
                    for asset in assets:
                        if asset["type"] == "audio" and asset["step"] is not None:
                            asset_filename = asset["fileName"]
                            asset["logAsFileName"] = asset_filename
                            if "." in asset_filename:
                                asset_filename, ext = asset_filename.split(".", 1)
                            else:
                                asset_filename, ext = asset_filename, ""
                            asset_filename = "%s-%s.%s" % (
                                asset_filename,
                                asset["step"],
                                ext,
                            )
                            asset["fileName"] = asset_filename
                        f.write(json.dumps(asset))
                        f.write("\n")

        filenames = set([])
        for asset in assets:
            asset_type = asset["type"] if asset["type"] else "asset"
            if self.flat:
                path = assets_path
            else:
                path = os.path.join(assets_path, asset_type)
            filename = sanitize_filename(asset["fileName"])
            file_path = os.path.join(path, filename)
            # Don't download a filename more than once:
            if file_path not in filenames and self._should_write(file_path):
                filenames.add(file_path)
                self.summary["assets"] += 1
                path, filename = os.path.split(file_path)
                os.makedirs(path, exist_ok=True)
                self.submit_task(
                    file_path, experiment, "get_asset", [asset["assetId"]], {}
                )

    def download_asset(self, experiment, asset_filename):
        # type: (APIExperiment) -> None
        """
        Given an APIExperiment and asset name, download the asset.

        Args:
            experiment: (APIExperiment) the experiment
            asset_filename: (str) name of asset
        """
        if self.flat:
            assets_path = self.root
        else:
            assets_path = self.get_experiment_path(experiment, "assets")

        assets = experiment.get_asset_list()
        for asset in assets:
            asset_type = asset["type"] if asset["type"] else "asset"
            if self.flat:
                path = assets_path
            else:
                path = os.path.join(assets_path, asset_type)
            filename = sanitize_filename(asset["fileName"])

            if filename == asset_filename:
                file_path = os.path.join(path, filename)
                # Don't download a filename more than once:
                if self._should_write(file_path):
                    self.summary["assets"] += 1
                    path, filename = os.path.split(file_path)
                    os.makedirs(path, exist_ok=True)
                    raw = experiment.get_asset(asset["assetId"])
                    with open(file_path, "wb+") as f:
                        f.write(raw)

    def download_experiment(self, experiment, top_level=True):
        # type: (APIExperiment, Optional[bool]) -> None
        """
        Given an APIExperiment, download all of the included resources.

        Args:
            experiment: (APIExperiment) the experiment
            top_level: (bool, optional) is this the top of the download
                hierarchy?
        """
        path = self.get_experiment_path(experiment)
        if os.path.exists(path) and self.sync == "experiment":
            print("Sync: skipping existing experiment path %r..." % path)
            return

        functions = []
        for resource in self.include:
            if resource in self.RESOURCE_FUNCTIONS:
                function = self.RESOURCE_FUNCTIONS[resource]
                functions.append(function)

        if top_level:
            functions = ProgressBar(functions, "Downloading experiment")
        elif self.flat:
            raise ValueError("--flat cannot be used with multiple experiment downloads")

        # Download experiment items:
        for function_name in functions:
            if function_name is None:
                continue
            function = getattr(self, function_name)
            try:
                function(experiment)

            except Exception as err:
                print("Error in experiment %r: %s" % (function, err))

    def download_project(self, workspace, project_name, top_level=True, query=None):
        # type: (str, str, Optional[bool]) -> None
        """
        Download a project.

        Args:
            workspace: (str) name of workspace
            project_name: (str) name of project
            query: (str, optional) Comet query string
            top_level: (bool, optional) is this the top of the download
                hierarchy?
        """
        self.verify_workspace(workspace)

        path = os.path.join(self.root, workspace, project_name)

        project_metadata = self.api.get_project(workspace, project_name)
        if project_metadata:
            project_metadata["cometDownloadVersion"] = __version__
        else:
            raise Exception("No such project: %s/%s" % (workspace, project_name))

        # If the project exists, anssync at project level, skip it
        if self.sync == "project" and os.path.exists(path):
            print("Sync: skipping existing project path %r..." % path)
            return

        filepath = os.path.join(path, "project_metadata.json")
        if self._should_write(filepath) and "project_metadata" in self.include:
            self.summary["project_metadata"] += 1
            os.makedirs(path, exist_ok=True)
            with open(filepath, "w") as f:
                f.write(json.dumps(project_metadata))

        filepath = os.path.join(path, "project_notes.md")
        if self._should_write(filepath) and "project_notes" in self.include:
            notes = self.api.get_project_notes(workspace, project_name)
            if notes:
                self.summary["project_notes"] += 1
                os.makedirs(path, exist_ok=True)
                with open(filepath, "w") as f:
                    f.write(notes)

        if "experiments" not in self.ignore:
            project_experiments = self.get_experiments(workspace, project_name, query)
            if top_level:
                if self.flat:
                    raise ValueError(
                        "--flat cannot be used with multiple experiment downloads"
                    )
                if not self._confirm_download(len(project_experiments)):
                    return
                project_experiments = ProgressBar(
                    project_experiments, "Downloading experiments"
                )

            for experiment in project_experiments:
                self.download_experiment(experiment, top_level=False)

    def download_workspace(self, workspace, top_level=True, query=None):
        # type: (str, Optional[bool]) -> None
        """
        Download a workspace.

        Args:
            workspace: (str) name of workspace
            top_level: (bool, optional) is this the top of the download
                hierarchy?
        """
        # If the workspace exists, and sync at workspace level, skip it
        path = os.path.join(self.root, workspace)
        if self.sync == "workspace" and os.path.exists(path):
            print("Sync: skipping existing workspace path %r..." % path)
            return

        self.verify_workspace(workspace)
        projects = self.api.get_projects(workspace)
        if top_level and len(projects) > 0:
            if self.flat:
                raise ValueError(
                    "--flat cannot be used with multiple experiment downloads"
                )
            total = 0
            if self.ask and "experiments" not in self.ignore:
                for project_name in ProgressBar(projects, "Calculating download"):
                    metadata = self.api.get_project(workspace, project_name)
                    total = total + int(metadata["numberOfExperiments"])
                if not self._confirm_download(total):
                    return
            projects = ProgressBar(projects, "Downloading projects")

        for project_name in projects:
            self.download_project(
                workspace,
                project_name,
                top_level=False,
                query=query,
            )

    def _confirm_download(self, total):
        # type: (int) -> bool
        """
        Get input from a user to confirm the download.

        Args:
            total: (int) the number of experiments to consider
                for downloading
        """
        if total < 2:
            return True
        if not self.ask:
            return True

        prompt = (
            "Consider {total} experiments (maximum) for downloading? (y/n) ".format(
                total=total
            )
        )
        return _input_user_yn(prompt)
