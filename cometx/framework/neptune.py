# -*- coding: utf-8 -*-
# ****************************************
#                               __
#    _________  ____ ___  ___  / /__  __
#   / ___/ __ \/ __ `__ \/ _ \/ __/ |/ /
#  / /__/ /_/ / / / / / /  __/ /_ >   <
#  \___/\____/_/ /_/ /_/\___/\__//_/|_/
#
#     Copyright (c) 2023-2025 Cometx
#  Development Team. All rights reserved.
# ****************************************

import json
import os

import comet_ml

from ..utils import remove_extra_slashes


def clean_for_filename(name):
    return name.replace("/", "-").replace(":", "")


class DownloadManager:
    """
    Class that is in charge of fetching all relevant information for a
    Neptune experiment
    """

    def __init__(
        self,
        include=None,
        ignore=None,
        output=None,
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
        self.root = output if output is not None else os.getcwd()
        self.debug = debug
        self.flat = flat
        self.ask = ask
        self.filename = filename
        self.asset_type = asset_type
        self.sync = sync
        self.ignore = ignore if ignore else []
        # Data:
        self._metrics = []  # metrics.jsonl
        self._parameters = []  # parameters.json
        self._others = []  # others.jsonl

    def download(self, PATH):
        import neptune
        from neptune.management import get_project_list

        neptune_api_token = os.environ.get("NEPTUNE_API_TOKEN")
        if neptune_api_token is None:
            raise Exception(
                "You need to have your NEPTUNE_API_TOKEN environment variable set"
            )

        path = remove_extra_slashes(PATH)
        path_parts = path.split("/")

        if len(path_parts) == 1:
            # Just given workspace:
            workspace = path_parts[0]
            projects = [
                proj.split("/", 1)[1]
                for proj in get_project_list()
                if proj.startswith(workspace + "/")
            ]
        elif len(path_parts) == 2:
            # Given workspace/project:
            workspace, project = path_parts
            projects = [project]
        else:
            raise Exception("invalid PATH: %r" % PATH)

        # Download items:

        for project_name in projects:
            neptune_project_name = workspace + "/" + project_name
            neptune_project = neptune.Project(
                neptune_project_name,
            )
            df = neptune_project.fetch_runs_table().to_pandas()
            # Only select non-deleted neptune runs:
            columns = df[df["sys/trashed"] == False]

            for row_id in columns["sys/id"]:
                run = neptune.Run(
                    with_id=row_id, project=neptune_project_name, mode="read-only"
                )
                self.export_all(run)
                self.write_data(workspace, project_name, run)

    def get_path(self, workspace, project, experiment, *subdirs, filename):
        if self.flat:
            path = self.root
        else:
            path = os.path.join(self.root, workspace, project, experiment, *subdirs)
        os.makedirs(path, exist_ok=True)
        if filename:
            path = os.path.join(path, filename)

        return path

    def write_data(self, workspace, project, run):
        experiment = run._sys_id
        self._others.append(
            {
                "name": "Name",
                "valueMax": experiment,
                "valueMin": experiment,
                "valueCurrent": experiment,
                "editable": False,
            }
        )
        # Make sure experiment is a good folder name from here down
        experiment = clean_for_filename(experiment)

        # Metadata:
        owner = self.get_owner(run)
        metadata = {
            "experimentName": run._sys_id,
            "userName": owner,
            "projectName": project,
            "workspaceName": workspace,
            "filePath": "unknown",
            "fileName": "unknown",
            "cometDownloadVersion": comet_ml.__version__,
        }
        path = self.get_path(workspace, project, experiment, filename="metadata.json")
        with open(path, "w") as fp:
            fp.write(json.dumps(metadata) + "\n")

        # Metrics:
        path = self.get_path(workspace, project, experiment, filename="metrics.jsonl")
        with open(path, "w") as fp:
            for data in self._metrics:
                fp.write(json.dumps(data) + "\n")

        # Others:
        path = self.get_path(workspace, project, experiment, filename="others.jsonl")
        with open(path, "w") as fp:
            for data in self._others:
                fp.write(json.dumps(data) + "\n")

        # Parameters:
        path = self.get_path(workspace, project, experiment, filename="parameters.json")
        with open(path, "w") as fp:
            fp.write(json.dumps(self._parameters) + "\n")

    def export_all(self, run):
        self.export_hyperparameters(run)
        self.export_metrics(run)
        self.export_final_metrics(run)
        self.export_metadata(run)
        # self.export_system_and_os_info()
        # self.export_standard_output()
        # self.export_dependencies()
        # self.export_model_graph()

    def get_owner(self, run):
        if run.exists("sys/owner"):
            owner = run["sys/owner"].fetch()
            return owner

    def export_hyperparameters(self, run):
        # Try both 'parameters' and 'hyperparameters' paths
        if run.exists("parameters"):
            hyperparameters = run["parameters"].fetch()
            for key, value in hyperparameters.items():
                self._parameters.append(
                    {
                        "name": key,
                        "valueMax": value,
                        "valueMin": value,
                        "valueCurrent": value,
                        "editable": False,
                    }
                )

        elif run.exists("hyperparameters"):
            hyperparameters = run["hyperparameters"].fetch()
            for key, value in hyperparameters.items():
                self._parameters.append(
                    {
                        "name": key,
                        "valueMax": value,
                        "valueMin": value,
                        "valueCurrent": value,
                        "editable": False,
                    }
                )

    def export_metrics(self, run):
        # Common metric paths in Neptune
        metric_paths = [
            "train/loss",
            "train/accuracy",
            "val/loss",
            "val/accuracy",
            "training/train/epoch/accuracy",
            "training/train/epoch/loss",
            "training/train/batch/accuracy",
            "training/train/batch/loss",
            "training/test/epoch/accuracy",
            "training/test/epoch/loss",
            "training/test/batch/accuracy",
            "training/test/batch/loss",
        ]

        plot_dict = {
            plot_name: run[plot_name].fetch_values()
            for plot_name in metric_paths
            if run.exists(plot_name)
        }
        for plot_name, plot in plot_dict.items():
            values = plot["value"].to_list()
            steps = plot["step"].to_list()
            times = plot["timestamp"].to_list()
            # Log each metric point individually with its step
            for step, value, ts in zip(steps, values, times):
                self._metrics.append(
                    {
                        "metricName": plot_name,
                        "metricValue": value,
                        "timestamp": int(ts.value / 1_000_000),
                        "step": int(step),
                        "epoch": None,
                        "runContext": None,
                    }
                )

    def export_final_metrics(self, run):
        """Export final metrics if they exist, as other"""
        if run.exists("final"):
            try:
                # Get structure from root object
                root_structure = run.get_structure()
                if "final" in root_structure:
                    final_structure = root_structure["final"]
                    for key in final_structure.keys():
                        metric_path = f"final/{key}"
                        if run.exists(metric_path):
                            value = run[metric_path].fetch()
                            self._others.append(
                                {
                                    "name": f"final/{key}",
                                    "valueMax": str(value),
                                    "valueMin": str(value),
                                    "valueCurrent": str(value),
                                    "editable": False,
                                }
                            )
            except Exception as e:
                print(f"Warning: Could not export final metrics: {e}")

    def export_metadata(self, run):
        """Export metadata if it exists"""
        if run.exists("metadata"):
            try:
                # Get structure from root object
                root_structure = run.get_structure()
                if "metadata" in root_structure:
                    metadata_structure = root_structure["metadata"]
                    for key in metadata_structure.keys():
                        metadata_path = f"metadata/{key}"
                        if run.exists(metadata_path):
                            value = run[metadata_path].fetch()
                            self._others.append(
                                {
                                    "name": f"metadata_{key}",
                                    "valueMax": str(value),
                                    "valueMin": str(value),
                                    "valueCurrent": str(value),
                                    "editable": False,
                                }
                            )
            except Exception as e:
                print(f"Warning: Could not export metadata: {e}")

    def end(self):
        # Close up parallel downloads
        pass
