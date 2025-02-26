#!/usr/bin/env python
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
"""
To copy experiment data to new experiments.

cometx copy [--symlink] SOURCE DESTINATION

where SOURCE is:

* if not --symlink, "WORKSPACE/PROJECT/EXPERIMENT", "WORKSPACE/PROJECT", or "WORKSPACE" folder
* if --symlink, then it is a Comet path to workspace or workspace/project
* "WORKSPACE/panels" or "WORKSPACE/panels/PANEL-ZIP-FILENAME" to copy panels

where DESTINATION is:

* WORKSPACE
* WORKSPACE/PROJECT

Not all combinations are possible:


| Destination:       | WORKSPACE            | WORKSPACE/PROJECT      |
| Source (below)     |                      |                        |
|--------------------|----------------------|------------------------|
| WORKSPACE          | Copies all projects  | N/A                    |
| WORKSPACE/PROJ     | N/A                  | Copies all experiments |
| WORKSPACE/PROJ/EXP | N/A                  | Copies experiment      |

Asset types:

* 3d-image
* 3d-points - deprecated
* audio
* confusion-matrix - may contain assets
* curve
* dataframe
* dataframe-profile
* datagrid
* embeddings - may reference image asset
* histogram2d - not used
* histogram3d - internal only, single histogram, partial logging
* histogram_combined_3d
* image
* llm_data
* model-element
* notebook
* source_code
* tensorflow-model-graph-text - not used
* text-sample
* video

"""

import argparse
import glob
import io
import json
import os
import re
import sys
import tempfile
import urllib.parse
import zipfile

from comet_ml import APIExperiment, Artifact, Experiment, OfflineExperiment
from comet_ml._typing import TemporaryFilePath
from comet_ml.utils import compress_git_patch
from comet_ml.file_uploader import GitPatchUploadProcessor
from comet_ml.messages import (
    GitMetadataMessage,
    HtmlMessage,
    InstalledPackagesMessage,
    MetricMessage,
    StandardOutputMessage,
    SystemDetailsMessage,
)
from comet_ml.offline_utils import write_experiment_meta_file

from ..api import API
from ..utils import remove_extra_slashes
from .copy_utils import upload_single_offline_experiment

ADDITIONAL_ARGS = False


class OfflineExperiment(OfflineExperiment):
    """
    Wrapper to alter start/stop times
    """

    START_TIME = None
    STOP_TIME = None

    def _write_experiment_meta_file(self):
        write_experiment_meta_file(
            tempdir=self.tmpdir,
            experiment_key=self.id,
            workspace=self.workspace,
            project_name=self.project_name,
            start_time=self.START_TIME or self.start_time,
            stop_time=self.STOP_TIME or self.stop_time,
            tags=self.get_tags(),
            resume_strategy=self.resume_strategy,
            customer_error_reported=self.customer_error_reported,
            customer_error_message=self.customer_error_message,
            user_provided_experiment_key=self.user_provided_experiment_key,
            comet_start_sourced=self.comet_start_sourced,
        )


def get_parser_arguments(parser):
    parser.add_argument(
        "COMET_SOURCE",
        help=(
            "The folder containing the experiments to copy: 'workspace', or 'workspace/project' or 'workspace/project/experiment'"
        ),
        type=str,
    )
    parser.add_argument(
        "COMET_DESTINATION",
        help=("The Comet destination: 'WORKSPACE', 'WORKSPACE/PROJECT'"),
        type=str,
    )
    parser.add_argument(
        "-i",
        "--ignore",
        help="Resource(s) (or 'experiments') to ignore.",
        nargs="+",
        default=[],
    )
    parser.add_argument(
        "--debug", help="If given, allow debugging", default=False, action="store_true"
    )
    parser.add_argument(
        "--quiet",
        help="If given, don't display update info",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--symlink",
        help="Instead of copying, create a link to an experiment in a project",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--sync",
        help="Check to see if experiment name has been created first; if so, skip",
        default=False,
        action="store_true",
    )


def copy(parsed_args, remaining=None):
    # Called via `cometx copy ...`
    try:
        copy_manager = CopyManager()
        copy_manager.copy(
            parsed_args.COMET_SOURCE,
            parsed_args.COMET_DESTINATION,
            parsed_args.symlink,
            parsed_args.ignore,
            parsed_args.debug,
            parsed_args.quiet,
            parsed_args.sync,
        )
        if parsed_args.debug:
            print("finishing...")

    except KeyboardInterrupt:
        if parsed_args.debug:
            raise
        else:
            print("Canceled by CONTROL+C")
    except Exception as exc:
        if parsed_args.debug:
            raise
        else:
            print("ERROR: " + str(exc))


def get_query_dict(url):
    """
    Given a URL, return query items as key/value dict.
    """
    result = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(result.query)
    return {key: values[0] for key, values in query.items()}


class CopyManager:
    def __init__(self):
        """
        | Destination:       | WORKSPACE            | WORKSPACE/PROJECT      |
        | Source (below)     |                      |                        |
        |--------------------|----------------------|------------------------|
        | WORKSPACE          | Copies all projects  | N/A                    |
        | WORKSPACE/PROJ     | N/A                  | Copies all experiments |
        | WORKSPACE/PROJ/EXP | N/A                  | Copies experiment      |
        """
        self.api = API()

    def copy(self, source, destination, symlink, ignore, debug, quiet, sync):
        """ """
        self.ignore = ignore
        self.debug = debug
        self.quiet = quiet
        self.sync = sync
        self.copied_reports = False
        comet_destination = remove_extra_slashes(destination)
        comet_destination = comet_destination.split("/")
        if len(comet_destination) == 2:
            workspace_dst, project_dst = comet_destination
        elif len(comet_destination) == 1:
            workspace_dst = comet_destination[0]
            project_dst = None
        else:
            raise Exception("invalid COMET_DESTINATION: %r" % destination)

        comet_source = remove_extra_slashes(source)
        comet_source = comet_source.split("/")

        if len(comet_source) == 3:
            workspace_src, project_src, experiment_src = comet_source
        elif len(comet_source) == 2:
            workspace_src, project_src = comet_source
            experiment_src = "*"
        elif len(comet_source) == 1:
            workspace_src = comet_source[0]
            project_src, experiment_src = "*", "*"
        else:
            raise Exception("invalid COMET_SOURCE: %r" % source)

        # First check to make sure workspace_dst exists:
        workspaces = self.api.get_workspaces()
        if workspace_dst not in workspaces:
            raise Exception(
                f"{workspace_dst} does not exist; use the Comet UI to create it"
            )

        if project_src == "panels":
            # experiment_src may be "*" or filename
            for filename in glob.glob(
                os.path.join(workspace_src, project_src, experiment_src)
            ):
                print("Uploading panel zip: %r to %r..." % (filename, workspace_dst))
                self.api.upload_panel_zip(workspace_dst, filename)
            return

        # For checking if the project_dst exists below:
        projects = self.api.get_projects(workspace_dst)

        for experiment_folder in self.get_experiment_folders(
            workspace_src, project_src, experiment_src
        ):
            if experiment_folder.count("/") >= 2:
                folder_workspace, folder_project, folder_experiment = (
                    experiment_folder
                ).rsplit("/", 2)
            else:
                print("Unknown folder: %r; ignoring" % experiment_folder)
                continue
            if folder_experiment in ["project_metadata.json"]:
                continue
            temp_project_dst = project_dst
            if temp_project_dst is None:
                temp_project_dst = folder_project

            # Next, check if the project_dst exists:
            if temp_project_dst not in projects:
                project_metadata_path = os.path.join(
                    workspace_src, project_src, "project_metadata.json"
                )
                if os.path.exists(project_metadata_path):
                    with open(project_metadata_path) as fp:
                        project_metadata = json.load(fp)
                    self.api.create_project(
                        workspace_dst,
                        temp_project_dst,
                        project_description=project_metadata["projectDescription"],
                        public=project_metadata["public"],
                    )
                projects.append(temp_project_dst)

            if symlink:
                print(
                    f"Creating symlink from {workspace_src}/{project_src}/{experiment_src} to {workspace_dst}/{temp_project_dst}"
                )
                experiment = APIExperiment(previous_experiment=experiment_src)
                experiment.create_symlink(temp_project_dst)
                print(
                    f"    New symlink created: {self.api._get_url_server()}/{workspace_dst}/{temp_project_dst}/{experiment_src}"
                )
            elif "experiments" not in self.ignore:
                self.copy_experiment_to(
                    experiment_folder, workspace_dst, temp_project_dst
                )

    def create_experiment(self, workspace_dst, project_dst, offline=True):
        """
        Create an experiment in destination workspace
        and project, and return an Experiment.
        """
        if not self.quiet:
            print("Creating experiment...")

        ExperimentClass = OfflineExperiment if offline else Experiment
        experiment = ExperimentClass(
            project_name=project_dst,
            workspace=workspace_dst,
            log_code=False,
            log_graph=False,
            auto_param_logging=False,
            auto_metric_logging=False,
            parse_args=False,
            auto_output_logging="simple",
            log_env_details=False,
            log_git_metadata=False,
            log_git_patch=False,
            disabled=False,
            log_env_gpu=False,
            log_env_host=False,
            display_summary=None,
            log_env_cpu=False,
            log_env_network=False,
            display_summary_level=1,
            optimizer_data=None,
            auto_weight_logging=None,
            auto_log_co2=False,
            auto_metric_step_rate=10,
            auto_histogram_tensorboard_logging=False,
            auto_histogram_epoch_rate=1,
            auto_histogram_weight_logging=False,
            auto_histogram_gradient_logging=False,
            auto_histogram_activation_logging=False,
            experiment_key=None,
        )

        def filter_messages(method):
            def filtered_method(message):
                if hasattr(message, "context") and message.context == "ignore":
                    return
                method(message)

            return filtered_method

        experiment.streamer.put_message_in_q = filter_messages(
            experiment.streamer.put_message_in_q
        )
        return experiment

    def get_experiment_folders(self, workspace_src, project_src, experiment_src):
        for path in glob.iglob(f"{workspace_src}/{project_src}/{experiment_src}"):
            if any(
                [path.endswith("~"), path.endswith(".json"), path.endswith(".jsonl")]
            ):
                continue
            else:
                yield path

    def copy_experiment_to(self, experiment_folder, workspace_dst, project_dst):
        title = experiment_folder
        experiment_name = None
        # See if there is a name:
        filename = os.path.join(experiment_folder, "others.jsonl")
        if os.path.isfile(filename):
            with open(filename) as fp:
                line = fp.readline()
                while line:
                    others_json = json.loads(line)
                    if others_json["name"] == "Name":
                        experiment_name = others_json["valueCurrent"]
                        title = (
                            f"{experiment_folder} (\"{others_json['valueCurrent']}\")"
                        )
                        break
                    line = fp.readline()
        print(f"Copying from {title} to {workspace_dst}/{project_dst}...")

        # Copy other project-level items to an experiment:
        if "reports" not in self.ignore and not self.copied_reports:
            experiment = None
            workspace_src, project_src, _ = experiment_folder.split("/")
            reports = os.path.join(workspace_src, project_src, "reports", "*")
            for filename in glob.glob(reports):
                if filename.endswith("reports_metadata.jsonl"):
                    continue
                basename = os.path.basename(filename)
                artifact = Artifact(basename, "Report")
                artifact.add(filename)
                if experiment is None:
                    experiment = self.create_experiment(
                        workspace_dst, project_dst, offline=False
                    )
                    experiment.log_other("Name", "Reports")
                experiment.log_artifact(artifact)
            if experiment:
                experiment.end()
            self.copied_reports = True

        if self.sync:
            if experiment_name is not None:
                experiment = self.api.get_experiment(
                    workspace_dst, project_dst, experiment_name
                )
                if experiment is not None:
                    print("    Experiment exists; skipping due to --sync")
                    return
                else:
                    print("   Experiment doesn't exist on destination; copying...")
            else:
                print("    Can't sync because source has no name; copying...")

        experiment = self.create_experiment(workspace_dst, project_dst)
        # copy experiment_folder stuff to experiment
        # copy all resources to existing or new experiment
        self.log_all(experiment, experiment_folder)
        experiment.end()

        print(
            f"Uploading {experiment.offline_directory}/{experiment._get_offline_archive_file_name()}"
        )
        url = upload_single_offline_experiment(
            offline_archive_path=os.path.join(
                experiment.offline_directory,
                experiment._get_offline_archive_file_name(),
            ),
            settings=self.api.config,
            force_upload=False,
        )
        if url:
            print("Experiment copied to: %s" % url)
        else:
            print("ERROR: this experiment failed to copy")

    def log_metadata(self, experiment, filename):
        if not self.quiet:
            with experiment.context_manager("ignore"):
                print("log_metadata...")
        if os.path.exists(filename):
            metadata = json.load(open(filename))
            experiment.add_tags(metadata.get("tags", []))
            if metadata.get("fileName", None):
                experiment.set_filename(metadata["fileName"])

            OfflineExperiment.START_TIME = metadata.get("startTimeMillis")
            OfflineExperiment.STOP_TIME = metadata.get("endTimeMillis")

    def log_system_details(self, experiment, filename):
        if not self.quiet:
            with experiment.context_manager("ignore"):
                print("log_system_details...")
        if os.path.exists(filename):
            system = json.load(open(filename))

            # System info:
            message = SystemDetailsMessage(
                command=system.get("command") or [],
                env=system.get("env") or {},
                hostname=system.get("hostname") or "",
                ip=system.get("ip") or "",
                machine=system.get("machine") or "",
                os_release=system.get("osRelease") or "",
                os_type=system.get("osType") or "",
                os=system.get("os") or "",
                pid=system.get("pid", None) or 0,
                processor=system.get("processor") or "",
                python_exe=system.get("executable") or "",
                python_version_verbose=system.get("pythonVersionVerbose") or "",
                python_version=system.get("pythonVersion") or "",
                user=system.get("user") or "",
            )
            experiment._enqueue_message(message)

    def log_graph(self, experiment, filename):
        if not self.quiet:
            with experiment.context_manager("ignore"):
                print("log_graph...")
        if os.path.exists(filename):
            experiment.set_model_graph(open(filename).read())

    def _log_asset_filename(
        self, experiment, asset_type, metadata, filename, step, log_filename
    ):
        if isinstance(filename, io.BytesIO):
            binary_io = filename
        else:
            binary_io = open(filename, "rb")

        # If the filename has a sequence number:
        sequence = re.search(r"\((\d+)\)$", log_filename)
        if sequence:
            log_filename = log_filename.rsplit("(", 1)[0].strip()

        result = experiment._log_asset(
            binary_io,
            file_name=log_filename,
            copy_to_tmp=True,
            asset_type=asset_type,
            metadata=metadata,
            step=step,
        )
        return result

    def update_datagrid_contents(
        self,
        experiment,
        asset_map,
        metadata,
        step,
        zip_file_path,
        log_as_filename,
        old_asset_id,
    ):
        """
        Take a datagrid zip, and replace the asset IDs with new
        asset IDs.
        """
        basename = os.path.basename(zip_file_path)
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_to_edit = None
            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                for file in zip_ref.namelist():
                    file_to_edit = file
                zip_ref.extractall(tmp_dir)

            file_path = os.path.join(tmp_dir, file_to_edit)
            print(file_path)
            with open(file_path, "r") as fp:
                json_data = json.load(fp)
                asset_columns = []
                for i, column in enumerate(json_data["columns"]):
                    if json_data["columns"][column] == "IMAGE-ASSET":
                        asset_columns.append(i)
                for row in json_data["rows"]:
                    for column in asset_columns:
                        old = row[column]["asset_id"]
                        row[column]["asset_id"] = asset_map[old]

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, basename)
            with open(file_path, "wb") as fp:
                with zipfile.ZipFile(fp, "w", zipfile.ZIP_DEFLATED) as zfp:
                    zfp.writestr(file_to_edit, json.dumps(json_data))
            result = self._log_asset_filename(
                experiment,
                "datagrid",
                metadata,
                file_path,
                step,
                log_as_filename,
            )
            asset_map[old_asset_id] = result["assetId"]

    def _log_asset(
        self, experiment, path, asset_type, log_filename, assets_metadata, asset_map
    ):
        log_as_filename = assets_metadata[log_filename].get(
            "logAsFileName",
            None,
        )
        step = assets_metadata[log_filename].get("step")
        epoch = assets_metadata[log_filename].get("epoch")
        old_asset_id = assets_metadata[log_filename].get("assetId")
        if asset_type in self.ignore:
            return
        if log_filename.startswith("/"):
            filename = os.path.join(path, asset_type, log_filename[1:])
        else:
            filename = os.path.join(path, asset_type, log_filename)

        filename = filename.replace(":", "-")

        if not os.path.isfile(filename):
            with experiment.context_manager("ignore"):
                print("Missing file %r: unable to copy" % filename)
            return

        metadata = assets_metadata[log_filename].get("metadata")
        metadata = json.loads(metadata) if metadata else {}

        if asset_type == "notebook":
            result = experiment.log_notebook(filename)  # done!
            asset_map[old_asset_id] = result["assetId"]
        elif asset_type == "embeddings":
            # This will come after contained assets
            with open(filename) as fp:
                em_json = json.load(fp)
            # go though JSON, replace asset_ids with new asset_ids
            # {"embeddings":
            #    [{"tensorName": "Comet Embedding",
            #      "tensorShape": [240, 5],
            #      "tensorPath": "/api/asset/download?assetId=b6edbf11e548417580af163b20d7fd23&experimentKey=fe5ed0231e4e4425a13b7c25ea82c51f",
            #      "metadataPath": "/api/asset/download?assetId=fcac2559f7cc42f8a14d20ebed4f8da1&experimentKey=fe5ed0231e4e4425a13b7c25ea82c51f",
            #      "sprite": {
            #         "imagePath": "/api/image/download?imageId=2052efea88b24d4b9111e0e4b0bdb003&experimentKey=fe5ed0231e4e4425a13b7c25ea82c51f",
            #         "singleImageDim": [6, 6]
            #      }
            #     }]
            # }
            for embedding in em_json["embeddings"]:
                if embedding.get("tensorPath"):
                    args = get_query_dict(embedding["tensorPath"])
                    new_args = {
                        "experimentKey": experiment.id,
                        "assetId": asset_map[args.get("assetId", args.get("imageId"))],
                    }
                    embedding["tensorPath"] = (
                        "/api/asset/download?assetId={assetId}&experimentKey={experimentKey}".format(
                            **new_args
                        )
                    )
                if embedding.get("metadataPath"):
                    args = get_query_dict(embedding["metadataPath"])
                    new_args = {
                        "experimentKey": experiment.id,
                        "assetId": asset_map[args.get("assetId", args.get("imageId"))],
                    }
                    embedding["metadataPath"] = (
                        "/api/asset/download?assetId={assetId}&experimentKey={experimentKey}".format(
                            **new_args
                        )
                    )
                if embedding.get("sprite"):
                    if embedding["sprite"].get("imagePath"):
                        args = get_query_dict(embedding["sprite"]["imagePath"])
                        new_args = {
                            "experimentKey": experiment.id,
                            "assetId": asset_map[
                                args.get("assetId", args.get("imageId"))
                            ],
                        }
                        embedding["sprite"]["imagePath"] = (
                            "/api/asset/download?assetId={assetId}&experimentKey={experimentKey}".format(
                                **new_args
                            )
                        )
            binary_io = io.BytesIO(json.dumps(em_json).encode())
            result = self._log_asset_filename(
                experiment,
                asset_type,
                metadata,
                binary_io,
                step,
                log_as_filename or log_filename,
            )
            asset_map[old_asset_id] = result["assetId"]
        elif asset_type == "datagrid":
            self.update_datagrid_contents(
                experiment,
                asset_map,
                metadata,
                step,
                filename,
                log_as_filename or log_filename,
                old_asset_id,
            )
        elif asset_type == "confusion-matrix":
            # This will come after contained assets
            with open(filename) as fp:
                cm_json = json.load(fp)
            # go though JSON, replace asset_ids with new asset_ids
            for row in cm_json["sampleMatrix"]:
                if row:
                    for cols in row:
                        if cols:
                            for cell in cols:
                                if cell and isinstance(cell, dict):
                                    old_cell_asset_id = cell["assetId"]
                                    new_cell_asset_id = asset_map[old_cell_asset_id]
                                    cell["assetId"] = new_cell_asset_id

            binary_io = io.BytesIO(json.dumps(cm_json).encode())
            result = self._log_asset_filename(
                experiment,
                asset_type,
                metadata,
                binary_io,
                step,
                log_as_filename or log_filename,
            )
            asset_map[old_asset_id] = result["assetId"]
        elif asset_type == "video":
            name = os.path.basename(filename)
            binary_io = open(filename, "rb")
            result = experiment.log_video(
                binary_io, name=log_as_filename or name, step=step, epoch=epoch
            )  # done!
            asset_map[old_asset_id] = result["assetId"]
        elif asset_type == "model-element":
            name = os.path.basename(filename)
            result = experiment.log_model(name, filename)
            asset_map[old_asset_id] = result["assetId"]
        else:
            result = self._log_asset_filename(
                experiment,
                asset_type,
                metadata,
                filename,
                step,
                log_as_filename or log_filename,
            )
            asset_map[old_asset_id] = result["assetId"]

    def log_assets(self, experiment, path, assets_metadata):
        if not self.quiet:
            with experiment.context_manager("ignore"):
                print("log_assets...")
        # Create mapping from old asset id to new asset id
        asset_map = {}
        # Process all of the non-nested assets first:
        for log_filename in assets_metadata:
            asset_type = assets_metadata[log_filename].get("type", "asset") or "asset"
            if asset_type not in ["confusion-matrix", "embeddings", "datagrid"]:
                if (
                    "remote" in assets_metadata[log_filename]
                    and assets_metadata[log_filename]["remote"]
                ):
                    asset = assets_metadata[log_filename]
                    experiment.log_remote_asset(
                        uri=asset["link"],
                        remote_file_name=asset["fileName"],
                        step=asset["step"],
                        metadata=asset["metadata"],
                    )
                else:
                    self._log_asset(
                        experiment,
                        path,
                        asset_type,
                        log_filename,
                        assets_metadata,
                        asset_map,
                    )
        # Process all nested assets:
        for log_filename in assets_metadata:
            asset_type = assets_metadata[log_filename].get("type", "asset") or "asset"
            if asset_type in ["confusion-matrix", "embeddings", "datagrid"]:
                self._log_asset(
                    experiment,
                    path,
                    asset_type,
                    log_filename,
                    assets_metadata,
                    asset_map,
                )

    def log_code(self, experiment, filename):
        """ """
        if not self.quiet:
            with experiment.context_manager("ignore"):
                print("log_code...")
        if os.path.exists(filename):
            if os.path.isfile(filename):
                experiment.log_code(str(filename))
            elif os.path.isdir(filename):
                experiment.log_code(folder=str(filename))

    def log_requirements(self, experiment, filename):
        """
        Requirements (pip packages)
        """
        if not self.quiet:
            with experiment.context_manager("ignore"):
                print("log_requirements...")
        if os.path.exists(filename):
            installed_packages_list = [package.strip() for package in open(filename)]
            if installed_packages_list is None:
                return
            message = InstalledPackagesMessage(
                installed_packages=installed_packages_list,
            )
            experiment._enqueue_message(message)

    def log_metrics(self, experiment, filename):
        """ """
        if os.path.exists(filename):
            if not self.quiet:
                with experiment.context_manager("ignore"):
                    print("log_metrics %s..." % filename)

            for line in open(filename):
                dict_line = json.loads(line)
                name = dict_line["metricName"]
                if name.startswith("sys.") and "system-metrics" in self.ignore:
                    continue
                value = dict_line.get("metricValue", None)
                if value is None:
                    continue
                step = dict_line.get("step", None)
                epoch = dict_line.get("epoch", None)
                context = dict_line.get("runContext", None)
                timestamp = dict_line.get("timestamp", None)
                message = MetricMessage(
                    context=context,
                    timestamp=timestamp,
                )
                message.set_metric(name, value, step=step, epoch=epoch)
                experiment._enqueue_message(message)

    def log_metrics_split(self, experiment, folder):
        """ """
        summary_filename = os.path.join(folder, "metrics_summary.jsonl")
        if os.path.exists(summary_filename):
            if not self.quiet:
                with experiment.context_manager("ignore"):
                    print("log_metrics from %s..." % summary_filename)

            for line in open(summary_filename):
                metric_summary = json.loads(line)
                self.log_metrics(
                    experiment,
                    os.path.join(
                        folder, "metrics", "metric_%05d.jsonl" % metric_summary["count"]
                    ),
                )

    def _prepare_parameter_value(self, value):
        if isinstance(value, list):
            return str(value)
        else:
            return value

    def log_parameters(self, experiment, filename):
        """ """
        if not self.quiet:
            with experiment.context_manager("ignore"):
                print("log_parameters...")
        if os.path.exists(filename):
            parameters = json.load(open(filename))
            parameter_dictionary = {
                parameter["name"]: self._prepare_parameter_value(
                    parameter["valueCurrent"]
                )
                for parameter in parameters
            }
            experiment.log_parameters(parameter_dictionary, nested_support=True)

    def log_others(self, experiment, filename):
        """ """
        if not self.quiet:
            with experiment.context_manager("ignore"):
                print("log_others...")
        if os.path.exists(filename):
            for line in open(filename):
                dict_line = json.loads(line)
                name = dict_line["name"]
                value = dict_line["valueCurrent"]
                experiment.log_other(key=name, value=value)

    def log_output(self, experiment, output_file):
        """ """
        if not self.quiet:
            with experiment.context_manager("ignore"):
                print("log_output...")
        if os.path.exists(output_file):
            for line in open(output_file):
                message = StandardOutputMessage(
                    output=line,
                    stderr=False,
                )
                experiment._enqueue_message(message)

    def log_html(self, experiment, filename):
        if not self.quiet:
            with experiment.context_manager("ignore"):
                print("log_html...")
        if os.path.exists(filename):
            html = open(filename).read()
            message = HtmlMessage(
                html=html,
            )
            experiment._enqueue_message(message)

    def log_git_metadata(self, experiment, filename):
        if os.path.exists(filename):
            with open(filename) as fp:
                metadata = json.load(fp)

            git_metadata = {
                "parent": metadata.get("parent", None),
                "repo_name": None,
                "status": None,
                "user": metadata.get("user", None),
                "root": metadata.get("root", None),
                "branch": metadata.get("branch", None),
                "origin": metadata.get("origin", None),
            }
            message = GitMetadataMessage(
                git_metadata=git_metadata,
            )
            experiment._enqueue_message(message)

    def log_git_patch(self, experiment, filename):
        if os.path.exists(filename):
            with open(filename) as fp:
                git_patch = fp.read()

            _, zip_path = compress_git_patch(git_patch)
            processor = GitPatchUploadProcessor(
                TemporaryFilePath(zip_path),
                experiment.asset_upload_limit,
                url_params=None,
                metadata=None,
                copy_to_tmp=False,
                error_message_identifier=None,
                tmp_dir=experiment.tmpdir,
                critical=False,
            )
            upload_message = processor.process()
            if upload_message:
                experiment._enqueue_message(upload_message)

    def log_all(self, experiment, experiment_folder):
        """ """
        # FIXME: missing notes (edited by human, not logged programmatically)
        if "metrics" not in self.ignore:
            # All together, in one file:
            self.log_metrics(
                experiment, os.path.join(experiment_folder, "metrics.jsonl")
            )
            # In separate files:
            self.log_metrics_split(experiment, experiment_folder)

        if "metadata" not in self.ignore:
            self.log_metadata(
                experiment, os.path.join(experiment_folder, "metadata.json")
            )

        if "parameters" not in self.ignore:
            self.log_parameters(
                experiment, os.path.join(experiment_folder, "parameters.json")
            )

        if "others" not in self.ignore:
            self.log_others(experiment, os.path.join(experiment_folder, "others.jsonl"))

        if "assets" not in self.ignore:
            assets_metadata_filename = os.path.join(
                experiment_folder, "assets", "assets_metadata.jsonl"
            )
            assets_metadata = {}
            if os.path.exists(assets_metadata_filename):
                for line in open(assets_metadata_filename):
                    data = json.loads(line)
                    assets_metadata[data["fileName"]] = data

                self.log_assets(
                    experiment,
                    os.path.join(experiment_folder, "assets"),
                    assets_metadata,
                )

        if "output" not in self.ignore:
            self.log_output(
                experiment, os.path.join(experiment_folder, "run/output.txt")
            )

        if "requirements" not in self.ignore:
            self.log_requirements(
                experiment, os.path.join(experiment_folder, "run/requirements.txt")
            )

        if "model-graph" not in self.ignore:
            self.log_graph(
                experiment, os.path.join(experiment_folder, "run/graph_definition.txt")
            )

        if "html" not in self.ignore:
            # NOTE: also logged as html asset
            html_filenames = os.path.join(experiment_folder, "assets", "html", "*")
            for html_filename in glob.glob(html_filenames):
                self.log_html(experiment, html_filename)
            # Deprecated:
            self.log_html(
                experiment,
                os.path.join(experiment_folder, "experiment.html"),
            )

        if "system-details" not in self.ignore:
            self.log_system_details(
                experiment, os.path.join(experiment_folder, "system_details.json")
            )

        if "git" not in self.ignore:
            self.log_git_metadata(
                experiment, os.path.join(experiment_folder, "run", "git_metadata.json")
            )
            self.log_git_patch(
                experiment, os.path.join(experiment_folder, "run", "git_diff.patch")
            )

        if "code" not in self.ignore:
            code_folder = os.path.join(experiment_folder, "run", "code")
            self.log_code(experiment, code_folder)
            # Deprecated:
            self.log_code(
                experiment, os.path.join(experiment_folder, "run", "script.py")
            )


def main(args):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    get_parser_arguments(parser)
    parsed_args = parser.parse_args(args)
    copy(parsed_args)


if __name__ == "__main__":
    # Called via `python -m cometx.cli.copy ...`
    main(sys.argv[1:])
