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
#  Copyright (c) 2022 Cometx Development
#      Team. All rights reserved.
# ****************************************
"""
Examples:

To log to an experiment or set other key:value to multiple experiments:


$ cometx log WORKSPACE/PROJECT/EXPERIMENT-KEY FILENAME ... --type=TYPE
$ cometx log WORKSPACE PANEL-ZIP-FILENAME ... --type=panel
$ cometx log WORKSPACE PANEL.py ... --type=panel
$ cometx log WORKSPACE PANEL-URL ... --type=panel
$ cometx log WORKSPACE/PROJECT --type=other --set "key:value"
$ cometx log WORKSPACE --type=other --set "key:value"

Where TYPE is one of the following names:

* all
* asset
* audio
* code
* image
* metrics
* notebook
* panel
* tensorflow-file
* text-sample
* video
* other
* tensorboard-folder-assets
"""
import argparse
import glob
import json
import os
import sys

from comet_ml import ExistingExperiment

from ..api import API
from ..panel_utils import create_panel_zip
from ..utils import get_file_extension, get_query_experiments
from .utils import (
    log_points_3d_off_file,
    log_points_3d_pcd_file,
    log_points_3d_xyz_file,
    log_tensorboard_folder_assets,
)

ADDITIONAL_ARGS = False
# From filename extension to Comet Asset Type
EXTENSION_MAP = {
    "asset": "asset",
    "avi": "video",
    "datagrid": "datagrid",
    "gif": "image",
    "ipynb": "notebook",
    "jpg": "image",
    "mp3": "audio",
    "mp4": "video",
    "mpeg": "video",
    "mpg": "video",
    "mpv": "video",
    "ogg": "video",
    "ogv": "video",
    "png": "image",
    "txt": "text-sample",
    "wav": "audio",
    "webm": "video",
    #    "curve": "curve", FIXME: add
}
# Fom CLI type to Comet Asset Type
# List only those that differ from
# type.lower() != Comet Asset Type
TYPE_MAP = {
    "text": "text-sample",
}


def get_parser_arguments(parser):
    parser.add_argument(
        "COMET_PATH",
        help=(
            "The Comet identifier, such as 'WORKSPACE', 'WORKSPACE/PROJECT', or "
            + "'WORKSPACE/PROJECT/EXPERIMENT'. Leave empty to use defaults."
        ),
        nargs="?",
        default=None,
        type=str,
    )
    parser.add_argument(
        "FILENAME",
        help=("The filename(s) to log"),
        nargs="*",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--type",
        help="The type of item to log",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--set",
        help="The key:value to log",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--query",
        help="A Comet Query string, see https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/API/#apiquery",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--debug", help="If given, allow debugging", default=False, action="store_true"
    )
    parser.add_argument(
        "--use-base-name",
        help="If given, using the basename for logging assets",
        default=False,
        action="store_true",
    )


def log(parsed_args, remaining=None):
    # Called via `cometx log ...`
    try:
        log_cli(parsed_args)
    except KeyboardInterrupt:
        print("Canceled by CONTROL+C")
    except Exception as exc:
        if parsed_args.debug:
            raise
        else:
            print("ERROR: " + str(exc))


def log_cli(parsed_args):
    comet_path = (
        parsed_args.COMET_PATH.split("/") if parsed_args.COMET_PATH is not None else []
    )

    if len(comet_path) == 1:
        workspace = comet_path[0]
        project_name = None
        experiment_key = None
    elif len(comet_path) == 2:
        workspace, project_name = comet_path
        experiment_key = None
    elif len(comet_path) == 3:
        workspace, project_name, experiment_key = comet_path
    else:
        raise Exception("invalid COMET_PATH: %r" % parsed_args.COMET_PATH)

    api = API()

    if parsed_args.type in ["panel", "tensorboard-folder-assets"]:
        experiments = []
    elif experiment_key:
        experiments = [api.get_experiment(workspace, project_name, experiment_key)]
    elif parsed_args.query is not None:
        experiments = get_query_experiments(
            api, parsed_args.query, workspace, project_name
        )
    else:
        experiments = api.get_experiments(workspace, project_name)

    if parsed_args.type == "panel":
        for item in parsed_args.FILENAME:
            if item.startswith("http"):
                print("Uploading panel code from URL...")
                api.upload_panel_url(workspace, item)
            elif item.endswith(".zip"):
                print("Uploading panel zip...")
                api.upload_panel_zip(workspace, item)
            elif item.endswith(".py"):
                print("Reading contents of zip file...")
                with open(item) as fp:
                    code = fp.read()
                nice_name = os.path.splitext(os.path.basename(item))[0]
                print("Creating zipped code...")
                filename = create_panel_zip(nice_name, code)
                print("Uploading panel %r..." % nice_name)
                api.upload_panel_zip(workspace, filename)
            else:
                # Assume it is a well-known panel name
                api.upload_panel(workspace=workspace, name=item)

    elif parsed_args.type == "code":
        if not parsed_args.FILENAME:
            raise Exception("Logging `code` requires file(s) or folder(s)")

        for experiment in experiments:
            for filename in parsed_args.FILENAME:
                log_experiment_code_from_file(experiment, filename)

    elif parsed_args.type == "other":
        # two possibilities: log key:value to set of experiments; log filename to experiment
        if parsed_args.FILENAME:
            for experiment in experiments:
                for filename in parsed_args.FILENAME:
                    log_experiment_others_from_file(experiment, filename)
            return

        elif not parsed_args.set or ":" not in parsed_args.set:
            raise Exception("Logging `other` without FILENAME requires --set key:value")

        key, value = parsed_args.set.split(":", 1)
        set_experiments_other(experiments, key, value)

    elif parsed_args.type == "metrics":
        if not parsed_args.FILENAME:
            raise Exception("Logging `metrics` requires file(s)")

        for experiment in experiments:
            for filename in parsed_args.FILENAME:
                log_experiment_metrics_from_file(experiment, filename)

    elif parsed_args.type == "parameters":
        if not parsed_args.FILENAME:
            raise Exception("Logging `parameters` requires file(s)")

        for experiment in experiments:
            for filename in parsed_args.FILENAME:
                log_experiment_parameters_from_file(experiment, filename)

    elif parsed_args.type == "tensorboard-folder-assets":
        # Special case for now: creates experiments:
        for filename in parsed_args.FILENAME:
            log_tensorboard_folder_assets(workspace, project_name, filename)

    elif parsed_args.type == "tensorflow-file":
        if not parsed_args.FILENAME:
            raise Exception("Logging `tensorflow-file` requires folder")

        for experiment in experiments:
            ee = ExistingExperiment(previous_experiment=experiment.id)
            for filename in parsed_args.FILENAME:
                ee.log_tensorflow_folder(filename)
            ee.end()

    else:
        if not parsed_args.FILENAME:
            raise Exception("Logging an asset requires file(s)")

        for experiment in experiments:
            for filename in parsed_args.FILENAME:
                log_experiment_assets_from_file(
                    experiment, filename, parsed_args.type, parsed_args.use_base_name
                )

    for experiment in experiments:
        experiment.end()


def log_experiment_assets_from_file(experiment, filename, file_type, use_base_name):
    SKELETON = filename
    base_name = os.path.basename(filename) if use_base_name else filename
    for filename in glob.glob(SKELETON):
        extension = get_file_extension(filename).lower()
        # FIXME: These types do not respect use_base_name
        if extension == "off":
            log_points_3d_off_file(experiment, filename)
        elif extension == "pcd":
            log_points_3d_pcd_file(experiment, filename)
        elif extension == "xyz":
            log_points_3d_xyz_file(experiment, filename)
        else:
            if file_type is None or file_type in ["asset", ""]:
                file_type = EXTENSION_MAP.get(extension, "asset")

            # metadata = FIXME: get metadata dict from args
            experiment._api._client.log_experiment_asset(
                experiment.id,
                file_data=filename,
                file_name=base_name,
                ftype=file_type,
                step=None,
                overwrite=None,
                context=None,
                metadata=None,
            )


def log_experiment_code_from_file(experiment, filename):
    if os.path.isfile(filename):
        experiment.log_code(filename)
    elif os.path.isdir(filename):
        experiment.log_code(folder=filename)
    else:
        raise Exception("cannot log code: %r; use filename or folder" % filename)


def set_experiments_other(experiments, key, value):
    from ..generate_utils import generate_experiment_name

    for count, experiment in enumerate(experiments):
        new_value = value
        new_value = new_value.replace("{random}", generate_experiment_name())
        new_value = new_value.replace("{count}", str(count + 1))
        experiment.log_other(key, new_value)


def log_experiment_metrics_from_file(experiment, filename):
    for line in open(filename):
        dict_line = json.loads(line)
        name = dict_line["metricName"]
        value = dict_line["metricValue"]
        step = dict_line["step"]
        epoch = dict_line["epoch"]
        # FIXME: does not log time
        experiment.log_metric(name=name, value=value, step=step, epoch=epoch)


def log_experiment_parameters_from_file(experiment, filename):
    parameters = json.load(open(filename))
    for parameter in parameters:
        name = parameter["name"]
        value = parameter["valueCurrent"]
        experiment.log_parameter(name, value)


def log_experiment_others_from_file(experiment, filename):
    for line in open(filename):
        dict_line = json.loads(line)
        name = dict_line["name"]
        value = dict_line["valueCurrent"]
        experiment.log_other(key=name, value=value)


def main(args):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    get_parser_arguments(parser)
    parsed_args = parser.parse_args(args)
    log(parsed_args)


if __name__ == "__main__":
    # Called via `python -m cometx.cli.log ...`
    main(sys.argv[1:])
