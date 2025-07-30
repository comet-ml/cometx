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
#  Copyright (c) 2024 Cometx Development
#      Team. All rights reserved.
# ****************************************
"""
To update existing experiments.

cometx update SOURCE DESTINATION

where SOURCE is a folder:

* "WORKSPACE/PROJECT/EXPERIMENT"
* "WORKSPACE/PROJECT"
* "WORKSPACE"

where DESTINATION is a Comet:

* WORKSPACE
* WORKSPACE/PROJECT

"""

import argparse
import glob
import json
import os
import sys

from comet_ml import API

from ..utils import remove_extra_slashes

ADDITIONAL_ARGS = False


def get_parser_arguments(parser):
    parser.add_argument(
        "COMET_SOURCE",
        help=(
            "The folder containing the experiments to update: 'workspace', or 'workspace/project' or 'workspace/project/experiment'"
        ),
        type=str,
    )
    parser.add_argument(
        "COMET_DESTINATION",
        help=("The Comet destination: 'WORKSPACE', 'WORKSPACE/PROJECT'"),
        type=str,
    )
    parser.add_argument(
        "--debug", help="If given, allow debugging", default=False, action="store_true"
    )


def get_experiment_folders(workspace_src, project_src, experiment_src):
    for path in glob.iglob(f"{workspace_src}/{project_src}/{experiment_src}"):
        if any([path.endswith("~"), path.endswith(".json"), path.endswith(".jsonl")]):
            continue
        else:
            yield path


def update_experiments(source, destination):
    api = API()

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

    for experiment_folder in get_experiment_folders(
        workspace_src, project_src, experiment_src
    ):
        if experiment_folder.count("/") >= 2:
            folder_workspace, folder_project, folder_experiment = (
                experiment_folder
            ).rsplit("/", 2)
        else:
            print("Unknown folder: %r; ignoring" % experiment_folder)
            continue

        print("Updating from %r..." % experiment_folder)
        act_workspace, act_project, _ = experiment_folder.split("/")
        # First, get experiment name:
        with open(os.path.join(experiment_folder, "metadata.json")) as fp:
            metadata = json.load(fp)
        # Next, look it up in destination:
        print(
            "    Attempting to get %s/%s/%s - name: %s"
            % (
                api.server_url,
                workspace_dst,
                project_dst or act_project,
                metadata.get("experimentName"),
            )
        )
        experiment = api.get_experiment(
            workspace_dst, project_dst or act_project, metadata.get("experimentName")
        )
        # Finally, update data:
        if experiment:
            print("    Updating to %r..." % experiment.url)
            if metadata.get("startTimeMillis") or metadata.get("endTimeMillis"):
                experiment._api._client.set_experiment_start_end(
                    experiment.id,
                    metadata["startTimeMillis"],
                    metadata["endTimeMillis"],
                )
            else:
                print("    Now times found; not updating start/stop times")
            git_metadata_path = os.path.join(
                experiment_folder, "run", "git_metadata.json"
            )
            if os.path.exists(git_metadata_path):
                with open(git_metadata_path) as fp:
                    git_metadata = json.load(fp)
                experiment.set_git_metadata(**git_metadata)
                print("    done!")
            else:
                print("    no metadata found; not updating git metadata")
        else:
            print("    no experiment found; skipping")


def update(parsed_args, remaining=None):
    # Called via `cometx update ...`
    try:
        update_experiments(parsed_args.COMET_SOURCE, parsed_args.COMET_DESTINATION)
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


def main(args):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    get_parser_arguments(parser)
    parsed_args = parser.parse_args(args)
    update(parsed_args)


if __name__ == "__main__":
    # Called via `python -m cometx.cli.copy ...`
    main(sys.argv[1:])
