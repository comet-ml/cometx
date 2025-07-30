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

import argparse
import os
import sys

from cometx.framework.comet.download_manager import DownloadManager, clean_comet_path

ADDITIONAL_ARGS = False


def get_parser_arguments(parser):
    parser.add_argument(
        "COMET_PATH",
        help=(
            "The Comet identifier, such as 'WORKSPACE', 'WORKSPACE/PROJECT', or "
            + "'WORKSPACE/PROJECT/EXPERIMENT'. Leave empty for all workspaces."
        ),
        type=str,
    )
    parser.add_argument(
        "OUTPUT_DIR",
        help=("The directory to copy files in order to reproduce the experiment"),
        type=str,
    )
    parser.add_argument(
        "--run",
        help=("Run the reproducable script"),
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--executable",
        help=("Run the reproducable script"),
        type=str,
        default="python",
    )


def reproduce(parsed_args, remaining=None):
    comet_path = (
        parsed_args.COMET_PATH.split("/") if parsed_args.COMET_PATH is not None else []
    )

    if len(comet_path) != 3:
        raise Exception(
            "invalid COMET_PATH: %r; requires workspace/project/experiment"
            % parsed_args.COMET_PATH
        )
    else:
        comet_path = clean_comet_path("/".join(comet_path))

    manager = DownloadManager()
    manager.root = parsed_args.OUTPUT_DIR
    manager.debug = False
    manager.use_name = False
    manager.flat = True
    manager.force = False
    manager.filename = None
    manager.overwrite = True
    manager.summary = {key: 0 for key in manager.RESOURCE_FUNCTIONS.keys()}
    manager.summary["artifacts"] = 0
    manager.summary["model-registry"] = 0

    experiment = manager.api.get(comet_path)
    if experiment is None:
        raise Exception(
            "invalid experiment: %r; not found, or not available" % comet_path
        )

    os.makedirs(parsed_args.OUTPUT_DIR, exist_ok=True)
    manager.download_code(experiment)
    manager.download_git(experiment)
    manager.download_requirements(experiment)
    for asset_filename in ["conda-spec.txt", "conda-info.yml", "conda-environment.yml"]:
        manager.download_asset(experiment, asset_filename)

    shell_commands = ""
    # "cd %s\n" % os.path.abspath(parsed_args.OUTPUT_DIR)
    if os.path.exists(os.path.join(parsed_args.OUTPUT_DIR, "conda-spec.txt")):
        shell_commands += "conda create --name reproduced-env --file conda-spec.txt\n"
        shell_commands += "conda activate reproduced-env\n"
    if os.path.exists(os.path.join(parsed_args.OUTPUT_DIR, "requirements.txt")):
        shell_commands += "pip install -r requirements.txt\n"
    if os.path.exists(os.path.join(parsed_args.OUTPUT_DIR, "git_diff.patch")):
        shell_commands += "git apply git_diff.patch\n"
    if os.path.exists(os.path.join(parsed_args.OUTPUT_DIR, "git_metadata.json")):
        shell_commands += manager.get_git_text(experiment)
        script = "../script.py"
    else:
        script = "script.py"

    shell_commands += "%s %s\n" % (parsed_args.executable, script)

    shell_script_name = os.path.join(parsed_args.OUTPUT_DIR, "script.sh")
    with open(shell_script_name, "w") as fp:
        fp.write(shell_commands)

    os.system("chmod +x %s" % shell_script_name)

    print("Shell command saved in: %s" % shell_script_name)
    if parsed_args.run:
        print("Running...")
        os.system("cd %s; %s" % (parsed_args.OUTPUT_DIR, shell_script_name))
    else:
        print("To run, cd into %s and execute the script." % parsed_args.OUTPUT_DIR)


def main(args):
    # Called via `cometx reproduce ...`
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    get_parser_arguments(parser)
    parsed_args = parser.parse_args(args)

    reproduce(parsed_args)


if __name__ == "__main__":
    # Called via `python -m cometx.cli.reproduce ...`
    # Called via `cometx reproduce ...`
    main(sys.argv[1:])
