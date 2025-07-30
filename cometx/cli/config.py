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

To config Comet special settings

$ cometx config --auto-log-notebook yes
$ cometx config --auto-log-notebook no

"""
import argparse
import os
import shutil
import sys

ADDITIONAL_ARGS = False
SNIPPET = '''# Begin Comet Integration
# ------------------------------------------------
# Assumes Comet config in env, or in .comet.config
# Requires experiment output summary in cell
# Will log notebook to all experiments
# ------------------------------------------------
import re
try:
    import comet_ml
except ImportError:
    comet_ml = None
    print("ERROR: comet_ml is not available for saving notebook")

EXPERIMENT_KEYS = set()

def pre_save_hook(model, path, contents_manager, **kwargs):
    """
    Grab all of the experment keys from cells in memory.
    NOTE: notebook has not yet been saved here.
    """
    global EXPERIMENT_KEYS
    if model['type'] != 'notebook' or comet_ml is None:
        return
    EXPERIMENT_KEYS.clear()
    # get the experiment keys mentioned in notebook:
    for cell in model["content"]["cells"]:
        if cell["cell_type"] == "code":
            for output in cell["outputs"]:
                if output["output_type"] == "stream" and output["name"] == "stderr":
                    for match in re.findall(r".*https?://(\\S+)", output["text"]):
                        try:
                            root_url, workspace, project, experiment_key = match.split("/")
                            EXPERIMENT_KEYS.add(experiment_key)
                        except Exception:
                            pass # a URL that is not a Comet URL

def post_save_hook(model, os_path, contents_manager):
    """
    Log notebook to all of the experiment keys found.
    We have to do this after the file has been saved.
    NOTE: model does not have contents here.
    """
    global EXPERIMENT_KEYS
    if comet_ml is None:
        return
    for experiment_key in EXPERIMENT_KEYS:
        try:
            experiment = comet_ml.ExistingExperiment(previous_experiment=experiment_key)
            experiment.log_notebook(os_path, overwrite=True)
            experiment.end()
        except Exception:
            print("ERROR in logging experiment")

c.FileContentsManager.pre_save_hook = pre_save_hook
c.FileContentsManager.post_save_hook = post_save_hook
# ------------------------------------------------
# End Comet Integration
'''


def get_parser_arguments(parser):
    parser.add_argument(
        "--debug", help="If given, allow debugging", default=False, action="store_true"
    )
    parser.add_argument(
        "--auto-log-notebook",
        help="Takes a 1/yes/true, or 0/no/false",
        type=str,
    )


def remove_comet_section(filename):
    orig_filename = filename
    updated_filename = filename + ".new"
    with open(orig_filename) as orig_fp:
        with open(updated_filename, "w") as new_fp:
            state = None
            for line in orig_fp:
                if state is None:
                    if line.startswith("# Begin Comet Integration"):
                        state = "inside"
                    else:
                        new_fp.write(line)
                elif state == "inside":
                    if line.startswith("# End Comet Integration"):
                        state = None
                    # otherwise don't write it

    # If everything went ok, copy the new file over the orig
    shutil.copyfile(updated_filename, orig_filename)


def add_comet_section(filename):
    with open(filename, "a+") as fp:
        fp.write(SNIPPET)


def create_config():
    os.system("jupyter notebook --generate-config")


def config(parsed_args):
    if parsed_args.auto_log_notebook is not None:
        jupyter_config_dir = os.popen("jupyter --config-dir").read().strip()
        jupyter_config_filename = os.path.join(
            jupyter_config_dir, "jupyter_notebook_config.py"
        )
        if not os.path.exists(jupyter_config_filename):
            create_config()
        remove_comet_section(jupyter_config_filename)
        if parsed_args.auto_log_notebook.lower() in ["1", "true", "yes"]:
            add_comet_section(jupyter_config_filename)


def main(args):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    get_parser_arguments(parser)
    parsed_args = parser.parse_args(args)
    config(parsed_args)


if __name__ == "__main__":
    # Called via `python -m cometx.cli.delete ...`
    main(sys.argv[1:])
