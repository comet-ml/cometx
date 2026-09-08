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

To delete experiments assets with optional query:

$ cometx delete-assets WORKSPACE/PROJECT --type=image --query='Other("Name") == "absolute_eagle_6153"'
$ cometx delete-assets WORKSPACE/PROJECT/EXPERIMENT --type=all

If `--query` is given as a Comet query string, only download those
experiments that match

Where TYPE is one of the following names:

* all
* asset
* audio
* code
* image
* notebook
* text-sample
* video
"""
import argparse
import sys

from comet_ml import API

from ..utils import get_query_experiments

ADDITIONAL_ARGS = False


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
        "--type",
        help="The type of item to log",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--debug", help="If given, allow debugging", default=False, action="store_true"
    )
    parser.add_argument(
        "--query",
        help="Only delete experiments that match this Comet query string",
        type=str,
        default=None,
    )


def delete_assets(parsed_args, remaining=None):
    # Called via `cometx delete-assets ...`
    try:
        delete_cli(parsed_args)
    except KeyboardInterrupt:
        print("Canceled by CONTROL+C")
    except Exception as exc:
        if parsed_args.debug:
            raise
        else:
            print("ERROR: " + str(exc))


def delete_cli(parsed_args):
    comet_path = (
        parsed_args.COMET_PATH.split("/") if parsed_args.COMET_PATH is not None else []
    )

    if not comet_path:
        workspace = None
        project_name = None
        experiment_key = None
    elif len(comet_path) == 1:
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

    if experiment_key:
        experiments = [api.get_experiment(workspace, project_name, experiment_key)]
    elif parsed_args.query is not None:
        experiments = get_query_experiments(
            api, parsed_args.query, workspace, project_name
        )
    else:
        experiments = api.get_experiments(workspace, project_name)

    delete_experiment_assets(api, experiments, parsed_args.type)


def delete_experiment_assets(api, experiments, asset_type):
    count = 0
    for experiment in experiments:
        print("Looking in %s..." % experiment.url)
        assets = experiment.get_asset_list(asset_type)
        for asset in assets:
            api._client.delete_experiment_asset(experiment.id, asset["assetId"])
            count += 1
    print("Deleted %d assets of type %r" % (count, asset_type))


def main(args):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    get_parser_arguments(parser)
    parsed_args = parser.parse_args(args)
    delete_assets(parsed_args)


if __name__ == "__main__":
    # Called via `python -m cometx.cli.delete ...`
    main(sys.argv[1:])
