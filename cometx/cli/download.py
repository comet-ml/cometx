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

To download experiments or experiment resources:

$ cometx download [RESOURCE ...] [FLAGS ...]
$ cometx download WORKSPACE [RESOURCE ...] [FLAGS ...]
$ cometx download WORKSPACE/PROJECT [RESOURCE ...] [FLAGS ...]
$ cometx download WORKSPACE/PROJECT/EXPERIMENT-KEY [RESOURCE ...] [FLAGS ...]

Where [RESOURCE ...] is zero or more of the following names:

* run - alias for: code, git, output, graph, and requirements
* system
* others
* parameters
* metadata
* metrics
* assets
* html
* project - alias for: project_notes, project_metadata

To download artifacts:

$ cometx download WORKSPACE/artifacts/NAME [FLAGS ...]
$ cometx download WORKSPACE/artifacts/NAME/VERSION-OR-ALIAS [FLAGS ...]

To download models from the registry:

$ cometx download WORKSPACE/model-registry/NAME [FLAGS ...]
$ cometx download WORKSPACE/model-registry/NAME/VERSION-OR-STAGE [FLAGS ...]

To download panels:

$ cometx download WORKSPACE/panels/NAME-OR-ID [FLAGS ...]
$ cometx download WORKSPACE/panels [FLAGS ...]

Where [FLAGS ...] is zero or more of the following:

* `--parallel N` - the number of threads to use (default is based on CPUs)
* `--query` - if given as a Comet query string, only download those
    experiments that match
* `--list` - use to list available workspaces, projects, experiments,
    artifacts, or models
* `--output` - download resources to folder other than current one
* `--flat` - don't use the normal hiearchy for downloaded items
* `--use-name` - use experiment names for folders and listings
* `--ignore` - don't download the following resources (use one or more
    RESOURCE names from above, or "experiments" to just get project info)
* `--asset-type` - asset type to match, or leave off to match all
* `--filename` - filename to match, or leave off to match all
* `--sync` - what level to to sync at: all, experiment, project, workspace
* `--ask` - ask before download
* `--help` - this message
"""

import argparse
import os
import sys

ADDITIONAL_ARGS = False


def get_parser_arguments(parser):
    parser.add_argument(
        "PATH",
        help=(
            "The source path, such as 'WORKSPACE', 'WORKSPACE/PROJECT', or "
            + "'WORKSPACE/PROJECT/EXPERIMENT'. Leave empty for all workspaces."
        ),
        nargs="?",
    )
    parser.add_argument(
        "RESOURCE",
        help=(
            "Resource(s) to include in download. For experiments, any of: system, run, code, git, "
            + "output, requirements, others, parameters, metadata, metrics, output, assets, "
            + "project, project_notes, or project_metadata."
        ),
        nargs="*",
        default=[],
    )
    parser.add_argument(
        "--from",
        help="Source of data to copy. Should be: comet, wandb, or neptune",
        default="comet",
        dest="FROM",
        metavar="from",
    )
    parser.add_argument(
        "-i",
        "--ignore",
        help="Resource(s) (or 'experiments') to ignore.",
        nargs="+",
        default=[],
    )
    parser.add_argument(
        "-j",
        "--parallel",
        help="The number of threads to use for parallel downloading; default (None) is based on CPUs",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output directory for downloads.",
        type=str,
    )
    parser.add_argument(
        "-u",
        "--use-name",
        help="Use experiment names for experiment folders and listings",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-l",
        "--list",
        help="List the items at this level (workspace, project, experiment, artifacts, or model-registry) rather than download.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--flat",
        help="Download the files without subfolders.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-f",
        "--ask",
        help="Queries the user; if flag not included system will answer `yes` for all queries",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--filename",
        help="Only get resources ending with this",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--query",
        help="Only download experiments that match this Comet query string",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--asset-type",
        help="Only get assets with this type",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--sync",
        help="What level to sync at: all, experiment, project, or workspace",
        type=str,
        default="all",
    )
    parser.add_argument(
        "--debug",
        help="Provide debug info",
        action="store_true",
        default=False,
    )


def download(parsed_args, remaining=None):
    from comet_ml.exceptions import Unauthorized

    from ..framework.comet import DownloadManager
    from ..utils import display_invalid_api_key

    try:
        downloader = DownloadManager()
    except ValueError:
        display_invalid_api_key()
        return

    max_workers = (
        min(32, os.cpu_count() + 4)
        if parsed_args.parallel is None
        else parsed_args.parallel
    )

    if parsed_args.FROM == "comet":
        try:
            downloader.download(
                comet_path=parsed_args.PATH,
                include=parsed_args.RESOURCE,
                ignore=parsed_args.ignore,
                output=parsed_args.output,
                use_name=parsed_args.use_name,
                list_items=parsed_args.list,
                flat=parsed_args.flat,
                ask=parsed_args.ask,
                filename=parsed_args.filename,
                asset_type=parsed_args.asset_type,
                sync=parsed_args.sync,
                debug=parsed_args.debug,
                query=parsed_args.query,
                max_workers=max_workers,
            )
            print("Waiting on threaded downloads...")
            downloader.end()
        except Unauthorized:
            display_invalid_api_key()
        except Exception as exc:
            if parsed_args.debug:
                raise exc from None
            else:
                print("Download aborted: %s" % str(exc))
        except KeyboardInterrupt:
            print("User canceled download by keyboard interrupt")
    elif parsed_args.FROM == "wandb":
        from ..framework.wandb import DownloadManager

        dm = DownloadManager(
            include=parsed_args.RESOURCE,
            ignore=parsed_args.ignore,
            output=parsed_args.output,
            list_items=parsed_args.list,
            flat=parsed_args.flat,
            ask=parsed_args.ask,
            filename=parsed_args.filename,
            asset_type=parsed_args.asset_type,
            sync=parsed_args.sync,
            debug=parsed_args.debug,
            query=parsed_args.query,
            max_workers=max_workers,
        )

        try:
            dm.download(parsed_args.PATH)
            print("Waiting on threaded downloads...")
            dm.end()
        except Exception as exc:
            if parsed_args.debug:
                raise exc from None
            else:
                print("Download aborted: %s" % str(exc))
        except KeyboardInterrupt:
            print("User canceled download by keyboard interrupt")


def main(args):
    # Called via `cometx download ...`
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    get_parser_arguments(parser)
    parsed_args = parser.parse_args(args)

    download(parsed_args)


if __name__ == "__main__":
    # Called via `python -m cometx.cli.download ...`
    # Called via `cometx download ...`
    main(sys.argv[1:])
