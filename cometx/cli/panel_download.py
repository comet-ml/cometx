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
#  Copyright (c) 2026 Cometx Development
#      Team. All rights reserved.
# ****************************************
"""
Download a Python panel's code from a Comet view or the Panel library,
to edit it in your own editor and upload it again with `cometx panel-upload`.

From a view, --name may be left out when the view has a single Python panel.
The code is written to --output, or printed when there is none.

Examples:

    cometx panel-download --project PROJECT --view "Training" -o loss.py
    cometx panel-download --project PROJECT --view "Training" \\
        --name "Loss curves" -o loss.py
    cometx panel-download --library --name "Loss curves" -o loss.py
"""

import argparse
import os
import sys

from ..api import API
from ..panel_studio import library_panel_code, resolve_workspace, view_panel_code
from .panel_studio import add_credential_arguments, apply_credentials

ADDITIONAL_ARGS = False


def get_parser_arguments(parser):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--view", help="the project view the panel is on")
    # The option's name in cometx 3.6.12.
    source.add_argument("--dashboard", dest="view", help=argparse.SUPPRESS)
    source.add_argument(
        "--library",
        action="store_true",
        help="download from the workspace's Panel library instead",
    )
    parser.add_argument(
        "--workspace", help="defaults to your configured or default workspace"
    )
    parser.add_argument("--project", help="the view's project")
    parser.add_argument("--name", help="the panel's name")
    parser.add_argument(
        "-o", "--output", help="the file to write; the code is printed otherwise"
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite --output if it exists"
    )
    add_credential_arguments(parser)


def panel_download(parsed_args, remaining=None):
    apply_credentials(parsed_args)
    if parsed_args.view is not None and not parsed_args.project:
        sys.exit("--project is required with --view.")
    if parsed_args.library and not parsed_args.name:
        sys.exit("--name is required with --library.")
    output = parsed_args.output
    if output and os.path.exists(output) and not parsed_args.force:
        sys.exit("%s already exists; use --force to overwrite it." % output)

    api = API(cache=False)
    workspace = resolve_workspace(api, parsed_args.workspace)
    try:
        if parsed_args.library:
            name = parsed_args.name
            code = library_panel_code(api, workspace, name)
        else:
            name, code = view_panel_code(
                api,
                workspace,
                parsed_args.project,
                parsed_args.view,
                parsed_args.name,
            )
    except ValueError as error:
        sys.exit(str(error))

    if not output:
        sys.stdout.write(code)
        return
    with open(output, "w") as fp:
        fp.write(code)
    print("Wrote panel %r to %s" % (name, output), file=sys.stderr)


def main(args):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    get_parser_arguments(parser)
    panel_download(parser.parse_args(args))


if __name__ == "__main__":
    # Called via `python -m cometx.cli.panel_download ...`
    main(sys.argv[1:])
