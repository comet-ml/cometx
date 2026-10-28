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
Upload a Python panel file to a Comet dashboard or to the Panel library.

To a dashboard: a Python panel with the same name on that dashboard has its
code replaced; otherwise the panel is added to the dashboard's "Panel Studio"
section. The dashboard is created if it doesn't exist.

To the Panel library: a library panel with the same name is replaced.

The panel's name defaults to the file's name.

Examples:

    cometx panel-upload loss_curves.py --project PROJECT --dashboard "Training"
    cometx panel-upload loss_curves.py --project PROJECT --dashboard "Training" \\
        --name "Loss curves"
    cometx panel-upload loss_curves.py --library
"""
import argparse
import os
import sys

from ..api import API
from ..panel_studio import (
    dashboard_url,
    resolve_workspace,
    upload_to_dashboard,
    upload_to_library,
)
from .panel_studio import add_credential_arguments, apply_credentials

ADDITIONAL_ARGS = False


def get_parser_arguments(parser):
    parser.add_argument("FILE", help="the panel's Python file")
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument(
        "--dashboard", help="the project dashboard to put the panel on"
    )
    destination.add_argument(
        "--library",
        action="store_true",
        help="save the panel to the workspace's Panel library instead",
    )
    parser.add_argument(
        "--workspace", help="defaults to your configured or default workspace"
    )
    parser.add_argument("--project", help="the dashboard's project")
    parser.add_argument("--name", help="the panel's name; defaults to the file's name")
    add_credential_arguments(parser)


def panel_upload(parsed_args, remaining=None):
    apply_credentials(parsed_args)
    try:
        with open(parsed_args.FILE) as fp:
            code = fp.read()
    except OSError as error:
        sys.exit("Could not read %s: %s" % (parsed_args.FILE, error))
    if parsed_args.dashboard is not None and not parsed_args.project:
        sys.exit("--project is required with --dashboard.")
    if parsed_args.dashboard is not None and not parsed_args.dashboard.strip():
        sys.exit("--dashboard needs a name.")

    name = parsed_args.name or os.path.splitext(os.path.basename(parsed_args.FILE))[0]
    # Uncached: an upload reads the dashboard or library and writes it back.
    api = API(cache=False)
    workspace = resolve_workspace(api, parsed_args.workspace)
    try:
        if parsed_args.library:
            template_id, created = upload_to_library(api, workspace, name, code)
            print(
                "%s %r in the Panel library of %s (%s)."
                % ("Added" if created else "Updated", name, workspace, template_id)
            )
        else:
            view, created = upload_to_dashboard(
                api, workspace, parsed_args.project, parsed_args.dashboard, name, code
            )
            print(
                "%s %r on dashboard %r: %s"
                % (
                    "Added" if created else "Updated",
                    name,
                    parsed_args.dashboard,
                    dashboard_url(
                        api.server_url, workspace, parsed_args.project, view.template_id
                    ),
                )
            )
    except ValueError as error:
        sys.exit(str(error))


def main(args):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    get_parser_arguments(parser)
    panel_upload(parser.parse_args(args))


if __name__ == "__main__":
    # Called via `python -m cometx.cli.panel_upload ...`
    main(sys.argv[1:])
