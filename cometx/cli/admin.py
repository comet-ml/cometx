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
To perform admin functions

cometx admin chargeback-report
cometx admin usage-report WORKSPACE [WORKSPACE ...]
cometx admin usage-report WORKSPACE/PROJECT [WORKSPACE/PROJECT ...]

"""

import argparse
import json
import sys
from urllib.parse import urlparse

from comet_ml import API

from .admin_usage_report import generate_usage_report

ADDITIONAL_ARGS = False


def add_global_arguments(parser):
    """Add global arguments that are available for all commands."""
    parser.add_argument("--api-key", help="Set the COMET_API_KEY", type=str)
    parser.add_argument("--url-override", help="Set the COMET_URL_OVERRIDE", type=str)


def get_parser_arguments(parser):
    # Add global arguments to the main admin parser so they can appear anywhere in the command line
    # (e.g., "cometx admin --api-key KEY usage-report" or "cometx admin usage-report --api-key KEY")
    # The top-level parser will consume them first, but having them here ensures they're accepted
    # and shown in help if they appear after "admin"
    add_global_arguments(parser)

    # Add common arguments that apply to all admin subcommands
    parser.add_argument(
        "--host",
        help="Override the HOST URL",
        type=str,
    )
    parser.add_argument(
        "--debug", help="If given, allow debugging", default=False, action="store_true"
    )

    # Create subparsers for different admin actions
    subparsers = parser.add_subparsers(
        dest="ACTION",
        help="The admin action to perform",
        required=True,
    )

    # chargeback-report subcommand
    chargeback_parser = subparsers.add_parser(
        "chargeback-report",
        help="Generate a chargeback report",
    )
    # Add global arguments to subparser so they show in help
    add_global_arguments(chargeback_parser)
    chargeback_parser.add_argument(
        "YEAR_MONTH",
        nargs="?",
        help="(deprecated) The YEAR-MONTH to run report for, eg 2024-09",
        metavar="YEAR-MONTH",
        type=str,
        default=None,
    )

    # usage-report subcommand
    usage_parser = subparsers.add_parser(
        "usage-report",
        help="Generate a usage report for one or more workspaces/projects",
        description="Generate usage reports with experiment counts by month for one or more workspaces/projects.",
    )
    # Add global arguments to subparser so they show in help
    add_global_arguments(usage_parser)
    usage_parser.add_argument(
        "WORKSPACE_PROJECT",
        nargs="+",
        help="One or more WORKSPACE or WORKSPACE/PROJECT to run usage report for",
        metavar="WORKSPACE",
        type=str,
    )
    usage_parser.add_argument(
        "--no-open",
        help="Don't automatically open the generated PDF file",
        default=False,
        action="store_true",
    )
    usage_parser.add_argument(
        "--max-experiments-per-chart",
        help="Maximum number of workspaces/projects per chart (default: 5). If more are provided, multiple charts will be generated.",
        type=int,
        default=5,
        metavar="N",
    )


def admin(parsed_args, remaining=None):
    # Called via `cometx admin ...`
    try:
        api = API()

        if parsed_args.ACTION == "chargeback-report":
            if parsed_args.host is not None:
                admin_url = parsed_args.host
            else:
                url = api.config["comet.url_override"]
                result = urlparse(url)
                admin_url = "%s://%s" % (
                    result.scheme,
                    result.netloc,
                )

            while admin_url.endswith("/"):
                admin_url = admin_url[:-1]

            admin_url += "/api/admin/chargeback/report"

            print("Attempting to get chargeback report from %s..." % admin_url)
            if parsed_args.YEAR_MONTH:
                response = api._client.get(
                    admin_url + ("?reportMonth=%s" % parsed_args.YEAR_MONTH),
                    headers={"Authorization": api.api_key},
                    params={},
                )
                filename = "comet-chargeback-report-%s.json" % parsed_args.YEAR_MONTH
            else:
                response = api._client.get(
                    admin_url,
                    headers={"Authorization": api.api_key},
                    params={},
                )
                filename = "comet-chargeback-report.json"
            print("Attempting to save chargeback report...")
            with open(filename, "w") as fp:
                fp.write(json.dumps(response.json()))
            print("Chargeback report is saved in %r" % filename)
        elif parsed_args.ACTION == "usage-report":
            # WORKSPACE_PROJECT is now required (nargs="+") so we always have at least one
            workspace_projects = parsed_args.WORKSPACE_PROJECT

            try:
                generate_usage_report(
                    api,
                    workspace_projects,
                    no_open=parsed_args.no_open,
                    max_datasets_per_chart=parsed_args.max_experiments_per_chart,
                    debug=parsed_args.debug,
                )
            except Exception as e:
                print("ERROR: " + str(e))
                return

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
    admin(parsed_args)


if __name__ == "__main__":
    # Called via `python -m cometx.cli.admin ...`
    main(sys.argv[1:])
