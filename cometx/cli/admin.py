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
Perform admin functions for Comet.

Commands:
    chargeback-report
        Generate a chargeback report from the Comet server.

        Usage:
            cometx admin chargeback-report [YEAR-MONTH]

        Arguments:
            YEAR-MONTH (optional, deprecated)
                The YEAR-MONTH to run report for, eg 2024-09.
                If not provided, generates a report for all available periods.

        Output:
            Saves a JSON file: comet-chargeback-report.json (or comet-chargeback-report-{YEAR-MONTH}.json)

    usage-report
        Generate a usage report with experiment counts and statistics for one or more workspaces/projects.

        Usage:
            cometx admin usage-report WORKSPACE [WORKSPACE ...]
            cometx admin usage-report WORKSPACE/PROJECT [WORKSPACE/PROJECT ...]

        Arguments:
            WORKSPACE_PROJECT (required, one or more)
                One or more WORKSPACE or WORKSPACE/PROJECT to run usage report for.
                If WORKSPACE is provided without a project, all projects in that workspace will be included.

        Options:
            --units {month,week,day,hour}
                Time unit for grouping experiments (default: month)
                - month: Group by month (YYYY-MM format)
                - week: Group by ISO week (YYYY-WW format)
                - day: Group by day (YYYY-MM-DD format)
                - hour: Group by hour (YYYY-MM-DD-HH format)

            --max-experiments-per-chart N
                Maximum number of workspaces/projects per chart (default: 5).
                If more workspaces/projects are provided, multiple charts will be generated.

            --no-open
                Don't automatically open the generated PDF file after generation.

        Output:
            Generates a PDF report containing:
            - Summary statistics (total experiments, users, run times, GPU utilization)
            - Experiment count charts by time unit
            - GPU utilization charts (if GPU data is available)
            - GPU memory utilization charts (if GPU data is available)

Global Options (available for all commands):
    --api-key KEY
        Set the COMET_API_KEY for authentication.

    --url-override URL
        Set the COMET_URL_OVERRIDE for custom Comet server.

    --host URL
        Override the HOST URL.

    --debug
        Enable debug output for troubleshooting.

Examples:
    cometx admin chargeback-report
    cometx admin chargeback-report 2024-09
    cometx admin usage-report my-workspace
    cometx admin usage-report my-workspace/project1 my-workspace/project2
    cometx admin usage-report workspace1 workspace2 --units week
    cometx admin usage-report workspace --units day --no-open

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
    chargeback_description = """Generate a chargeback report from the Comet server.

Arguments:
    YEAR-MONTH (optional, deprecated)
        The YEAR-MONTH to run report for, eg 2024-09.
        If not provided, generates a report for all available periods.

Output:
    Saves a JSON file: comet-chargeback-report.json (or comet-chargeback-report-{YEAR-MONTH}.json)

Examples:
    cometx admin chargeback-report
    cometx admin chargeback-report 2024-09
"""
    chargeback_parser = subparsers.add_parser(
        "chargeback-report",
        help="Generate a chargeback report",
        description=chargeback_description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
    usage_report_description = """Generate a usage report with experiment counts and statistics for one or more workspaces/projects.

Arguments:
    WORKSPACE_PROJECT (optional, one or more)
        One or more WORKSPACE or WORKSPACE/PROJECT to run usage report for.
        If WORKSPACE is provided without a project, all projects in that workspace will be included.
        Not needed when using --app flag.

Options:
    --app
        Launch interactive Streamlit web app instead of generating PDF.

Output:
    Generates a PDF report containing:
    - Summary statistics (total experiments, users, run times, GPU utilization)
    - Experiment count charts by time unit
    - GPU utilization charts (if GPU data is available)
    - GPU memory utilization charts (if GPU data is available)

    With --app, launches an interactive web interface where you can:
    - Select workspace and project from dropdowns
    - View statistics and charts interactively
    - Change time units and regenerate reports

Examples:
    cometx admin usage-report my-workspace
    cometx admin usage-report my-workspace/project1 my-workspace/project2
    cometx admin usage-report workspace1 workspace2 --units week
    cometx admin usage-report workspace --units day --no-open
    cometx admin usage-report --app
"""
    usage_parser = subparsers.add_parser(
        "usage-report",
        help="Generate a usage report for one or more workspaces/projects",
        description=usage_report_description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Add global arguments to subparser so they show in help
    add_global_arguments(usage_parser)
    usage_parser.add_argument(
        "WORKSPACE_PROJECT",
        nargs="*",
        help="One or more WORKSPACE or WORKSPACE/PROJECT to run usage report for (not needed with --app)",
        metavar="WORKSPACE",
        type=str,
    )
    usage_parser.add_argument(
        "--app",
        help="Launch interactive Streamlit web app instead of generating PDF",
        default=False,
        action="store_true",
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
        default=None,
        metavar="N",
    )
    usage_parser.add_argument(
        "--units",
        help="Time unit for grouping experiments (default: month)",
        choices=["month", "week", "day", "hour"],
        default="month",
        type=str,
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
            if parsed_args.app:
                # Launch Streamlit app
                import os
                import sys

                # Set environment variables if --api-key or --url-override were provided
                if parsed_args.api_key:
                    os.environ["COMET_API_KEY"] = parsed_args.api_key
                if parsed_args.url_override:
                    os.environ["COMET_URL_OVERRIDE"] = parsed_args.url_override

                # Run the Streamlit app using streamlit's CLI
                try:
                    import streamlit.web.cli as stcli

                    # Get the path to admin_app.py
                    admin_app_path = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), "admin_app.py"
                    )

                    # Launch streamlit with the app
                    sys.argv = ["streamlit", "run", admin_app_path]
                    stcli.main()
                except Exception as e:
                    print(f"ERROR launching Streamlit app: {e}")
                    if parsed_args.debug:
                        raise
                    return
            else:
                # Generate PDF report
                workspace_projects = parsed_args.WORKSPACE_PROJECT

                if not workspace_projects:
                    print(
                        "ERROR: At least one workspace/project is required when not using --app"
                    )
                    return

                try:
                    generate_usage_report(
                        api,
                        workspace_projects,
                        no_open=parsed_args.no_open,
                        max_datasets_per_chart=parsed_args.max_experiments_per_chart,
                        units=parsed_args.units,
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
