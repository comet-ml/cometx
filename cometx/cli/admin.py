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
                Maximum number of workspaces/projects per chart (default: 100).
                If more workspaces/projects are provided, multiple charts will be generated.

            --no-open
                Don't automatically open the generated PDF file after generation.

        Output:
            Generates a PDF report containing:
            - Summary statistics (total experiments, users, run times, GPU utilization)
            - Experiment count charts by time unit
            - GPU utilization charts (if GPU data is available)
            - GPU memory utilization charts (if GPU data is available)

    gpu-report
        Generate a GPU usage report for one or more workspaces/projects.

        Usage:
            cometx admin gpu-report WORKSPACE [WORKSPACE ...] --start-date DATE
            cometx admin gpu-report WORKSPACE/PROJECT [WORKSPACE/PROJECT ...] --start-date DATE

        Arguments:
            WORKSPACE_PROJECT (required, one or more)
                One or more WORKSPACE or WORKSPACE/PROJECT to run GPU report for.
                If WORKSPACE is provided without a project, all projects in that workspace will be included.

        Options:
            --start-date DATE
                Start date for the report in YYYY-MM-DD format (required).

            --end-date DATE
                End date for the report in YYYY-MM-DD format (optional).
                If not provided, reports from start-date onwards.

            --metrics METRIC [METRIC ...]
                List of metrics to track (optional).
                If not provided, uses default GPU metrics.

        Output:
            Returns a dictionary of metrics keyed by experiment key.

    optimizer-report
        Generate a report for an optimizer instance.

        Usage:
            cometx admin optimizer-report OPTIMIZER_ID
            cometx admin optimizer-report OPTIMIZER_ID --app

        Arguments:
            OPTIMIZER_ID (required)
                The optimizer instance ID to generate a report for.

        Options:
            --app
                Launch interactive Streamlit web app instead of generating JSON file.

        Output:
            Without --app:
                Generates a JSON file containing all dashboard/data items:
                - optimizer-report-{OPTIMIZER_ID}.json

            With --app:
                Launches an interactive web interface where you can:
                - View optimizer dashboard with statistics
                - Filter by status
                - Navigate through paginated job assignments
                - View summary statistics

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
    cometx admin gpu-report my-workspace --start-date 2024-01-01
    cometx admin gpu-report my-workspace --start-date 2024-01-01 --end-date 2024-12-31
    cometx admin gpu-report workspace1/project1 workspace2 --start-date 2024-01-01 --metrics sys.gpu.0.gpu_utilization
    cometx admin optimizer-report abc123def456
    cometx admin optimizer-report abc123def456 --app

"""

import argparse
import json
import os
import sys
from urllib.parse import urlparse

from comet_ml import API

from .admin_gpu_report import main as gpu_report_main
from .admin_optimizer_report import generate_json_report
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
        help="Maximum number of workspaces/projects per chart (default: 100). If more are provided, multiple charts will be generated.",
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

    # gpu-report subcommand
    gpu_report_description = """Generate a GPU usage report for one or more workspaces/projects.

Arguments:
    WORKSPACE_PROJECT (required, one or more)
        One or more WORKSPACE or WORKSPACE/PROJECT to run GPU report for.
        If WORKSPACE is provided without a project, all projects in that workspace will be included.

Options:
    --start-date DATE
        Start date for the report in YYYY-MM-DD format (optional, collects all data if not provided).

    --end-date DATE
        End date for the report in YYYY-MM-DD format (optional).
        If not provided, reports from start-date onwards.

    --metrics METRIC [METRIC ...]
        List of metrics to track (optional).
        If not provided, uses default GPU metrics:
        - sys.gpu.0.gpu_utilization
        - sys.gpu.0.memory_utilization
        - sys.gpu.0.used_memory
        - sys.gpu.0.power_usage
        - sys.gpu.0.temperature

    --open
        Automatically open the generated PDF file after generation.

    --app
        Launch interactive Streamlit web app instead of generating PDF.
        The JSON file will be generated first and automatically loaded in the app.
        The app allows you to interactively change aggregation (workspace, project, user),
        time units (year, month, week, day), and date ranges.

Output:
    Without --app:
        Generates a PDF report and JSON file containing:
        - Summary statistics (total experiments, workspaces, metrics tracked)
        - Average metrics by workspace charts
        - Maximum metrics by month charts
        - JSON file: gpu_report_{workspace_projects}.json

    With --app:
        Launches an interactive web interface where you can:
        - Change aggregation (workspace, project, user)
        - Change time units (year, month, week, day)
        - Select start and end dates from the data
        - View charts for each metric

Examples:
    cometx admin gpu-report my-workspace
    cometx admin gpu-report my-workspace --start-date 2024-01-01
    cometx admin gpu-report my-workspace --start-date 2024-01-01 --end-date 2024-12-31
    cometx admin gpu-report workspace1/project1 workspace2
    cometx admin gpu-report my-workspace --metrics sys.gpu.0.gpu_utilization sys.gpu.0.memory_utilization
    cometx admin gpu-report my-workspace --open
    cometx admin gpu-report my-workspace --app
"""
    gpu_parser = subparsers.add_parser(
        "gpu-report",
        help="Generate a GPU usage report for one or more workspaces/projects",
        description=gpu_report_description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Add global arguments to subparser so they show in help
    add_global_arguments(gpu_parser)
    gpu_parser.add_argument(
        "WORKSPACE_PROJECT",
        nargs="*",
        help="One or more WORKSPACE or WORKSPACE/PROJECT to run GPU report for",
        metavar="WORKSPACE",
        type=str,
    )
    gpu_parser.add_argument(
        "--start-date",
        help="Start date for the report in YYYY-MM-DD format (optional, collects all data if not provided)",
        type=str,
        required=False,
    )
    gpu_parser.add_argument(
        "--end-date",
        help="End date for the report in YYYY-MM-DD format (optional)",
        type=str,
        default=None,
    )
    gpu_parser.add_argument(
        "--metrics",
        help="List of metrics to track (optional, uses defaults if not provided)",
        nargs="+",
        type=str,
        default=None,
    )
    gpu_parser.add_argument(
        "--open",
        help="Automatically open the generated PDF file after generation",
        default=False,
        action="store_true",
    )
    gpu_parser.add_argument(
        "--app",
        help="Launch interactive Streamlit web app instead of generating PDF",
        default=False,
        action="store_true",
    )

    # optimizer-report subcommand
    optimizer_report_description = """Generate a report for an optimizer instance.

Arguments:
    OPTIMIZER_ID (required)
        The optimizer instance ID to generate a report for.

Options:
    --app
        Launch interactive Streamlit web app instead of generating JSON file.

Output:
    Without --app:
        Generates a JSON file containing all dashboard/data items:
        - optimizer-report-{OPTIMIZER_ID}.json

    With --app:
        Launches an interactive web interface where you can:
        - View optimizer dashboard with statistics
        - Filter by status
        - Navigate through paginated job assignments
        - View summary statistics

Examples:
    cometx admin optimizer-report abc123def456
    cometx admin optimizer-report abc123def456 --app
"""
    optimizer_parser = subparsers.add_parser(
        "optimizer-report",
        help="Generate a report for an optimizer instance",
        description=optimizer_report_description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Add global arguments to subparser so they show in help
    add_global_arguments(optimizer_parser)
    optimizer_parser.add_argument(
        "OPTIMIZER_ID",
        help="The optimizer instance ID to generate a report for",
        metavar="OPTIMIZER_ID",
        type=str,
    )
    optimizer_parser.add_argument(
        "--app",
        help="Launch interactive Streamlit web app instead of generating JSON file",
        default=False,
        action="store_true",
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
        elif parsed_args.ACTION == "gpu-report":
            workspace_projects = parsed_args.WORKSPACE_PROJECT or []
            start_date = parsed_args.start_date
            end_date = parsed_args.end_date
            metrics = parsed_args.metrics

            if parsed_args.app:
                # Require workspace_projects for --app
                if not workspace_projects:
                    print(
                        "ERROR: At least one workspace/project is required when using --app"
                    )
                    return

                # Generate JSON first
                json_file_path = None
                try:
                    print("Generating GPU report data...")
                    result = gpu_report_main(
                        workspace_projects=workspace_projects,
                        start_date=start_date,
                        end_date=end_date,
                        metrics=metrics,
                        max_workers=None,  # Use default
                    )
                    if result:
                        json_file_path = result.get("json_file")
                        if json_file_path:
                            print(f"JSON report saved: {json_file_path}")
                        else:
                            print("ERROR: JSON file was not created")
                            return
                    else:
                        print("ERROR: Failed to generate GPU report data")
                        return
                except Exception as e:
                    print(f"ERROR: Could not generate JSON file: {e}")
                    if parsed_args.debug:
                        import traceback

                        traceback.print_exc()
                    return

                # Launch Streamlit app
                # Set environment variables if --api-key or --url-override were provided
                if parsed_args.api_key:
                    os.environ["COMET_API_KEY"] = parsed_args.api_key
                if parsed_args.url_override:
                    os.environ["COMET_URL_OVERRIDE"] = parsed_args.url_override

                # Run the Streamlit app using streamlit's CLI
                try:
                    import tempfile

                    import streamlit.web.cli as stcli

                    # Get the path to admin_gpu_app.py
                    gpu_app_path = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), "admin_gpu_app.py"
                    )

                    # Write JSON file path to a temp file
                    # The Streamlit app will read this file
                    temp_dir = tempfile.gettempdir()
                    json_path_file = os.path.join(
                        temp_dir, "comet_gpu_report_json_path.txt"
                    )
                    with open(json_path_file, "w") as f:
                        f.write(os.path.abspath(json_file_path))

                    # Launch streamlit with the app
                    sys.argv = ["streamlit", "run", gpu_app_path]
                    stcli.main()
                except Exception as e:
                    print(f"ERROR launching Streamlit app: {e}")
                    if parsed_args.debug:
                        raise
                    return
            else:
                # Generate report (JSON is always saved by gpu_report_main)
                if not workspace_projects:
                    print(
                        "ERROR: At least one workspace/project is required when not using --app"
                    )
                    return
                try:
                    result = gpu_report_main(
                        workspace_projects=workspace_projects,
                        start_date=start_date,
                        end_date=end_date,
                        metrics=metrics,
                        max_workers=None,  # Use default
                    )
                    if result:
                        num_experiments = len(result.get("metrics", {}))
                        num_charts = len(result.get("charts", []))
                        pdf_file = result.get("pdf_file")
                        json_file = result.get("json_file")

                        print(
                            f"\nGPU report completed. Processed {num_experiments} experiments."
                        )
                        if num_charts > 0:
                            print(f"Generated {num_charts} charts:")
                            for chart_file in result.get("charts", []):
                                print(f"  - {chart_file}")
                        if json_file:
                            print(f"JSON report: {json_file}")
                        if pdf_file:
                            print(f"PDF report: {pdf_file}")
                            # Open PDF if --open flag is set
                            if parsed_args.open:
                                from .admin_gpu_report import open_pdf

                                open_pdf(pdf_file, debug=parsed_args.debug)
                except Exception as e:
                    print("ERROR: " + str(e))
                    if parsed_args.debug:
                        import traceback

                        traceback.print_exc()
                    return
        elif parsed_args.ACTION == "optimizer-report":
            optimizer_id = parsed_args.OPTIMIZER_ID

            if parsed_args.app:
                # Launch Streamlit app
                # Set environment variables if --api-key or --url-override were provided
                if parsed_args.api_key:
                    os.environ["COMET_API_KEY"] = parsed_args.api_key
                if parsed_args.url_override:
                    os.environ["COMET_URL_OVERRIDE"] = parsed_args.url_override

                # Run the Streamlit app using streamlit's CLI
                try:
                    import streamlit.web.cli as stcli

                    # Get the path to admin_optimizer_report.py
                    optimizer_app_path = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "admin_optimizer_report.py",
                    )

                    # Set optimizer ID in environment so the app can access it
                    # Use COMET_OPTIMIZER_ID if not already set
                    if "COMET_OPTIMIZER_ID" not in os.environ:
                        os.environ["COMET_OPTIMIZER_ID"] = optimizer_id

                    # Launch streamlit with the app (Streamlit will automatically open the browser)
                    sys.argv = ["streamlit", "run", optimizer_app_path]
                    stcli.main()
                except Exception as e:
                    print(f"ERROR launching Streamlit app: {e}")
                    if parsed_args.debug:
                        raise
                    return
            else:
                # Generate JSON report
                try:
                    # Get API key and optimizer URL from config or arguments
                    from comet_ml.config import get_config, get_optimizer_address

                    config_obj = get_config()

                    # Get API key from parsed args, config, or environment
                    api_key = parsed_args.api_key
                    if not api_key:
                        try:
                            api_key = config_obj["comet.api_key"]
                        except (KeyError, TypeError):
                            api_key = os.environ.get("COMET_API_KEY")

                    # Get optimizer URL from config
                    optimizer_url = get_optimizer_address(config_obj)

                    result = generate_json_report(
                        optimizer_id=optimizer_id,
                        api_key=api_key,
                        optimizer_url=optimizer_url,
                    )
                    if result:
                        print(f"\nOptimizer report generated successfully: {result}")
                    else:
                        print("ERROR: Failed to generate optimizer report")
                        return
                except Exception as e:
                    print("ERROR: " + str(e))
                    if parsed_args.debug:
                        import traceback

                        traceback.print_exc()
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
