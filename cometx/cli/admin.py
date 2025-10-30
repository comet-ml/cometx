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
cometx admin usage-report WORKSPACE/PROJECT

"""

import argparse
import json
import sys
from urllib.parse import urlparse

from comet_ml import API

from .admin_usage_report import generate_usage_report

ADDITIONAL_ARGS = False


def get_parser_arguments(parser):
    parser.add_argument(
        "ACTION",
        help="The admin action to perform (chargeback-report, usage-report)",
        type=str,
    )
    parser.add_argument(
        "YEAR_MONTH",
        nargs="?",
        help="(deprecated) The YEAR-MONTH to run report for, eg 2024-09",
        metavar="YEAR-MONTH",
        type=str,
        default=None,
    )
    parser.add_argument(
        "WORKSPACE_PROJECT",
        nargs="?",
        help="The WORKSPACE/PROJECT to run usage report for (required for usage-report action)",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--host",
        help="Override the HOST URL",
        type=str,
    )
    parser.add_argument(
        "--debug", help="If given, allow debugging", default=False, action="store_true"
    )
    parser.add_argument(
        "--no-open",
        help="Don't automatically open the generated PDF file",
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
            # For usage-report, the workspace/project is passed as YEAR_MONTH argument
            workspace_project = parsed_args.YEAR_MONTH or parsed_args.WORKSPACE_PROJECT
            if not workspace_project:
                print("ERROR: WORKSPACE/PROJECT is required for usage-report action")
                print("Usage: cometx admin usage-report WORKSPACE/PROJECT")
                return
            try:
                generate_usage_report(api, workspace_project, parsed_args.no_open)

            except Exception as e:
                print("ERROR: " + str(e))
                return
        else:
            print(
                "Unknown action %r; should be one of these: 'chargeback-report', 'usage-report'"
                % parsed_args.ACTION
            )

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
