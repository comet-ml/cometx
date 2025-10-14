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

    cometx download
    cometx copy
    cometx count
    cometx log
    cometx list
    cometx reproduce
    cometx delete-assets
    cometx config
    cometx smoke-test
    cometx update
    cometx admin

For more information:
    cometx COMMAND --help
"""
import argparse
import os
import sys

from cometx import __version__

# Import CLI commands:
from . import (
    admin,
    config,
    copy,
    count,
    delete_assets,
    download,
    list_command,
    log,
    reproduce,
    smoke_test,
    update,
)


def add_subparser(subparsers, module, name):
    """
    Loads scripts and creates subparser.

    Assumes: NAME works for:
       * NAME.NAME is the function
       * comet_NAME.ADDITIONAL_ARGS is set to True/False
       * comet_NAME.get_parser_arguments is defined
    """
    func = getattr(module, name.replace("-", "_"))
    additional_args = module.ADDITIONAL_ARGS
    get_parser_arguments = module.get_parser_arguments
    docs = module.__doc__

    parser = subparsers.add_parser(
        name, description=docs, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.set_defaults(func=func)
    parser.set_defaults(additional_args=additional_args)
    get_parser_arguments(parser)


def main(raw_args=sys.argv[1:]):
    # Create single parser with global arguments and subparsers
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--version",
        help="Display comet_ml version",
        action="store_const",
        const=True,
        default=False,
    )
    parser.add_argument("--api-key", help="Set the COMET_API_KEY", type=str)
    parser.add_argument("--url-override", help="Set the COMET_URL_OVERRIDE", type=str)

    # Add subparsers to the same parser
    subparsers = parser.add_subparsers()

    # Register CLI commands:
    add_subparser(subparsers, download, "download")
    add_subparser(subparsers, copy, "copy")
    add_subparser(subparsers, update, "update")
    add_subparser(subparsers, admin, "admin")
    add_subparser(subparsers, log, "log")
    add_subparser(subparsers, delete_assets, "delete-assets")
    add_subparser(subparsers, list_command, "list")
    add_subparser(subparsers, count, "count")
    add_subparser(subparsers, reproduce, "reproduce")
    add_subparser(subparsers, config, "config")
    add_subparser(subparsers, smoke_test, "smoke-test")

    # Parse arguments
    args, rest = parser.parse_known_args(raw_args)

    # Set global environment variables early
    if args.api_key:
        os.environ["COMET_API_KEY"] = args.api_key
    if args.url_override:
        os.environ["COMET_URL_OVERRIDE"] = args.url_override

    # Handle version flag
    if args.version:
        print(__version__)
        return

    # Handle subcommands
    if hasattr(args, "additional_args") and args.additional_args:
        parser_func = args.func
        parser_func(args, rest)
    elif hasattr(args, "func"):
        # If the subcommand doesn't need extra args, reparse in strict mode so
        # the users get a nice message in case of unsupported CLI argument
        args = parser.parse_args(raw_args)
        parser_func = args.func
        parser_func(args)
    else:
        # No subcommand provided, show help
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
