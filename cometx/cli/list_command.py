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

import argparse
import sys

from comet_ml.exceptions import Unauthorized

from cometx.framework.comet import DownloadManager
from cometx.utils import display_invalid_api_key

ADDITIONAL_ARGS = False


def get_parser_arguments(parser):
    parser.add_argument(
        "COMET_PATH",
        help=(
            "The Comet identifier, such as 'WORKSPACE', 'WORKSPACE/PROJECT', or "
            + "'WORKSPACE/PROJECT/EXPERIMENT'. Leave empty to use defaults."
        ),
        nargs="?",
    )
    parser.add_argument(
        "-u",
        "--use-name",
        help="Use experiment names for experiment folders and listings",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--query",
        help="Only list experiments that match this Comet query string",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--debug",
        help="Provide debug info",
        action="store_true",
        default=False,
    )


def list(parsed_args, remaining=None):
    try:
        downloader = DownloadManager()
    except ValueError as exc:
        if parsed_args.debug:
            raise exc from None
        else:
            display_invalid_api_key()
            return
    except Exception as exc:
        if parsed_args.debug:
            raise exc from None
        else:
            print("List aborted: %s" % str(exc))

    try:
        downloader.download(
            comet_path=parsed_args.COMET_PATH,
            use_name=parsed_args.use_name,
            list_items=True,
            query=parsed_args.query,
        )
    except Unauthorized as exc:
        if parsed_args.debug:
            raise exc from None
        else:
            display_invalid_api_key()
    except Exception as exc:
        if parsed_args.debug:
            raise exc from None
        else:
            print("List aborted: %s" % str(exc))
    except KeyboardInterrupt:
        print("User canceled list by keyboard interrupt")


def main(args):
    # Called via `cometx list ...`
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    get_parser_arguments(parser)
    parsed_args = parser.parse_args(args)

    list(parsed_args)


if __name__ == "__main__":
    # Called via `python -m cometx.cli.list ...`
    # Called via `cometx list ...`
    main(sys.argv[1:])
