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
Preview a Python panel file while you edit it in your own editor.

Runs the panel the way a Comet dashboard does, against your real experiments,
and opens it in your browser. Each time you save the file, the panel runs
again with the new code. Stop with Ctrl-C.

Examples:

    cometx panel-preview loss_curves.py --project PROJECT
    cometx panel-preview loss_curves.py --project PROJECT --experiments 10
    cometx panel-preview loss_curves.py --project PROJECT --experiments all
    cometx panel-preview loss_curves.py --project PROJECT \\
        --experiments run-01-sgd,run-02-adam

--experiments is a number (the first N, as a dashboard shows its first page;
25 by default), "all", or experiment names, keys or key prefixes separated by
commas. --sort orders experiments before taking the first N.
"""
import argparse
import os
import signal
import sys
import time
import webbrowser

from comet_ml.config import get_config

from ..api import API
from ..panel_studio import (
    choose_experiments,
    clientlib_url,
    file_signature,
    list_experiments,
    page_name_for,
    panel_config,
    read_settled,
    resolve_workspace,
    sort_entries,
    write_panel_page,
)
from .panel_studio import (
    add_credential_arguments,
    apply_credentials,
    free_port,
    make_work_dir,
    port_number,
    start_panel_server,
)

ADDITIONAL_ARGS = False

# How often to check the file for changes, in seconds.
POLL_INTERVAL = 0.5


def get_parser_arguments(parser):
    parser.add_argument("FILE", help="the panel's Python file")
    parser.add_argument("--project", required=True, help="the project to preview in")
    parser.add_argument(
        "--workspace", help="defaults to your configured or default workspace"
    )
    parser.add_argument(
        "--experiments",
        default="25",
        help='a number, "all", or names/keys separated by commas (default 25)',
    )
    parser.add_argument(
        "--sort",
        choices=["updated", "name"],
        default="updated",
        help="order of experiments: recently updated (default) or name",
    )
    parser.add_argument(
        "--port",
        type=port_number,
        default=8502,
        help="the panel's port (default 8502)",
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="don't open the panel in a browser"
    )
    add_credential_arguments(parser)


def read_first(path, attempts=10):
    """The file's (signature, text) once it's readable and not mid-save, or
    None if it's missing, a folder, or never settles."""
    if not os.path.isfile(path):
        return None
    for _ in range(attempts):
        settled = read_settled(path)
        if settled is not None:
            return settled
    return None


def panel_preview(parsed_args, remaining=None):
    apply_credentials(parsed_args)
    path = os.path.abspath(parsed_args.FILE)
    first = read_first(path)
    if first is None:
        sys.exit("Could not read the file %r." % parsed_args.FILE)

    api = API()
    workspace = resolve_workspace(api, parsed_args.workspace)
    entries = sort_entries(
        list_experiments(api, workspace, parsed_args.project), parsed_args.sort
    )
    try:
        experiment_keys = choose_experiments(entries, parsed_args.experiments)
    except ValueError as error:
        sys.exit(str(error))

    port = free_port(parsed_args.port)
    if port != parsed_args.port:
        print(
            "Port %d is in use; the panel uses port %d." % (parsed_args.port, port),
            flush=True,
        )
    _, pages_dir = start_panel_server(make_work_dir(), port)

    page_name = page_name_for(path)
    cfg = panel_config(
        clientlib_url(get_config("comet.url_override") or "https://www.comet.com"),
        api.api_key,
        workspace,
        parsed_args.project,
        experiment_keys,
        # A dashboard's Python panel gets no options.
        {},
        None,
        None,
        page_name,
    )

    write_panel_page(first[1], pages_dir, page_name, cfg)
    url = "http://localhost:%d/%s?embed=true" % (port, page_name)
    print(
        "Previewing %s in %s/%s with %d experiment(s):\n  %s\n"
        "Save the file to run it again; Ctrl-C to stop."
        % (
            parsed_args.FILE,
            workspace,
            parsed_args.project,
            len(experiment_keys),
            url,
        ),
        flush=True,
    )
    if not parsed_args.no_browser:
        webbrowser.open(url)

    # Stop on `kill` as on Ctrl-C, so the panel server stops and the temp folder
    # holding the API key is removed.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    # The panel server re-runs a page when it changes, so rewriting the page
    # is all it takes to run the panel again.
    last = first[0]
    try:
        while True:
            time.sleep(POLL_INTERVAL)
            current = file_signature(path)
            if current is None or current == last:
                continue
            settled = read_settled(path)
            if settled is None:  # still being written; look again next time
                continue
            last, code = settled
            write_panel_page(code, pages_dir, page_name, cfg)
            print(
                "%s  ran %s again" % (time.strftime("%H:%M:%S"), parsed_args.FILE),
                flush=True,
            )
    except KeyboardInterrupt:
        pass


def main(args):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    get_parser_arguments(parser)
    panel_preview(parser.parse_args(args))


if __name__ == "__main__":
    # Called via `python -m cometx.cli.panel_preview ...`
    main(sys.argv[1:])
