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
Write and test a Comet Python panel locally, then upload it.

Opens an editor in your browser. Pick a workspace, project and experiments in
the sidebar, edit the panel's code, and press Run: the panel runs the same way
it does on comet.com, against your real experiments. Upload puts it on a project
dashboard or in your workspace's Panel library.

Examples:

    cometx panel-studio
    cometx panel-studio my_panel.py
    cometx panel-studio my_panel.py --workspace WS --project PROJECT

The API key and server come from your Comet config (COMET_API_KEY,
COMET_URL_OVERRIDE, ~/.comet.config), or --api-key / --url-override.
"""
import argparse
import atexit
import os
import shutil
import socket
import subprocess
import sys
import tempfile

ADDITIONAL_ARGS = False

# The panel server's start page; the panels themselves are pages beside it.
PANEL_SERVER_MAIN = """\
import streamlit as st

st.markdown("Press **Run** in the panel studio to show the panel here.")
"""


def get_parser_arguments(parser):
    parser.add_argument(
        "FILE",
        nargs="?",
        default=None,
        help="the panel's Python file; it is created when you save",
    )
    parser.add_argument("--workspace", help="the workspace to start with")
    parser.add_argument("--project", help="the project to start with")
    parser.add_argument(
        "--port",
        type=port_number,
        default=8501,
        help="the editor's port (default 8501)",
    )
    parser.add_argument(
        "--panel-port",
        type=port_number,
        default=8502,
        help="the port the panel itself is served on (default 8502)",
    )
    add_credential_arguments(parser)


def port_number(value):
    """argparse type for a TCP port, 1 through 65535."""
    try:
        port = int(value)
    except ValueError:
        port = 0
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("%r is not a port number (1-65535)" % (value,))
    return port


def add_credential_arguments(parser):
    # SUPPRESS: when not given here, keep a value given before the command
    # (cometx --api-key KEY panel-preview ...) instead of replacing it with None.
    parser.add_argument("--api-key", help="a Comet API key", default=argparse.SUPPRESS)
    parser.add_argument(
        "--url-override",
        help="a Comet server, e.g. https://comet.example.com/clientlib/",
        default=argparse.SUPPRESS,
    )


def apply_credentials(parsed_args):
    """Make --api-key and --url-override what comet_ml uses."""
    if getattr(parsed_args, "api_key", None):
        os.environ["COMET_API_KEY"] = parsed_args.api_key
    if getattr(parsed_args, "url_override", None):
        os.environ["COMET_URL_OVERRIDE"] = parsed_args.url_override


def make_work_dir():
    """A private temp folder for panel pages, removed on exit. The pages hold
    the API key, as on Comet's panel server."""
    work_dir = tempfile.mkdtemp(prefix="cometx-panel-studio-")
    # Removed after the panel server stops (atexit runs in reverse order).
    atexit.register(shutil.rmtree, work_dir, ignore_errors=True)
    return work_dir


def port_in_use(port):
    """True if something is already listening on this local port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def free_port(port, taken=()):
    """This port, or the next one up that is free and not in taken."""
    while port in taken or port_in_use(port):
        port += 1
    return port


def start_panel_server(work_dir, port):
    """Start the Streamlit server that shows the panel, as Comet's panel server
    does: a start page, with each panel written as a page in pages/."""
    pages_dir = os.path.join(work_dir, "pages")
    os.makedirs(pages_dir, exist_ok=True)
    main_path = os.path.join(work_dir, "panel_server.py")
    with open(main_path, "w") as fp:
        fp.write(PANEL_SERVER_MAIN)

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            main_path,
            "--server.port",
            str(port),
            "--server.headless",
            "true",
            "--server.runOnSave",
            "true",
            "--browser.gatherUsageStats",
            "false",
            "--client.toolbarMode",
            "minimal",
            # Each panel is a page of this server; don't list them beside it.
            "--client.showSidebarNavigation",
            "false",
        ],
        cwd=work_dir,
        # Only the editor's address is for the user; the panel server's
        # startup messages and logs go to a file instead.
        stdout=open(os.path.join(work_dir, "panel_server.log"), "w"),
        stderr=subprocess.STDOUT,
    )
    atexit.register(process.terminate)
    return process, pages_dir


def panel_studio(parsed_args, remaining=None):
    apply_credentials(parsed_args)

    # Move past ports that are taken, such as by another studio.
    port = free_port(parsed_args.port)
    panel_port = free_port(parsed_args.panel_port, taken=(port,))
    for name, wanted, chosen in (
        ("editor", parsed_args.port, port),
        ("panel", parsed_args.panel_port, panel_port),
    ):
        if chosen != wanted:
            print("Port %d is in use; the %s uses port %d." % (wanted, name, chosen))
    parsed_args.port, parsed_args.panel_port = port, panel_port

    work_dir = make_work_dir()
    _, pages_dir = start_panel_server(work_dir, parsed_args.panel_port)

    os.environ["COMETX_PANEL_STUDIO_PAGES"] = pages_dir
    os.environ["COMETX_PANEL_STUDIO_PANEL_PORT"] = str(parsed_args.panel_port)
    if parsed_args.FILE:
        os.environ["COMETX_PANEL_STUDIO_FILE"] = os.path.abspath(parsed_args.FILE)
    if parsed_args.workspace:
        os.environ["COMETX_PANEL_STUDIO_WORKSPACE"] = parsed_args.workspace
    if parsed_args.project:
        os.environ["COMETX_PANEL_STUDIO_PROJECT"] = parsed_args.project

    import streamlit.web.cli as stcli

    app_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "panel_studio_app.py"
    )
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--server.port",
        str(parsed_args.port),
        "--browser.gatherUsageStats",
        "false",
        "--client.toolbarMode",
        "minimal",
    ]
    stcli.main()


def main(args):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    get_parser_arguments(parser)
    parsed_args = parser.parse_args(args)
    panel_studio(parsed_args)


if __name__ == "__main__":
    # Called via `python -m cometx.cli.panel_studio ...`
    main(sys.argv[1:])
