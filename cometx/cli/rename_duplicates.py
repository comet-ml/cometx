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
Rename duplicate experiments in Comet ML.

cometx rename-duplicates [PATH] [--dry-run] [--debug]

where PATH is optional and can be:
* WORKSPACE - process all projects in a workspace
* WORKSPACE/PROJECT - process a single project
* (empty) - process all workspaces and projects

This command traverses Comet ML workspaces, projects, and experiments,
identifying experiments with duplicate names within the same project,
and renaming them to NAME-1, NAME-2, etc. while avoiding conflicts with
existing names.
"""

import argparse
import sys
from collections import defaultdict

import comet_ml
from tqdm import tqdm

ADDITIONAL_ARGS = False


def get_parser_arguments(parser):
    parser.add_argument(
        "PATH",
        nargs="?",
        default=None,
        help="Optional: WORKSPACE or WORKSPACE/PROJECT to restrict scope",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without renaming",
    )
    parser.add_argument(
        "--debug",
        help="Provide debug info",
        action="store_true",
        default=False,
    )


def get_next_available_name(base_name, existing_names):
    """Find NAME-N where N is the smallest integer making it unique.

    Args:
        base_name: The original experiment name
        existing_names: Set of names already in use

    Returns:
        A unique name in the format base_name-N
    """
    n = 1
    while True:
        new_name = f"{base_name}-{n}"
        if new_name not in existing_names:
            return new_name
        n += 1


def process_project(api, workspace, project_name, dry_run=False, debug=False):
    """Process a single project, renaming duplicate experiments.

    Args:
        api: Comet ML API instance
        workspace: Workspace name
        project_name: Project name
        dry_run: If True, preview changes without renaming
        debug: If True, print debug info

    Returns:
        List of tuples (experiment, old_name, new_name) for all renames
    """
    if debug:
        print(f"Getting experiments for {workspace}/{project_name}...")

    experiments = api.get_experiments(workspace, project_name)

    # Group experiments by name
    name_to_exps = defaultdict(list)
    for exp in experiments:
        name_to_exps[exp.name].append(exp)

    # Set of all names (for conflict detection)
    existing_names = set(name_to_exps.keys())

    # Process duplicates
    renames = []
    for name, exps in name_to_exps.items():
        if len(exps) > 1:
            # Keep first experiment unchanged, rename the rest
            for exp in exps[1:]:
                new_name = get_next_available_name(name, existing_names)
                existing_names.add(new_name)
                renames.append((exp, name, new_name))
                if not dry_run:
                    if debug:
                        print(f"  Renaming {exp.id}: '{name}' -> '{new_name}'")
                    exp.set_name(new_name)

    return renames


def rename_duplicates(parsed_args):
    """Main entry point for the rename-duplicates command."""
    debug = parsed_args.debug
    dry_run = parsed_args.dry_run
    path = parsed_args.PATH

    try:
        # Parse PATH argument
        filter_workspace = None
        filter_project = None
        if path:
            if "/" in path:
                filter_workspace, filter_project = path.split("/", 1)
            else:
                filter_workspace = path

        api = comet_ml.API()

        if dry_run:
            tqdm.write("DRY RUN - No changes will be made\n")

        if debug:
            print(f"Filter workspace: {filter_workspace}")
            print(f"Filter project: {filter_project}")

        # Collect all workspace/project pairs to process
        projects_to_process = []
        for workspace in api.get_workspaces():
            if filter_workspace and workspace != filter_workspace:
                continue

            for project_name in api.get_projects(workspace):
                if filter_project and project_name != filter_project:
                    continue
                projects_to_process.append((workspace, project_name))

        if debug:
            print(f"Projects to process: {len(projects_to_process)}")

        total_renames = 0

        for workspace, project_name in tqdm(
            projects_to_process, desc="Processing projects"
        ):
            renames = process_project(api, workspace, project_name, dry_run, debug)

            if renames:
                tqdm.write(f"[{workspace}/{project_name}]")
                for exp, old_name, new_name in renames:
                    action = "Would rename" if dry_run else "Renamed"
                    tqdm.write(f"  {action}: '{old_name}' -> '{new_name}' ({exp.id})")
                total_renames += len(renames)

        print(
            f"\nTotal: {total_renames} experiment(s) "
            f"{'would be ' if dry_run else ''}renamed"
        )

    except Exception as exc:
        if debug:
            raise
        else:
            print("ERROR: " + str(exc))


def main(args):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    get_parser_arguments(parser)
    parsed_args = parser.parse_args(args)
    rename_duplicates(parsed_args)


if __name__ == "__main__":
    main(sys.argv[1:])
