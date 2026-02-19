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
Migrate users into workspaces from a source environment to a destination
environment based on a chargeback report.

The chargeback report is fetched automatically from the source environment's
admin API, or you can provide a local JSON file with --chargeback-report.

Examples:

$ cometx --api-key DEST_KEY migrate-users --source-api-key SOURCE_KEY --dry-run
$ cometx --api-key DEST_KEY migrate-users --source-api-key SOURCE_KEY
$ cometx --api-key DEST_KEY migrate-users --chargeback-report /path/to/report.json
"""

import argparse
import base64
import json
import os
import sys
import requests

ADDITIONAL_ARGS = False

COMET_CLOUD_URL = "https://www.comet.com"


def get_parser_arguments(parser):
    parser.add_argument(
        "--api-key",
        help="API key for the destination environment (used to add workspace members)",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--source-api-key",
        help="API key for the source environment (used to fetch the chargeback report)",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--chargeback-report",
        help="Path to a local chargeback report JSON file (skips fetching from source environment)",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--create-workspaces",
        help="Create workspaces on the destination environment if they don't already exist",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--dry-run",
        help="Print what would happen without making any changes",
        default=False,
        action="store_true",
    )


def _resolve_server_url(api_key):
    if "*" not in api_key:
        return COMET_CLOUD_URL

    _, encoded = api_key.split("*", 1)
    payload = json.loads(base64.b64decode(encoded))
    return payload["baseUrl"].rstrip("/")


def _fetch_chargeback_report(server_url, source_api_key):
    url = f"{server_url}/api/admin/chargeback/report"
    print(f"Fetching chargeback report from {url}...")
    resp = requests.get(
        url,
        headers={"Authorization": source_api_key},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _load_chargeback_report(path):
    print(f"Loading chargeback report from {path}...")
    with open(path, "r") as f:
        return json.load(f)


def _get_existing_workspaces(dest_url, headers):
    url = f"{dest_url}/api/rest/v2/workspaces"
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    return set(resp.json())


def _create_workspace(dest_url, headers, workspace_name):
    url = f"{dest_url}/api/rest/v2/write/workspace/new"
    resp = requests.post(
        url,
        headers=headers,
        json={"name": workspace_name},
        timeout=15,
    )
    resp.raise_for_status()


def _add_member(url, headers, email, workspace_name):
    """Attempt to add a single member. Returns (status, error_info_or_None)."""
    payload = {
        "userEmail": email,
        "workspaceName": workspace_name,
        "admin": False,
    }
    try:
        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(payload),
            timeout=15,
        )

        if response.status_code == 200:
            return "added", None

        try:
            error_msg = response.json()
        except ValueError:
            error_msg = response.text

        is_already_member = (
            isinstance(error_msg, dict)
            and "already member of" in error_msg.get("msg", "")
        )
        if is_already_member:
            return "already_member", None

        return "failed", {
            "status": response.status_code,
            "error": error_msg,
        }

    except requests.exceptions.RequestException as e:
        return "failed", {
            "status": "exception",
            "error": str(e),
        }


def migrate_users(parsed_args):
    api_key = parsed_args.api_key or os.environ.get("COMET_API_KEY")
    if not api_key:
        print("[ERROR] No API key found. Set COMET_API_KEY or pass --api-key.")
        sys.exit(1)

    source_api_key = parsed_args.source_api_key or os.environ.get("COMET_API_KEY")
    create_workspaces = parsed_args.create_workspaces
    dry_run = parsed_args.dry_run

    if not parsed_args.chargeback_report and not source_api_key:
        print("[ERROR] Provide either --source-api-key or --chargeback-report.")
        sys.exit(1)

    dest_url = _resolve_server_url(api_key)
    print(f"Destination URL: {dest_url}")

    if parsed_args.chargeback_report:
        data = _load_chargeback_report(parsed_args.chargeback_report)
    else:
        source_url = _resolve_server_url(source_api_key)
        print(f"Source URL: {source_url}")
        try:
            data = _fetch_chargeback_report(source_url, source_api_key)
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Failed to fetch chargeback report: {e}")
            sys.exit(1)

    workspaces = data.get("workspaces", [])
    print(f"Found {len(workspaces)} workspaces\n")

    headers = {
        "Authorization": api_key,
        "Content-Type": "application/json",
    }

    if create_workspaces and not dry_run:
        print("Checking existing workspaces on destination...")
        try:
            existing = _get_existing_workspaces(dest_url, headers)
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Failed to list destination workspaces: {e}")
            sys.exit(1)

        ws_created = 0
        for workspace in workspaces:
            ws_name = workspace["name"]
            if ws_name not in existing:
                try:
                    _create_workspace(dest_url, headers, ws_name)
                    print(f"  [CREATED] Workspace '{ws_name}'")
                    ws_created += 1
                except requests.exceptions.RequestException as e:
                    print(f"  [ERROR] Failed to create workspace '{ws_name}': {e}")
        print(f"Created {ws_created} new workspace(s)\n")
    elif create_workspaces and dry_run:
        print("[DRY RUN] Would check/create workspaces on destination\n")

    url = f"{dest_url}/api/rest/v2/write/add-workspace-member"

    total_added = 0
    total_skipped = 0
    total_no_email = 0
    total_already_member = 0
    total_failed = 0
    failures = []

    for workspace in workspaces:
        ws_name = workspace["name"]
        members = workspace.get("members", [])

        if not members:
            print(f"[SKIP] Workspace '{ws_name}' — no members, skipping")
            total_skipped += 1
            continue

        print(f"\n{'=' * 60}")
        print(f"Workspace: {ws_name} ({len(members)} members)")
        print(f"{'=' * 60}")

        ws_added = 0
        ws_no_email = 0
        ws_already_member = 0
        ws_failed = 0

        for member in members:
            email = member.get("email")

            if not email:
                ws_no_email += 1
                total_no_email += 1
                print(f"  [SKIP] '{member['userName']}' has no email, skipping")
                continue

            if dry_run:
                ws_added += 1
                total_added += 1
                continue

            status, error_info = _add_member(url, headers, email, ws_name)

            if status == "added":
                ws_added += 1
                total_added += 1
            elif status == "already_member":
                ws_already_member += 1
                total_already_member += 1
            else:
                print(
                    f"  [FAIL] '{email}' -> '{ws_name}' "
                    f"— {error_info['status']}: {error_info['error']}"
                )
                ws_failed += 1
                total_failed += 1
                failures.append(
                    {
                        "workspace": ws_name,
                        "email": email,
                        **error_info,
                    }
                )

        if dry_run:
            print(f"  [DRY RUN] Would add {ws_added}/{len(members)} users")
        else:
            parts = [f"  Added {ws_added}/{len(members)} users successfully"]
            if ws_already_member:
                parts.append(f"{ws_already_member} already a member")
            if ws_no_email:
                parts.append(f"{ws_no_email} skipped (no email)")
            if ws_failed:
                parts.append(f"{ws_failed} failed")
            print(
                parts[0]
                + (" (" + ", ".join(parts[1:]) + ")" if len(parts) > 1 else "")
            )

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Workspaces processed : {len(workspaces)}")
    print(f"  Workspaces skipped   : {total_skipped} (no members)")
    print(f"  Users added          : {total_added}")
    print(f"  Already a member     : {total_already_member}")
    print(f"  Skipped (no email)   : {total_no_email}")
    print(f"  Failures             : {total_failed}")

    if dry_run:
        print(f"\n  *** DRY RUN MODE — no API calls were made ***")
        print(f"  *** Remove --dry-run to execute for real ***")

    if failures:
        failures_file = "bulk_add_failures_by_email.json"
        with open(failures_file, "w") as f:
            json.dump(failures, f, indent=2)
        print(f"\n  Failed operations saved to {failures_file}")


def main(args):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    get_parser_arguments(parser)
    parsed_args = parser.parse_args(args)
    migrate_users(parsed_args)


if __name__ == "__main__":
    main(sys.argv[1:])
