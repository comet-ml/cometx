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
#  Copyright (c) 2025 Cometx Development
#      Team. All rights reserved.
# ****************************************

"""
Script to count workspaces and/or projects in a Comet deployment using Python SDK only.

This script uses only the Comet Python SDK and is configurable to count:
- Only workspaces
- Workspaces and projects
- Workspaces, projects, and experiments (with optional query filtering)

Usage:
    cometx count --workspaces-only
    cometx count --with-projects
    cometx count  # defaults to workspaces and projects
    cometx count --query "QUERY_STRING"  # only count workspaces/projects with matching experiments

Examples:
  # Count only workspaces (fastest)
  cometx count --workspaces-only

  # Count workspaces and projects (default)
  cometx count
  cometx count --with-projects

  # Count workspaces, projects, and experiments (most detailed)
  cometx count --with-experiments

  # Count everything: workspaces, projects, artifacts, and experiments
  cometx count --count-all
  cometx count --count-all --limit 5

  # Limit to first 10 workspaces (useful for testing)
  cometx count --limit 10
  cometx count --with-experiments --limit 5

  # Count only workspaces/projects with experiments matching a query
  cometx count --query "Metadata('start_server_timestamp') < datetime(2023, 10, 1)"
  cometx count --with-experiments --query "Metric('accuracy') > 0.8"
"""

import sys
from typing import Any, Dict, Optional

import comet_ml

from ..utils import get_query_experiments

ADDITIONAL_ARGS = False


def get_parser_arguments(parser):
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--workspaces-only", action="store_true", help="Count only workspaces (fastest)"
    )
    mode_group.add_argument(
        "--with-projects",
        action="store_true",
        help="Count workspaces and projects (default)",
    )
    mode_group.add_argument(
        "--with-experiments",
        action="store_true",
        help="Count workspaces, projects, and experiments (slowest, most detailed)",
    )
    mode_group.add_argument(
        "--count-all",
        action="store_true",
        help=(
            "Count everything: workspaces, projects, artifacts, and experiments "
            "(most comprehensive)"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N workspaces (useful for testing)",
    )
    parser.add_argument(
        "--query",
        help=(
            "Only count workspaces/projects with experiments matching this Comet query string. "
            "See https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/API/#apiquery"
        ),
        type=str,
        default=None,
    )
    parser.add_argument(
        "--include-size",
        help=(
            "Calculate and display storage size for matching experiments (requires --query). "
            "This is expensive as it requires fetching asset lists for each experiment."
        ),
        action="store_true",
        default=False,
    )


class CometResourceCounter:
    """Count workspaces and projects in a Comet deployment using Python SDK."""

    def __init__(self):
        self.api = comet_ml.API()

    def _get_experiment_size_bytes(self, experiment) -> float:
        """
        Get the total size in bytes for an experiment by summing all asset file sizes.

        Args:
            experiment: Comet experiment object

        Returns:
            Total size in bytes (float)
        """
        try:
            assets = experiment.get_asset_list()
            if not assets:
                return 0.0
            total_size = sum(asset.get("fileSize", 0) for asset in assets)
            return float(total_size)
        except Exception:
            # If we can't get assets, return 0
            return 0.0

    @staticmethod
    def _format_size(self_or_cls, size_bytes: float) -> str:
        """
        Format size in bytes to a human-readable format.

        Args:
            size_bytes: Size in bytes

        Returns:
            Formatted string (e.g., "1.23 MB" or "4.56 GB")
        """
        if size_bytes == 0:
            return "0 B"
        size_mb = size_bytes / (1024 * 1024)
        if size_mb >= 1024:
            return f"{size_mb / 1024:.2f} GB"
        else:
            return f"{size_mb:.2f} MB"

    def count_workspaces_only(
        self, limit: Optional[int] = None, query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Count only workspaces using the Comet Python SDK.

        Args:
            limit: Optional limit on number of workspaces to process
            query: Optional Comet query string to filter experiments

        Returns:
            Dictionary containing workspace counts and details
        """
        results = {
            "workspaces": {
                "count": 0,
                "total_count": 0,  # Total available
                "names": [],
                "limited": False,
                "error": None,
            }
        }

        # Get workspaces
        all_workspaces = self.api.get_workspaces()
        results["workspaces"]["total_count"] = len(all_workspaces)

        # Apply limit if specified
        if limit is not None and limit < len(all_workspaces):
            workspaces_to_check = all_workspaces[:limit]
            results["workspaces"]["limited"] = True
        else:
            workspaces_to_check = all_workspaces

        # If query is provided, filter workspaces to only those with matching experiments
        if query:
            matching_workspaces = []
            print(f"\n🔍 Filtering workspaces by query: {query}", flush=True)
            print("-" * 60, flush=True)
            for i, workspace in enumerate(workspaces_to_check, 1):
                try:
                    print(
                        f"📊 [{i}/{len(workspaces_to_check)}] Checking {workspace}...",
                        end=" ",
                        flush=True,
                    )
                    projects = self.api.get_projects(workspace)
                    if projects:
                        for project_name in projects:
                            try:
                                experiments = get_query_experiments(
                                    self.api, query, workspace, project_name
                                )
                                if experiments and len(experiments) > 0:
                                    matching_workspaces.append(workspace)
                                    print(f"✅ Found matching experiments")
                                    break
                            except Exception:
                                continue
                    if workspace not in matching_workspaces:
                        print("❌ No matching experiments")
                except Exception as e:
                    print(f"❌ ERROR: {str(e)[:50]}")
            workspaces = matching_workspaces
        else:
            workspaces = workspaces_to_check

        results["workspaces"]["count"] = len(workspaces)
        results["workspaces"]["names"] = workspaces

        return results

    def count_workspaces_and_projects(
        self, limit: Optional[int] = None, query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Count workspaces and projects using the Comet Python SDK.

        Args:
            limit: Optional limit on number of workspaces to process
            query: Optional Comet query string to filter experiments

        Returns:
            Dictionary containing counts and details
        """
        results = {
            "workspaces": {
                "count": 0,
                "total_count": 0,
                "names": [],
                "limited": False,
                "error": None,
            },
            "projects": {"total_count": 0, "by_workspace": {}, "error": None},
        }

        # Get workspaces
        all_workspaces = self.api.get_workspaces()
        results["workspaces"]["total_count"] = len(all_workspaces)

        # Apply limit if specified
        if limit is not None and limit < len(all_workspaces):
            workspaces_to_check = all_workspaces[:limit]
            results["workspaces"]["limited"] = True
            print(
                f"\n⚠️  Limiting to first {limit} of {len(all_workspaces)} workspaces",
                flush=True,
            )
        else:
            workspaces_to_check = all_workspaces

        if query:
            print(f"\n🔍 Filtering by query: {query}", flush=True)

        # Count projects for each workspace
        total_projects = 0
        matching_workspaces = []
        print(
            f"\n🔍 Counting projects for {len(workspaces_to_check)} workspaces...",
            flush=True,
        )
        print("-" * 60, flush=True)

        for i, workspace in enumerate(workspaces_to_check, 1):
            try:
                # Show progress
                print(
                    f"📊 [{i}/{len(workspaces_to_check)}] {workspace}...",
                    end=" ",
                    flush=True,
                )

                projects = self.api.get_projects(workspace)
                if not projects:
                    print("📁 0 project(s)")
                    if not query:
                        # Only include workspace if no query (for backward compatibility)
                        matching_workspaces.append(workspace)
                    continue

                # If query is provided, filter projects to only those with matching experiments
                if query:
                    matching_projects = []
                    for project_name in projects:
                        try:
                            experiments = get_query_experiments(
                                self.api, query, workspace, project_name
                            )
                            if experiments and len(experiments) > 0:
                                matching_projects.append(project_name)
                        except Exception:
                            continue
                    project_count = len(matching_projects)
                    if project_count > 0:
                        matching_workspaces.append(workspace)
                        results["projects"]["by_workspace"][workspace] = project_count
                        total_projects += project_count
                        print(
                            f"📁 {project_count} project(s) with matching experiments"
                        )
                    else:
                        print("📁 0 project(s) with matching experiments")
                else:
                    project_count = len(projects)
                    matching_workspaces.append(workspace)
                    results["projects"]["by_workspace"][workspace] = project_count
                    total_projects += project_count
                    print(f"📁 {project_count} project(s)")

            except Exception as e:
                print(f"❌ ERROR: {str(e)[:50]}")
                results["projects"]["by_workspace"][workspace] = 0

        results["workspaces"]["count"] = len(matching_workspaces)
        results["workspaces"]["names"] = matching_workspaces
        results["projects"]["total_count"] = total_projects

        return results

    def count_workspaces_projects_and_experiments(
        self,
        limit: Optional[int] = None,
        query: Optional[str] = None,
        include_size: bool = False,
    ) -> Dict[str, Any]:
        """
        Count workspaces, projects, and experiments using the Comet Python SDK.

        Args:
            limit: Optional limit on number of workspaces to process
            query: Optional Comet query string to filter experiments

        Returns:
            Dictionary containing counts and details
        """
        results = {
            "workspaces": {
                "count": 0,
                "total_count": 0,
                "names": [],
                "limited": False,
                "error": None,
            },
            "projects": {
                "total_count": 0,
                "by_workspace": {},
                "details": {},  # Store project details including experiment counts
                "error": None,
            },
            "experiments": {
                "total_count": 0,
                "by_project": {},  # Format: {workspace/project: count}
                "error": None,
            },
            "sizes": {
                "total_bytes": 0.0,
                "by_workspace": {},  # Format: {workspace: bytes}
                "by_project": {},  # Format: {workspace/project: bytes}
            },
        }

        # Get workspaces
        all_workspaces = self.api.get_workspaces()
        results["workspaces"]["total_count"] = len(all_workspaces)

        # Apply limit if specified
        if limit is not None and limit < len(all_workspaces):
            workspaces = all_workspaces[:limit]
            results["workspaces"]["limited"] = True
            print(
                f"\n⚠️  Limiting to first {limit} of {len(all_workspaces)} workspaces",
                flush=True,
            )
        else:
            workspaces = all_workspaces

        results["workspaces"]["count"] = len(workspaces)
        results["workspaces"]["names"] = workspaces

        # Count projects and experiments for each workspace
        total_projects = 0
        total_experiments = 0
        total_size_bytes = 0.0
        matching_workspaces = []

        if query:
            print(f"\n🔍 Filtering by query: {query}", flush=True)
        if include_size and query:
            print(
                f"📊 Calculating storage sizes for matching experiments...", flush=True
            )

        print(
            f"\n🔍 Counting projects and experiments for {len(workspaces)} workspaces...",
            flush=True,
        )
        print("=" * 70, flush=True)

        for i, workspace in enumerate(workspaces, 1):
            try:
                print(
                    f"\n🏢 [{i}/{len(workspaces)}] Workspace: {workspace}", flush=True
                )

                projects = self.api.get_projects(workspace)
                if not projects:
                    print(f"  📁 Projects: 0", flush=True)
                    if not query:
                        matching_workspaces.append(workspace)
                    continue

                # If query is provided, filter projects to only those with matching experiments
                if query:
                    matching_projects = []
                    for project_name in projects:
                        try:
                            experiments = get_query_experiments(
                                self.api, query, workspace, project_name
                            )
                            if experiments and len(experiments) > 0:
                                matching_projects.append(project_name)
                        except Exception:
                            continue
                    projects_to_count = matching_projects
                    project_count = len(projects_to_count)
                else:
                    projects_to_count = projects
                    project_count = len(projects_to_count)

                if project_count > 0:
                    matching_workspaces.append(workspace)
                    results["projects"]["by_workspace"][workspace] = project_count
                    total_projects += project_count
                    print(f"  📁 Projects: {project_count}", flush=True)
                else:
                    print(f"  📁 Projects: 0 (no matching experiments)", flush=True)
                    continue

                # Count experiments for each project
                workspace_experiments = 0
                if projects_to_count:
                    for j, project_name in enumerate(projects_to_count, 1):
                        try:
                            print(
                                f"    🧪 [{j}/{project_count}] {project_name}...",
                                end=" ",
                                flush=True,
                            )

                            # Get experiments for this project
                            if query:
                                experiments = get_query_experiments(
                                    self.api, query, workspace, project_name
                                )
                            else:
                                experiments = self.api.get_experiments(
                                    workspace, project_name
                                )
                            exp_count = len(experiments) if experiments else 0

                            project_key = f"{workspace}/{project_name}"
                            results["experiments"]["by_project"][
                                project_key
                            ] = exp_count
                            workspace_experiments += exp_count
                            total_experiments += exp_count

                            # Calculate size if requested
                            project_size_bytes = 0.0
                            if include_size and query and experiments:
                                for exp in experiments:
                                    try:
                                        size = self._get_experiment_size_bytes(exp)
                                        project_size_bytes += size
                                    except Exception:
                                        pass
                                results["sizes"]["by_project"][
                                    project_key
                                ] = project_size_bytes
                                if workspace not in results["sizes"]["by_workspace"]:
                                    results["sizes"]["by_workspace"][workspace] = 0.0
                                results["sizes"]["by_workspace"][
                                    workspace
                                ] += project_size_bytes
                                total_size_bytes += project_size_bytes

                            if include_size and query and project_size_bytes > 0:
                                size_str = CometResourceCounter._format_size(
                                    None, project_size_bytes
                                )
                                print(f"🧪 {exp_count} experiment(s), {size_str}")
                            else:
                                print(f"🧪 {exp_count} experiment(s)")

                        except Exception as e:
                            print(f"❌ ERROR: {str(e)[:40]}")
                            project_key = f"{workspace}/{project_name}"
                            results["experiments"]["by_project"][project_key] = 0

                print(
                    f"  🧪 Total experiments in workspace: {workspace_experiments}",
                    flush=True,
                )

            except Exception as e:
                print(f"  ❌ ERROR: {str(e)[:50]}")
                results["projects"]["by_workspace"][workspace] = 0

        results["workspaces"]["count"] = len(matching_workspaces)
        results["workspaces"]["names"] = matching_workspaces
        results["projects"]["total_count"] = total_projects
        results["experiments"]["total_count"] = total_experiments
        results["sizes"]["total_bytes"] = total_size_bytes

        print("\n" + "=" * 70, flush=True)
        print("✅ Counting complete!", flush=True)

        return results

    def count_all_resources(
        self,
        limit: Optional[int] = None,
        query: Optional[str] = None,
        include_size: bool = False,
    ) -> Dict[str, Any]:
        """
        Count all resources: workspaces, projects, artifacts, and experiments
        using the Comet Python SDK.

        Args:
            limit: Optional limit on number of workspaces to process
            query: Optional Comet query string to filter experiments

        Returns:
            Dictionary containing counts and details
        """
        results = {
            "workspaces": {
                "count": 0,
                "total_count": 0,
                "names": [],
                "limited": False,
                "error": None,
            },
            "projects": {"total_count": 0, "by_workspace": {}, "error": None},
            "artifacts": {"total_count": 0, "by_workspace": {}, "error": None},
            "experiments": {"total_count": 0, "by_project": {}, "error": None},
            "sizes": {
                "total_bytes": 0.0,
                "by_workspace": {},
                "by_project": {},
            },
        }

        # Get workspaces
        all_workspaces = self.api.get_workspaces()
        results["workspaces"]["total_count"] = len(all_workspaces)

        # Apply limit if specified
        if limit is not None and limit < len(all_workspaces):
            workspaces_to_check = all_workspaces[:limit]
            results["workspaces"]["limited"] = True
            print(
                f"\n⚠️  Limiting to first {limit} of {len(all_workspaces)} workspaces",
                flush=True,
            )
        else:
            workspaces_to_check = all_workspaces

        if query:
            print(f"\n🔍 Filtering by query: {query}", flush=True)
        if include_size and query:
            print(
                f"📊 Calculating storage sizes for matching experiments...", flush=True
            )

        # Count projects, artifacts, and experiments for each workspace
        total_projects = 0
        total_artifacts = 0
        total_experiments = 0
        total_size_bytes = 0.0
        matching_workspaces = []

        print(
            f"\n🔍 Counting all resources for {len(workspaces_to_check)} workspaces...",
            flush=True,
        )
        print("=" * 70, flush=True)

        for i, workspace in enumerate(workspaces_to_check, 1):
            try:
                print(
                    f"\n🏢 [{i}/{len(workspaces_to_check)}] Workspace: {workspace}",
                    flush=True,
                )

                # Count projects
                projects = self.api.get_projects(workspace)
                if not projects:
                    print(f"  📁 Projects: 0", flush=True)
                    if not query:
                        matching_workspaces.append(workspace)
                    continue

                # If query is provided, filter projects to only those with matching experiments
                if query:
                    matching_projects = []
                    for project_name in projects:
                        try:
                            experiments = get_query_experiments(
                                self.api, query, workspace, project_name
                            )
                            if experiments and len(experiments) > 0:
                                matching_projects.append(project_name)
                        except Exception:
                            continue
                    projects_to_count = matching_projects
                    project_count = len(projects_to_count)
                else:
                    projects_to_count = projects
                    project_count = len(projects_to_count)

                if project_count > 0:
                    matching_workspaces.append(workspace)
                    results["projects"]["by_workspace"][workspace] = project_count
                    total_projects += project_count
                    print(f"  📁 Projects: {project_count}", flush=True)
                else:
                    print(f"  📁 Projects: 0 (no matching experiments)", flush=True)
                    # Still count artifacts even if no matching experiments
                    try:
                        print(f"  📦 Counting artifacts...", end=" ", flush=True)
                        artifacts_response = self.api.get_artifact_list(workspace)
                        artifact_list = (
                            artifacts_response.get("artifacts", [])
                            if artifacts_response
                            else []
                        )
                        artifact_count = len(artifact_list)
                        results["artifacts"]["by_workspace"][workspace] = artifact_count
                        total_artifacts += artifact_count
                        print(f"📦 {artifact_count} artifact(s)")
                    except Exception as e:
                        print(f"❌ ERROR: {str(e)[:40]}")
                        results["artifacts"]["by_workspace"][workspace] = 0
                    continue

                # Count artifacts
                try:
                    print(f"  📦 Counting artifacts...", end=" ", flush=True)
                    artifacts_response = self.api.get_artifact_list(workspace)
                    # get_artifact_list returns a dict like {'artifacts': [...]}
                    artifact_list = (
                        artifacts_response.get("artifacts", [])
                        if artifacts_response
                        else []
                    )
                    artifact_count = len(artifact_list)
                    results["artifacts"]["by_workspace"][workspace] = artifact_count
                    total_artifacts += artifact_count
                    print(f"📦 {artifact_count} artifact(s)")
                except Exception as e:
                    print(f"❌ ERROR: {str(e)[:40]}")
                    results["artifacts"]["by_workspace"][workspace] = 0

                # Count experiments for each project
                workspace_experiments = 0
                if projects_to_count:
                    for j, project_name in enumerate(projects_to_count, 1):
                        try:
                            print(
                                f"    🧪 [{j}/{project_count}] {project_name}...",
                                end=" ",
                                flush=True,
                            )

                            # Get experiments for this project
                            if query:
                                experiments = get_query_experiments(
                                    self.api, query, workspace, project_name
                                )
                            else:
                                experiments = self.api.get_experiments(
                                    workspace, project_name
                                )
                            exp_count = len(experiments) if experiments else 0

                            project_key = f"{workspace}/{project_name}"
                            results["experiments"]["by_project"][
                                project_key
                            ] = exp_count
                            workspace_experiments += exp_count
                            total_experiments += exp_count

                            # Calculate size if requested
                            project_size_bytes = 0.0
                            if include_size and query and experiments:
                                for exp in experiments:
                                    try:
                                        size = self._get_experiment_size_bytes(exp)
                                        project_size_bytes += size
                                    except Exception:
                                        pass
                                results["sizes"]["by_project"][
                                    project_key
                                ] = project_size_bytes
                                if workspace not in results["sizes"]["by_workspace"]:
                                    results["sizes"]["by_workspace"][workspace] = 0.0
                                results["sizes"]["by_workspace"][
                                    workspace
                                ] += project_size_bytes
                                total_size_bytes += project_size_bytes

                            if include_size and query and project_size_bytes > 0:
                                size_str = CometResourceCounter._format_size(
                                    None, project_size_bytes
                                )
                                print(f"🧪 {exp_count} experiment(s), {size_str}")
                            else:
                                print(f"🧪 {exp_count} experiment(s)")

                        except Exception as e:
                            print(f"❌ ERROR: {str(e)[:40]}")
                            project_key = f"{workspace}/{project_name}"
                            results["experiments"]["by_project"][project_key] = 0

                print(
                    f"  🧪 Total experiments in workspace: {workspace_experiments}",
                    flush=True,
                )

            except Exception as e:
                print(f"  ❌ ERROR: {str(e)[:50]}")
                results["projects"]["by_workspace"][workspace] = 0
                results["artifacts"]["by_workspace"][workspace] = 0

        results["workspaces"]["count"] = len(matching_workspaces)
        results["workspaces"]["names"] = matching_workspaces
        results["projects"]["total_count"] = total_projects
        results["artifacts"]["total_count"] = total_artifacts
        results["experiments"]["total_count"] = total_experiments
        results["sizes"]["total_bytes"] = total_size_bytes

        print("\n" + "=" * 70, flush=True)
        print("✅ Counting complete!", flush=True)

        return results


def print_workspaces_only(results: Dict[str, Any]):
    """Print only workspace results."""
    print(f"\n{'='*60}")
    print("📊 RESULTS")
    print(f"{'='*60}")

    if "error" in results and results["error"]:
        print(f"❌ Error: {results['error']}")
        return

    # Print workspace count
    if "workspaces" in results:
        ws = results["workspaces"]
        if ws.get("error"):
            print(f"🏢 Workspaces: ❌ Error - {ws['error']}")
        else:
            if ws.get("limited"):
                print(
                    f"Workspaces Processed: {ws['count']} "
                    f"(of {ws.get('total_count', ws['count'])} total)"
                )
            else:
                print(f"🏢 Total Workspaces: {ws['count']}")

            if ws["names"]:
                print("\n📋 Workspace Names:")
                for i, name in enumerate(sorted(ws["names"]), 1):
                    print(f"  {i}. 🏢 {name}")


def print_workspaces_and_projects(results: Dict[str, Any]):
    """Print workspace and project results."""
    print(f"\n{'='*60}")
    print("📊 RESULTS")
    print(f"{'='*60}")

    if "error" in results and results["error"]:
        print(f"❌ Error: {results['error']}")
        return

    # Print workspace count
    if "workspaces" in results:
        ws = results["workspaces"]
        if ws.get("error"):
            print(f"🏢 Workspaces: ❌ Error - {ws['error']}")
        else:
            if ws.get("limited"):
                print(
                    f"Workspaces Processed: {ws['count']} "
                    f"(of {ws.get('total_count', ws['count'])} total)"
                )
            else:
                print(f"🏢 Total Workspaces: {ws['count']}")

    # Print project count
    if "projects" in results:
        proj = results["projects"]
        if proj.get("error"):
            print(f"📁 Projects: ❌ Error - {proj['error']}")
        else:
            print(f"📁 Total Projects: {proj['total_count']}")

            # Print projects by workspace
            if proj["by_workspace"]:
                print("\n📊 Projects per workspace (sorted by count):")
                for workspace, count in sorted(
                    proj["by_workspace"].items(), key=lambda x: x[1], reverse=True
                ):
                    print(f"  🏢 {workspace}: 📁 {count} project(s)")


def print_workspaces_projects_and_experiments(results: Dict[str, Any]):
    """Print workspace, project, and experiment results."""
    print(f"\n{'='*60}")
    print("📊 RESULTS")
    print(f"{'='*60}")

    if "error" in results and results["error"]:
        print(f"❌ Error: {results['error']}")
        return

    # Print workspace count
    if "workspaces" in results:
        ws = results["workspaces"]
        if ws.get("error"):
            print(f"🏢 Workspaces: ❌ Error - {ws['error']}")
        else:
            if ws.get("limited"):
                print(
                    f"Workspaces Processed: {ws['count']} "
                    f"(of {ws.get('total_count', ws['count'])} total)"
                )
            else:
                print(f"🏢 Total Workspaces: {ws['count']}")

    # Print project count
    if "projects" in results:
        proj = results["projects"]
        if proj.get("error"):
            print(f"📁 Projects: ❌ Error - {proj['error']}")
        else:
            print(f"📁 Total Projects: {proj['total_count']}")

    # Print experiment count
    if "experiments" in results:
        exp = results["experiments"]
        if exp.get("error"):
            print(f"🧪 Experiments: ❌ Error - {exp['error']}")
        else:
            print(f"🧪 Total Experiments: {exp['total_count']}")

            # Print size information if available
            if "sizes" in results and results["sizes"].get("total_bytes", 0) > 0:
                total_size = results["sizes"]["total_bytes"]
                size_str = CometResourceCounter._format_size(None, total_size)
                print(f"💾 Total Storage Size: {size_str}")

            # Print top projects by experiment count
            if exp["by_project"]:
                print("\n🏆 Top 50 projects by experiment count:")
                sorted_projects = sorted(
                    exp["by_project"].items(), key=lambda x: x[1], reverse=True
                )
                for i, (project_key, count) in enumerate(sorted_projects[:50], 1):
                    size_info = ""
                    if "sizes" in results and project_key in results["sizes"].get(
                        "by_project", {}
                    ):
                        size_bytes = results["sizes"]["by_project"][project_key]
                        if size_bytes > 0:
                            size_str = CometResourceCounter._format_size(
                                None, size_bytes
                            )
                            size_info = f", {size_str}"
                    print(f"  {i}. 🧪 {project_key}: {count} experiment(s){size_info}")

                if len(sorted_projects) > 50:
                    print(f"\n  📊 ... and {len(sorted_projects) - 50} more projects")


def print_all_resources(results: Dict[str, Any]):
    """Print workspace, project, artifact, and experiment results."""
    print(f"\n{'='*60}")
    print("📊 RESULTS")
    print(f"{'='*60}")

    if "error" in results and results["error"]:
        print(f"❌ Error: {results['error']}")
        return

    # Print workspace count
    if "workspaces" in results:
        ws = results["workspaces"]
        if ws.get("error"):
            print(f"🏢 Workspaces: ❌ Error - {ws['error']}")
        else:
            if ws.get("limited"):
                print(
                    f"Workspaces Processed: {ws['count']} "
                    f"(of {ws.get('total_count', ws['count'])} total)"
                )
            else:
                print(f"🏢 Total Workspaces: {ws['count']}")

    # Print project count
    if "projects" in results:
        proj = results["projects"]
        if proj.get("error"):
            print(f"📁 Projects: ❌ Error - {proj['error']}")
        else:
            print(f"📁 Total Projects: {proj['total_count']}")

    # Print artifact count
    if "artifacts" in results:
        art = results["artifacts"]
        if art.get("error"):
            print(f"📦 Artifacts: ❌ Error - {art['error']}")
        else:
            print(f"📦 Total Artifacts: {art['total_count']}")

            # Print top workspaces by artifact count
            if art["by_workspace"]:
                print("\n🏆 Top 20 workspaces by artifact count:")
                sorted_workspaces = sorted(
                    art["by_workspace"].items(), key=lambda x: x[1], reverse=True
                )
                for i, (workspace, count) in enumerate(sorted_workspaces[:20], 1):
                    print(f"  {i}. 🏢 {workspace}: 📦 {count} artifact(s)")

                if len(sorted_workspaces) > 20:
                    print(
                        f"\n  📊 ... and {len(sorted_workspaces) - 20} more workspaces"
                    )

    # Print experiment count
    if "experiments" in results:
        exp = results["experiments"]
        if exp.get("error"):
            print(f"🧪 Experiments: ❌ Error - {exp['error']}")
        else:
            print(f"🧪 Total Experiments: {exp['total_count']}")

            # Print size information if available
            if "sizes" in results and results["sizes"].get("total_bytes", 0) > 0:
                total_size = results["sizes"]["total_bytes"]
                size_str = CometResourceCounter._format_size(None, total_size)
                print(f"💾 Total Storage Size: {size_str}")

            # Print top projects by experiment count
            if exp["by_project"]:
                print("\n🏆 Top 30 projects by experiment count:")
                sorted_projects = sorted(
                    exp["by_project"].items(), key=lambda x: x[1], reverse=True
                )
                for i, (project_key, count) in enumerate(sorted_projects[:30], 1):
                    size_info = ""
                    if "sizes" in results and project_key in results["sizes"].get(
                        "by_project", {}
                    ):
                        size_bytes = results["sizes"]["by_project"][project_key]
                        if size_bytes > 0:
                            size_str = CometResourceCounter._format_size(
                                None, size_bytes
                            )
                            size_info = f", {size_str}"
                    print(f"  {i}. 🧪 {project_key}: {count} experiment(s){size_info}")

                if len(sorted_projects) > 30:
                    print(f"\n  📊 ... and {len(sorted_projects) - 30} more projects")


def main(args):
    # Called via `cometx list ...`
    count(args)


def count(args, remaining=None):
    # Determine mode (default to with-projects if none is specified)
    workspaces_only = args.workspaces_only
    with_experiments = args.with_experiments
    count_all = args.count_all
    # Determine if we should count projects (default behavior)

    # Print header
    print("=" * 60)
    print("🚀 Comet Resource Counter (Python SDK)")
    print("=" * 60)

    # Show mode
    if workspaces_only:
        print("🏢 Mode: Counting workspaces only")
    elif count_all:
        print("🔍 Mode: Counting workspaces, projects, artifacts, and experiments")
    elif with_experiments:
        print("🧪 Mode: Counting workspaces, projects, and experiments")
    else:
        print("📁 Mode: Counting workspaces and projects")

    if args.query:
        print(f"🔍 Query filter: {args.query}")
    if args.include_size:
        if not args.query:
            print(
                "⚠️  Warning: --include-size requires --query. Size calculation skipped.",
                flush=True,
            )
        else:
            print(f"📊 Size calculation enabled (this may be slow for large datasets)")

    try:
        counter = CometResourceCounter()

        if workspaces_only:
            # Count only workspaces
            print("\n🏢 Counting workspaces...")
            results = counter.count_workspaces_only(limit=args.limit, query=args.query)
            print_workspaces_only(results)

        elif count_all:
            # Count everything: workspaces, projects, artifacts, and experiments
            print("\n🔍 Counting all resources...", flush=True)
            results = counter.count_all_resources(
                limit=args.limit, query=args.query, include_size=args.include_size
            )
            print_all_resources(results)

        elif with_experiments:
            # Count workspaces, projects, and experiments
            print("\n🧪 Counting workspaces, projects, and experiments...", flush=True)
            results = counter.count_workspaces_projects_and_experiments(
                limit=args.limit, query=args.query, include_size=args.include_size
            )
            print_workspaces_projects_and_experiments(results)

        else:
            # Count workspaces and projects
            print("\n📁 Counting workspaces and projects...", flush=True)
            results = counter.count_workspaces_and_projects(
                limit=args.limit, query=args.query
            )
            print_workspaces_and_projects(results)

        # Summary
        print(f"\n{'='*60}")
        print("📈 SUMMARY")
        print(f"{'='*60}")

        if not results.get("error"):
            if "workspaces" in results and not results["workspaces"].get("error"):
                print(f"✅ 🏢 Total Workspaces: {results['workspaces']['count']}")

            if "projects" in results and not results["projects"].get("error"):
                print(f"✅ 📁 Total Projects: {results['projects']['total_count']}")

            if "artifacts" in results and not results["artifacts"].get("error"):
                print(f"✅ 📦 Total Artifacts: {results['artifacts']['total_count']}")

            if "experiments" in results and not results["experiments"].get("error"):
                exp_count = results["experiments"]["total_count"]
                size_info = ""
                if "sizes" in results and results["sizes"].get("total_bytes", 0) > 0:
                    total_size = results["sizes"]["total_bytes"]
                    size_str = CometResourceCounter._format_size(None, total_size)
                    size_info = f" ({size_str})"
                print(f"✅ 🧪 Total Experiments: {exp_count}{size_info}")
        else:
            print(f"❌ Error occurred: {results['error']}")
            sys.exit(1)

    except Exception as e:
        print(f"\n💥 Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # Called via `python -m cometx.cli.list ...`
    # Called via `cometx list ...`
    main(sys.argv[1:])
