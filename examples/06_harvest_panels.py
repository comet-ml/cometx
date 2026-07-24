# -*- coding: utf-8 -*-
"""
06 - Reuse custom and Python panels found on existing dashboards.

Built-in panels (line, bar, scatter, ...) can be written from nothing. Custom
JavaScript panels and Python panels cannot: they reference a panel instance that
already exists in the Panel gallery, identified by an ``instanceId`` (custom) or
``templateId``/``revisionId`` (Python). Those ids are not invented -- they are
harvested from a dashboard that already uses the panel.

So this script scans views with API.get_views() and Experiment.get_views(),
collects every custom/Python panel it finds, and can rebuild them into a new
"Panel gallery" view -- the pattern for "take that neat panel someone built and
put it on my dashboards too".

    # list every custom/Python panel visible in a workspace
    python 06_harvest_panels.py --workspace WS

    # rebuild the harvested panels into one view in a target project
    python 06_harvest_panels.py --workspace WS --into view-api-demo --apply

    # harvest from one workspace, install into another
    python 06_harvest_panels.py --workspace SRC --into-workspace DST \
                                --into my-project --apply
"""

import argparse
import json

import comet_ml

from cometx.utils import resolve_workspace
from cometx.views import Section, build_view, custom_panel, describe_view, python_panel


def iter_panels(view):
    """Yield every panel dict in a view, from v3 sections or legacy chart_state."""
    if isinstance(view.v3, dict):
        for section in view.v3.get("sections") or []:
            for panel in section.get("panels") or []:
                yield panel
    try:
        charts = json.loads(view.chart_state or "{}").get("charts", {})
    except ValueError:
        return
    for bucket in charts.values():
        if isinstance(bucket, list):
            for panel in bucket:
                if isinstance(panel, dict):
                    yield panel


def harvest(api, workspace, projects):
    """Return {key: panel} for every distinct custom/Python panel found."""
    found = {}
    for project in projects:
        try:
            views = api.get_views(workspace, project)
        except Exception as error:  # a project can 404 mid-scan
            print("  ! %s: %s" % (project, error))
            continue
        for view in views:
            for panel in iter_panels(view):
                kind = panel.get("chartType")
                if kind == "custom" and panel.get("instanceId"):
                    key = ("custom", panel["instanceId"])
                elif kind == "python" and panel.get("templateId"):
                    key = ("python", panel["templateId"])
                else:
                    continue
                if key not in found:
                    found[key] = (panel, project, view.name)
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", help="workspace to harvest from")
    parser.add_argument("--projects", nargs="*", help="limit the scan to these projects")
    parser.add_argument("--limit", type=int, default=25, help="max projects to scan")
    parser.add_argument("--into", help="project that should receive the rebuilt view")
    parser.add_argument("--into-workspace", help="defaults to --workspace")
    parser.add_argument("--name", default="Panel Gallery")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    comet_ml.login()
    workspace = resolve_workspace(args.workspace)
    api = comet_ml.API()

    projects = args.projects or api.get_projects(workspace)[: args.limit]
    print("Scanning %d project(s) in %s for custom/Python panels...\n" % (len(projects), workspace))
    found = harvest(api, workspace, projects)

    if not found:
        print("No custom or Python panels found. Add one from the Panel gallery in the "
              "UI, then re-run -- its ids become reusable from the SDK.")
        return

    for (kind, identifier), (panel, project, view_name) in sorted(found.items()):
        label = panel.get("instanceName") or panel.get("chartName") or "(unnamed)"
        print("  %-7s %-40s %s" % (kind, label, identifier))
        print("          found in %s / view %r" % (project, view_name))

    if not args.into:
        print("\nPass --into PROJECT to rebuild these panels into a new view.")
        return

    section = Section("Harvested panels", columns=2, height=2)
    for (kind, identifier), (panel, _, _) in sorted(found.items()):
        if kind == "custom":
            section.add(custom_panel(
                identifier,
                instance_name=panel.get("instanceName", ""),
                default_config=panel.get("defaultConfig", ""),
            ))
        else:
            section.add(python_panel(
                identifier,
                revision_id=panel.get("revisionId", ""),
                name=panel.get("chartName", ""),
            ))

    view = build_view(args.name, sections=[section])
    print("\nComposed:")
    print(describe_view(view, indent="  "))

    if not args.apply:
        print("\nDry run -- re-run with --apply to create it.")
        return

    target_workspace = args.into_workspace or workspace
    created = api.create_view(target_workspace, args.into, view)
    print("\n[%s] %r in %s/%s" % ("OK" if created else "FAILED", args.name,
                                  target_workspace, args.into))
    if created and target_workspace != workspace:
        print("Note: a panel instance belongs to its own workspace. Copying one "
              "across workspaces only works if the panel is shared/public.")


if __name__ == "__main__":
    main()
