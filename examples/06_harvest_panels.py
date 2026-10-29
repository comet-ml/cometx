# -*- coding: utf-8 -*-
"""
06 - Reuse code panels found on existing dashboards.

Code panels come in four kinds, and all four can be copied from one dashboard to
another by reading the panel off a view and rebuilding it:

  * a JavaScript panel (``javascript_panel``), code kept in the view,
  * a Python panel with its code kept in the view (``python_panel(code=...)``),
  * a Python panel added in the UI, which points at a template by
    ``templateId``/``revisionId``,
  * a Panel library panel placed in the UI, which points at a panel instance by
    ``instanceId`` (``custom_panel``).

The last two can't be created with an API key -- only reused, which is why this
script harvests them. It scans views with API.get_views(), collects every code
panel it finds, and can rebuild them into a "Panel Gallery" view: the pattern for
"take that neat panel someone built and put it on my dashboards too". Rerunning
with --apply updates the gallery in place.

    # list every code panel visible in a workspace
    python 06_harvest_panels.py --workspace WS

    # rebuild the harvested panels into one view in a target project
    python 06_harvest_panels.py --workspace WS --into view-api-demo --apply

    # harvest from one workspace, install into another
    python 06_harvest_panels.py --workspace SRC --into-workspace DST \
                                --into my-project --apply
"""

import argparse
import hashlib
import json

import comet_ml

from cometx import API
from cometx.views import (
    Section,
    build_view,
    custom_panel,
    describe_view,
    javascript_panel,
    python_panel,
)


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


def _digest(value):
    return hashlib.sha1(json.dumps(value, sort_keys=True).encode()).hexdigest()[:10]


def panel_key(panel):
    """A key that identifies a code panel, or None for any other panel."""
    kind = panel.get("chartType")
    if kind == "custom" and panel.get("instanceId"):
        return ("custom", panel["instanceId"])
    if kind == "python" and panel.get("code"):
        return ("python", _digest(panel["code"]))
    if kind == "python" and panel.get("templateId"):
        return ("python", panel["templateId"])
    if kind == "javascript" and isinstance(panel.get("code"), dict):
        return ("javascript", _digest(panel["code"]))
    return None


def rebuild(kind, identifier, panel):
    """A fresh copy of a harvested panel, with a new chartId."""
    name = panel.get("chartName", "")
    if kind == "custom":
        return custom_panel(
            identifier,
            instance_name=panel.get("instanceName", ""),
            name=name,
            default_config=panel.get("defaultConfig", ""),
        )
    if kind == "javascript":
        code = panel["code"]
        return javascript_panel(
            code.get("code", ""),
            html=code.get("html", ""),
            css=code.get("css", ""),
            resources=code.get("resources", []),
            name=name or "JavaScript panel",
        )
    if panel.get("code"):
        return python_panel(code=panel["code"], name=name)
    return python_panel(identifier, revision_id=panel.get("revisionId", ""), name=name)


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
                key = panel_key(panel)
                if key is None:
                    continue
                if key not in found:
                    found[key] = (panel, project, view.name)
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", help="workspace to harvest from")
    parser.add_argument(
        "--projects", nargs="*", help="limit the scan to these projects"
    )
    parser.add_argument("--limit", type=int, default=25, help="max projects to scan")
    parser.add_argument("--into", help="project that should receive the rebuilt view")
    parser.add_argument("--into-workspace", help="defaults to --workspace")
    parser.add_argument("--name", default="Panel Gallery")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    comet_ml.login()
    api = API()
    workspace = args.workspace or api.get_default_workspace()

    projects = (args.projects or api.get_projects(workspace))[: args.limit]
    print(
        "Scanning %d project(s) in %s for code panels...\n" % (len(projects), workspace)
    )
    found = harvest(api, workspace, projects)

    if not found:
        print(
            "No code panels found. Run 07_code_panels.py, or add a panel in the "
            "UI, then re-run."
        )
        return

    for (kind, identifier), (panel, project, view_name) in sorted(found.items()):
        label = panel.get("instanceName") or panel.get("chartName") or "(unnamed)"
        print("  %-10s %-40s %s" % (kind, label, identifier))
        print("          found in %s / view %r" % (project, view_name))

    if not args.into:
        print("\nPass --into PROJECT to rebuild these panels into a new view.")
        return

    section = Section("Harvested panels", columns=2, height=2)
    for (kind, identifier), (panel, _, _) in sorted(found.items()):
        section.add(rebuild(kind, identifier, panel))

    view = build_view(args.name, sections=[section])
    print("\nComposed:")
    print(describe_view(view, indent="  "))

    if not args.apply:
        print("\nDry run -- re-run with --apply to create it.")
        return

    target_workspace = args.into_workspace or workspace
    existing = [
        v for v in api.get_views(target_workspace, args.into) if v.name == args.name
    ]
    if existing:
        view.template_id = existing[0].template_id
        created = api.update_view(target_workspace, args.into, view)
    else:
        created = api.create_view(target_workspace, args.into, view)
    print(
        "\n[%s] %s %r in %s/%s"
        % (
            "OK" if created else "FAILED",
            "updated" if existing else "created",
            args.name,
            target_workspace,
            args.into,
        )
    )
    if created:
        print(
            "Open: %s/%s/%s/view/%s/panels"
            % (
                api.server_url.rstrip("/"),
                target_workspace,
                args.into,
                created.template_id,
            )
        )
    if created and target_workspace != workspace:
        print(
            "Note: a panel instance belongs to its own workspace. Copying one "
            "across workspaces only works if the panel is shared/public."
        )


if __name__ == "__main__":
    main()
