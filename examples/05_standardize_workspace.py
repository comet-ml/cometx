# -*- coding: utf-8 -*-
"""
05 - Push one standard dashboard into every project in a workspace.

The scaling demo: a team agrees on a house dashboard, and this script applies it
across a whole workspace -- adapting each project's panels to the metrics that
project actually logs, so no dashboard arrives full of empty charts.

Per project it:
  1. reads the project's metric names,
  2. builds a "Standard" view out of whichever standard metrics exist there,
  3. skips the project if that view already exists (unless --update),
  4. creates it with API.create_view().

    python 05_standardize_workspace.py --workspace WS              # dry run
    python 05_standardize_workspace.py --workspace WS --apply
    python 05_standardize_workspace.py --workspace WS --apply --update
    python 05_standardize_workspace.py --workspace WS --projects a b c --apply
"""

import argparse

import comet_ml

from cometx.utils import resolve_workspace
from cometx.views import (
    Section,
    build_view,
    global_config,
    line_panel,
    scalar_panel,
    table_state,
)

VIEW_NAME = "Standard"

# Metric name -> ("higher is better"?), in the order we like to see them.
STANDARD_METRICS = [
    ("loss", False),
    ("val_loss", False),
    ("accuracy", True),
    ("val_accuracy", True),
    ("f1", True),
    ("auc", True),
]

SYSTEM_METRICS = ["sys.cpu.percent.avg", "sys.ram.percent.used", "sys.gpu.0.gpu_utilization"]


def project_metric_names(api, workspace, project):
    """Union of metric names across the project's experiments (sampled)."""
    names = set()
    for experiment in api.get_experiments(workspace, project)[:5]:
        for metric in experiment.get_metrics_summary() or []:
            if isinstance(metric, dict) and metric.get("name"):
                names.add(metric["name"])
    return names


def build(available):
    """Build the standard view from whichever standard metrics are present."""
    present = [(name, higher_better) for name, higher_better in STANDARD_METRICS if name in available]
    if not present:
        return None

    tiles = [
        scalar_panel(name, aggregation="max" if higher_better else "min")
        for name, higher_better in present[:6]
    ]
    curves = [line_panel(name, name="%s over steps" % name, smoothing=0.4) for name, _ in present]

    sections = [
        Section("Key results", tiles, columns=3),
        Section("Curves", curves, columns=3),
    ]

    system = [m for m in SYSTEM_METRICS if m in available]
    if system:
        sections.append(Section("System", [line_panel(m, name=m) for m in system],
                                columns=3, expanded=False))

    return build_view(
        VIEW_NAME,
        sections=sections,
        table=table_state(
            columns=["Name", "experimentTags", "duration"] + [n for n, _ in present],
            sort_by=present[0][0],
            descending=present[0][1],
        ),
        config=global_config(smoothing=0.4, sample_size=500),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace")
    parser.add_argument("--projects", nargs="*", help="limit to these projects")
    parser.add_argument("--limit", type=int, help="stop after N projects")
    parser.add_argument("--update", action="store_true",
                        help="rebuild the view even if it already exists")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    comet_ml.login()
    workspace = resolve_workspace(args.workspace)
    api = comet_ml.API()

    projects = args.projects or api.get_projects(workspace)
    if args.limit:
        projects = projects[: args.limit]
    print("Workspace %s: %d project(s)\n" % (workspace, len(projects)))

    created = skipped = empty = 0
    for project in projects:
        existing = {v.name: v for v in api.get_views(workspace, project)}
        if VIEW_NAME in existing and not args.update:
            print("  %-40s already has %r" % (project, VIEW_NAME))
            skipped += 1
            continue

        view = build(project_metric_names(api, workspace, project))
        if view is None:
            print("  %-40s no standard metrics -- skipped" % project)
            empty += 1
            continue

        panels = sum(len(s["panels"]) for s in view.v3["sections"])
        if not args.apply:
            print("  %-40s would create %r (%d panels)" % (project, VIEW_NAME, panels))
            created += 1
            continue

        # Reusing the existing template_id updates that view in place instead of
        # adding a second one with the same name. create_view() cannot do this:
        # it calls as_portable(), which deliberately clears template_id, so an
        # in-place update has to go through the lower-level upsert.
        if VIEW_NAME in existing:
            view.template_id = existing[VIEW_NAME].template_id
            result = api._client.upsert_view(
                api.get_project(workspace, project)["projectId"], view
            )
            ok = bool(result)
        else:
            ok = bool(api.create_view(workspace, project, view))
        print("  %-40s [%s] %d panels" % (project, "OK" if ok else "FAILED", panels))
        created += ok

    print("\n%d created/updated, %d already present, %d without standard metrics"
          % (created, skipped, empty))
    if not args.apply:
        print("Dry run -- re-run with --apply.")


if __name__ == "__main__":
    main()
