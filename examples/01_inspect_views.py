# -*- coding: utf-8 -*-
"""
01 - Read views with API.get_views() and Experiment.get_views().

Shows what each getter returns and how the two scopes differ:

  API.get_views(workspace, project)      project-level dashboards (the entries
                                         in the project's view selector)
  APIExperiment.get_views()              views scoped to a single experiment,
                                         from both the dashboard-template and
                                         the chart-template tables

and what the widening flags do:

  API.get_views(..., include_workspace_views=True)
  APIExperiment.get_views(include_workspace_views=True,
                          include_workspace_project_views=True)

    python 01_inspect_views.py --workspace WS [--project view-api-demo]
    python 01_inspect_views.py --workspace WS --json    # dump raw state
"""

import argparse
import json

import comet_ml

from cometx.utils import resolve_workspace
from cometx.views import describe_view


def dump_raw(view):
    """Print every field of a View, unpacking the serialized JSON blobs."""
    print("  name              %r" % view.name)
    print("  template_id       %s" % view.template_id)
    print("  project_id        %s" % view.project_id)
    print("  experiment_key    %s" % view.experiment_key)
    print("  view_source       %s" % view.view_source)
    print("  created_by        %s" % view.created_by)
    print("  project_default   %s" % view.project_default)
    print("  personal_default  %s" % view.project_personal_default)
    print("  auto_refresh      %s" % view.auto_refresh_enabled)
    print("  last_update       %s" % view.last_update)
    print("  pinned            %s" % view.pinned_experiments)
    for field in ("query_state", "chart_state", "table_state"):
        raw = getattr(view, field)
        if raw:
            print("  %s:" % field)
            print(_indent(json.dumps(json.loads(raw), indent=2), 4))
    for field in ("v2", "v3"):
        raw = getattr(view, field)
        if raw:
            print("  %s:" % field)
            print(_indent(json.dumps(raw, indent=2), 4))


def _indent(text, spaces):
    pad = " " * spaces
    return "\n".join(pad + line for line in text.splitlines())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace")
    parser.add_argument("--project", default="view-api-demo")
    parser.add_argument("--json", action="store_true", help="dump every field of every view")
    args = parser.parse_args()

    comet_ml.login()
    workspace = resolve_workspace(args.workspace)
    api = comet_ml.API()

    print("=" * 72)
    print("API.get_views(%r, %r)" % (workspace, args.project))
    print("=" * 72)
    project_views = api.get_views(workspace, args.project)
    print("%d project-level view(s)\n" % len(project_views))
    for view in project_views:
        print(describe_view(view, indent="  "))
        if args.json:
            dump_raw(view)
        print()

    # The same call, widened to the whole workspace. Useful for finding a
    # dashboard you built in another project and want to reuse here.
    workspace_views = api.get_views(workspace, args.project, include_workspace_views=True)
    extra = len(workspace_views) - len(project_views)
    print("include_workspace_views=True adds %d view(s) from sibling projects" % max(0, extra))
    for view in workspace_views[len(project_views):][:10]:
        print("  - %r (project_id=%s)" % (view.name, view.project_id))
    if extra > 10:
        print("  ... and %d more" % (extra - 10))
    print()

    experiments = api.get_experiments(workspace, args.project)
    if not experiments:
        print("No experiments in this project -- skipping the experiment-level part.")
        return
    experiment = experiments[0]

    print("=" * 72)
    print("APIExperiment.get_views()  [experiment %s]" % experiment.id)
    print("=" * 72)
    experiment_views = experiment.get_views()
    print("%d experiment-level view(s)\n" % len(experiment_views))
    for view in experiment_views:
        # view_source tells you which table the view came from:
        #   "dashboard_template" -> the project dashboard tables
        #   "chart_template"     -> the single-experiment chart tables
        print(describe_view(view, indent="  "))
        if args.json:
            dump_raw(view)
        print()

    widened = experiment.get_views(
        include_workspace_views=True,
        include_workspace_project_views=True,
    )
    print("widened to the workspace: %d view(s)" % len(widened))


if __name__ == "__main__":
    main()
