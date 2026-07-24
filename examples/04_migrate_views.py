# -*- coding: utf-8 -*-
"""
04 - Copy views between projects, workspaces, or accounts.

Combines all four functions: read with API.get_views() and
Experiment.get_views(), write with API.create_view() and
Experiment.create_view().

The one thing to know: ``View.as_portable()`` strips the identifiers that only
mean something in the source project (template_id, project_id, experiment_key,
pinned experiment keys). ``create_view()`` calls it for you, so a view fetched
from one project can be handed straight to another.

    # dry run (default) -- lists what would be copied
    python 04_migrate_views.py --src-workspace A --src-project p1 \
                               --dst-workspace B --dst-project p2

    # actually copy
    python 04_migrate_views.py ... --apply

    # different accounts: give the destination its own key
    python 04_migrate_views.py ... --dst-api-key KEY --apply

    # only some views, and skip the transient UI ones
    python 04_migrate_views.py ... --match "Weekly" --apply
"""

import argparse

import comet_ml

from cometx.views import describe_view

# Views the UI creates as scratch state; rarely worth copying.
TRANSIENT_NAMES = {"Unsaved Changes"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-workspace", required=True)
    parser.add_argument("--src-project", required=True)
    parser.add_argument("--dst-workspace", required=True)
    parser.add_argument("--dst-project", required=True)
    parser.add_argument("--src-api-key", help="defaults to the configured key")
    parser.add_argument("--dst-api-key", help="defaults to the source key")
    parser.add_argument("--match", help="only copy views whose name contains this")
    parser.add_argument("--include-transient", action="store_true",
                        help="also copy 'Unsaved Changes' views")
    parser.add_argument("--experiment-views", action="store_true",
                        help="also copy the first experiment's own views")
    parser.add_argument("--apply", action="store_true", help="perform the copy")
    return parser.parse_args()


def wanted(view, args):
    if not args.include_transient and view.name in TRANSIENT_NAMES:
        return False
    if args.match and args.match.lower() not in view.name.lower():
        return False
    return True


def main():
    args = parse_args()
    comet_ml.login()

    source = comet_ml.API(api_key=args.src_api_key) if args.src_api_key else comet_ml.API()
    destination = comet_ml.API(api_key=args.dst_api_key) if args.dst_api_key else source

    print("=== Project-level views in %s/%s ===" % (args.src_workspace, args.src_project))
    project_views = [v for v in source.get_views(args.src_workspace, args.src_project)
                     if wanted(v, args)]
    for view in project_views:
        print(describe_view(view, indent="  "))
    if not project_views:
        print("  (none matched)")

    experiment_views = []
    if args.experiment_views:
        experiments = source.get_experiments(args.src_workspace, args.src_project)
        if experiments:
            print("\n=== Experiment-level views on %s ===" % experiments[0].id)
            experiment_views = [v for v in experiments[0].get_views() if wanted(v, args)]
            for view in experiment_views:
                print(describe_view(view, indent="  "))
            if not experiment_views:
                print("  (none matched)")

    if not args.apply:
        print("\nDry run. Re-run with --apply to copy %d project view(s)%s to %s/%s."
              % (len(project_views),
                 " and %d experiment view(s)" % len(experiment_views) if experiment_views else "",
                 args.dst_workspace, args.dst_project))
        return

    print("\n=== Creating in %s/%s ===" % (args.dst_workspace, args.dst_project))
    for view in project_views:
        # as_portable() is applied inside create_view(); call it explicitly only
        # if you want to inspect or further edit the portable copy first.
        created = destination.create_view(args.dst_workspace, args.dst_project, view)
        print("  [%s] %r" % ("OK" if created else "FAILED", view.name))

    if experiment_views:
        targets = destination.get_experiments(args.dst_workspace, args.dst_project)
        if not targets:
            print("  no experiments in the destination -- skipping experiment views")
            return
        target = targets[0]
        print("\n=== Creating experiment views on %s ===" % target.id)
        for view in experiment_views:
            created = target.create_view(view)
            print("  [%s] %r" % ("OK" if created else "FAILED", view.name))

    print("\nDone.")


if __name__ == "__main__":
    main()
