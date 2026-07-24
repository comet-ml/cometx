# -*- coding: utf-8 -*-
"""
03 - Build a single-experiment view with Experiment.create_view().

An experiment-level view is the layout you see on one experiment's own page.
The sections and panels are exactly the ones 02 uses, but the storage envelope
differs -- experiment views live in the chart-template scope, where the state
sits under ``view.v2["panels"]`` rather than ``view.v3``. Use
``build_experiment_view()`` instead of ``build_view()`` and cometx.views handles
the difference.
``APIExperiment.create_view()`` stamps the experiment key on for you, so you
never set ``experiment_key`` yourself.

The payoff is in --all: one layout, applied to every run in the project.

    python 03_build_experiment_view.py --workspace WS [--project view-api-demo]
    python 03_build_experiment_view.py --workspace WS --all
    python 03_build_experiment_view.py --dry-run
"""

import argparse
import json

import comet_ml

from cometx.utils import get_first_experiment, resolve_workspace
from cometx.views import (
    Section,
    build_experiment_view,
    data_panel,
    describe_view,
    line_panel,
    scalar_panel,
)


def build(name):
    """A layout aimed at reading one run in detail rather than comparing runs."""
    summary = Section(
        "Summary",
        columns=3,
        panels=[
            scalar_panel("accuracy", aggregation="last", description="Accuracy at the last step"),
            scalar_panel("val_accuracy", aggregation="max", description="Best validation accuracy"),
            scalar_panel("loss", aggregation="min", description="Lowest loss"),
        ],
    )

    convergence = Section(
        "Convergence",
        columns=2,
        panels=[
            line_panel(["loss", "val_loss"], name="Loss", smoothing=0.5),
            line_panel(["accuracy", "val_accuracy"], name="Accuracy", smoothing=0.5),
            # Overfitting check, held at its own settings by locked=True.
            line_panel("val_loss", name="Validation loss (unsmoothed)", locked=True),
            line_panel("loss", name="Loss, log scale", transform_y="log"),
        ],
    )

    # Full-width tabular output: one column, taller rows.
    outputs = Section(
        "Outputs",
        columns=1,
        height=2,
        panels=[data_panel("per_class_metrics.csv", name="Per-class metrics")],
    )

    return build_experiment_view(
        name,
        sections=[summary, convergence, outputs],
        x_axis="step",
        smoothing=0.2,
        sample_size=1000,
        # Images on an experiment page are sorted by these settings; a run's
        # logged media appears in the Images tab rather than as a panel.
        media_sort_by="name",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace")
    parser.add_argument("--project", default="view-api-demo")
    parser.add_argument("--name", default="Single-run Deep Dive")
    parser.add_argument("--all", action="store_true", help="apply to every experiment in the project")
    parser.add_argument("--dry-run", action="store_true", help="print the view instead of creating it")
    args = parser.parse_args()

    if args.dry_run:
        view = build(args.name)
        print(describe_view(view, indent="  "))
        print("\n--- v2 payload ---")
        print(json.dumps(view.v2, indent=2))
        return

    comet_ml.login()
    workspace = resolve_workspace(args.workspace)
    api = comet_ml.API()

    if args.all:
        targets = api.get_experiments(workspace, args.project)
        if not targets:
            raise SystemExit("No experiments in %s/%s." % (workspace, args.project))
    else:
        experiment = get_first_experiment(api, workspace, args.project)
        if experiment is None:
            raise SystemExit(
                "No experiments in %s/%s -- run 00_seed_demo_project.py first."
                % (workspace, args.project)
            )
        targets = [experiment]

    print("Applying %r to %d experiment(s)" % (args.name, len(targets)))
    for experiment in targets:
        # Build a fresh view per experiment so panel ids stay unique.
        created = experiment.create_view(build(args.name))
        print("  [%s] %s  %s" % ("OK" if created else "FAILED", experiment.id, experiment.name or ""))

    print("\nReading back with Experiment.get_views():")
    for view in targets[0].get_views():
        print(describe_view(view, indent="  "))
    print("\nOpen an experiment's page and pick the view from its selector.")


if __name__ == "__main__":
    main()
