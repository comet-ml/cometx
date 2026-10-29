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

One experiment view serves every run in its project: create it from any run,
and it shows up in the view selector on all of them. Rerunning updates it in
place. Experiment view names must be unique across all your projects, so a name
already used in another project is refused; pass --name to pick another.

    python 03_build_experiment_view.py --workspace WS [--project view-api-demo]
    python 03_build_experiment_view.py --dry-run
"""

import argparse
import json

import comet_ml

from cometx import API
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
            scalar_panel(
                "accuracy", aggregation="last", description="Accuracy at the last step"
            ),
            scalar_panel(
                "val_accuracy",
                aggregation="max",
                description="Best validation accuracy",
            ),
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
            # Locked too, or the page-wide transform setting resets transform_y.
            line_panel("loss", name="Loss, log scale", transform_y="log", locked=True),
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
    parser.add_argument("--name", default="SDK Run Layout")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the view instead of creating it"
    )
    args = parser.parse_args()

    if args.dry_run:
        view = build(args.name)
        print(describe_view(view, indent="  "))
        print("\n--- v2 payload ---")
        print(json.dumps(view.v2, indent=2))
        return

    comet_ml.login()
    api = API()
    workspace = args.workspace or api.get_default_workspace()

    experiments = api.get_experiments(workspace, args.project)
    if not experiments:
        raise SystemExit(
            "No experiments in %s/%s -- run 00_seed_demo_project.py first."
            % (workspace, args.project)
        )
    experiment = experiments[0]

    view = build(args.name)
    existing = [v for v in experiment.get_views() if v.name == args.name]
    if existing:
        view.template_id = existing[0].template_id
        saved = api.update_experiment_view(experiment, view)
    else:
        try:
            saved = api.create_experiment_view(experiment, view)
        except ValueError as error:
            raise SystemExit(str(error))
    if not saved:
        raise SystemExit("The backend rejected the view.")

    print("%s %r" % ("Updated" if existing else "Created", saved.name))
    print(describe_view(saved, indent="  "))
    print(
        "\nOpen: %s/%s/%s/%s?experiment-tab=panels&viewId=%s"
        % (
            api.server_url.rstrip("/"),
            workspace,
            args.project,
            experiment.id,
            saved.template_id,
        )
    )
    print("Every run in %s shows it in its view selector." % args.project)


if __name__ == "__main__":
    main()
