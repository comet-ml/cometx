# -*- coding: utf-8 -*-
"""
02 - Build a project dashboard from scratch with API.create_view().

This is the widest demonstration: one script that creates a multi-section
dashboard containing every panel type the SDK can express, plus a saved filter,
a configured experiment table, and dashboard-wide axis/smoothing settings.

    python 02_build_project_dashboard.py --workspace WS [--project view-api-demo]
    python 02_build_project_dashboard.py --workspace WS --dry-run   # print JSON

Run 00_seed_demo_project.py first so the panels have data to draw.
"""

import argparse
import json

import comet_ml

from cometx.utils import get_project_url, resolve_workspace
from cometx.views import (
    Section,
    all_of,
    any_of,
    bar_panel,
    build_view,
    data_panel,
    describe_view,
    global_config,
    image_panel,
    legend_key,
    line_panel,
    parallel_panel,
    query_state,
    rule,
    scalar_panel,
    scatter_panel,
    table_state,
)


def build(name):
    """Compose the dashboard. Returns a View; makes no network calls."""

    # --- Section 1: headline numbers -------------------------------------
    headline = Section(
        "At a glance",
        columns=3,
        height=1,
        panels=[
            scalar_panel("accuracy", aggregation="max", description="Best accuracy reached"),
            scalar_panel("loss", aggregation="min", description="Lowest training loss"),
            scalar_panel("val_accuracy", aggregation="last", description="Final validation accuracy"),
        ],
    )

    # --- Section 2: training curves --------------------------------------
    curves = Section(
        "Training curves",
        columns=2,
        panels=[
            # Two metrics overlaid in a single chart.
            line_panel(["loss", "val_loss"], name="Loss: train vs validation", smoothing=0.6),
            line_panel(["accuracy", "val_accuracy"], name="Accuracy: train vs validation", smoothing=0.6),
            # Log-scale y-axis, and a legend keyed on a hyperparameter.
            line_panel(
                "loss",
                name="Loss (log scale), colored by optimizer",
                transform_y="log",
                legend=[legend_key("Name"), legend_key("optimizer", source="params")],
            ),
            # x-axis other than step; locked so the dashboard-wide x-axis
            # setting below does not override it.
            line_panel("accuracy", x="duration", name="Accuracy over wall-clock", locked=True),
        ],
    )

    # --- Section 3: comparing runs ---------------------------------------
    comparison = Section(
        "Run comparison",
        columns=3,
        panels=[
            bar_panel("accuracy", name="Final accuracy per run", aggregation="last"),
            bar_panel(
                "val_accuracy",
                name="Validation accuracy spread by optimizer",
                aggregation="last",
                plot_type="BOX",
                group_by_aggregation="mean",
            ),
            scatter_panel(
                "learning_rate",
                "accuracy",
                name="Learning rate vs accuracy",
                params=["learning_rate"],
                metrics=["accuracy"],
            ),
            scatter_panel(
                "loss",
                "val_loss",
                z="accuracy",
                name="3D: loss / val_loss / accuracy",
            ),
            parallel_panel(
                params=["learning_rate", "batch_size", "hidden_size", "optimizer"],
                metrics=["accuracy"],
                target="accuracy",
                name="Hyperparameter sweep",
            ),
        ],
    )

    # --- Section 4: media and tables, collapsed by default ---------------
    artifacts = Section(
        "Media & tables",
        columns=2,
        height=2,
        expanded=False,
        panels=[
            image_panel(["prediction-grid.png"], name="Prediction grids"),
            data_panel("per_class_metrics.csv", name="Per-class metrics"),
        ],
    )

    return build_view(
        name,
        sections=[headline, curves, comparison, artifacts],
        # Only show runs that got somewhere: accuracy > 0.5 OR loss < 0.5.
        filters=query_state(
            any_of(
                all_of(rule("accuracy", "greater", 0.5)),
                all_of(rule("loss", "less", 0.5)),
            )
        ),
        table=table_state(
            columns=[
                "Name",
                "experimentTags",
                "optimizer",
                "learning_rate",
                "batch_size",
                "accuracy",
                "val_accuracy",
                "duration",
            ],
            sort_by="accuracy",
            page_size=50,
        ),
        config=global_config(x_axis="step", smoothing=0.3, sample_size=500),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace")
    parser.add_argument("--project", default="view-api-demo")
    parser.add_argument("--name", default="SDK Full Dashboard")
    parser.add_argument("--dry-run", action="store_true", help="print the view instead of creating it")
    args = parser.parse_args()

    view = build(args.name)

    print("Composed view:")
    print(describe_view(view, indent="  "))

    if args.dry_run:
        print("\n--- v3 payload ---")
        print(json.dumps(view.v3, indent=2))
        return

    comet_ml.login()
    workspace = resolve_workspace(args.workspace)
    api = comet_ml.API()

    created = api.create_view(workspace, args.project, view)
    if not created:
        raise SystemExit("create_view returned nothing -- the backend rejected the view.")

    print("\nCreated %r (template_id=%s)" % (created.name, created.template_id))
    print("Open: %s" % get_project_url(workspace, args.project))
    print("(Pick the view from the view selector at the top of the project page.)")


if __name__ == "__main__":
    main()
