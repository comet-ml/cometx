# -*- coding: utf-8 -*-
"""
00 - Seed a demo project.

Creates a handful of small experiments so that the view-building demos have
metrics, hyperparameters, images and tables to point their panels at. Nothing
here uses the view API; it just gives the other scripts something to visualize.

    python 00_seed_demo_project.py --workspace WS [--project view-api-demo]

Logs per experiment:
    metrics      loss, accuracy, val_loss, val_accuracy (over 60 steps)
    params       learning_rate, batch_size, optimizer, hidden_size
    image        prediction-grid.png
    table        per_class_metrics.csv
"""

import argparse
import math
import random

import comet_ml

from cometx.utils import resolve_workspace

OPTIMIZERS = ["adam", "sgd", "rmsprop"]


def log_one(workspace, project, index):
    experiment = comet_ml.Experiment(
        workspace=workspace,
        project_name=project,
        auto_metric_logging=False,
        auto_param_logging=False,
        display_summary_level=0,
    )
    learning_rate = 10 ** random.uniform(-4, -2)
    batch_size = random.choice([16, 32, 64, 128])
    optimizer = OPTIMIZERS[index % len(OPTIMIZERS)]
    hidden_size = random.choice([64, 128, 256])

    experiment.set_name("run-%02d-%s" % (index, optimizer))
    experiment.add_tags([optimizer, "seeded"])
    experiment.log_parameters(
        {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "optimizer": optimizer,
            "hidden_size": hidden_size,
        }
    )

    # A plausible-looking training curve driven by the hyperparameters.
    speed = 40 * learning_rate * math.log(hidden_size)
    for step in range(60):
        progress = 1 - math.exp(-speed * step / 10.0)
        noise = random.uniform(-0.02, 0.02)
        loss = max(0.01, 2.5 * (1 - progress) + noise)
        experiment.log_metric("loss", loss, step=step)
        experiment.log_metric("accuracy", min(0.999, 0.1 + 0.88 * progress + noise), step=step)
        experiment.log_metric("val_loss", loss * random.uniform(1.0, 1.3), step=step)
        experiment.log_metric("val_accuracy", min(0.999, 0.1 + 0.84 * progress + noise), step=step)

    try:
        import numpy

        grid = numpy.random.randint(0, 255, (64, 64, 3), dtype="uint8")
        experiment.log_image(grid, name="prediction-grid.png", step=59)
    except ImportError:
        print("  (numpy missing -- skipping the image; the image panel will be empty)")

    experiment.log_table(
        "per_class_metrics.csv",
        tabular_data=[
            ["class", "precision", "recall"],
            ["cat", round(random.uniform(0.7, 0.99), 3), round(random.uniform(0.7, 0.99), 3)],
            ["dog", round(random.uniform(0.7, 0.99), 3), round(random.uniform(0.7, 0.99), 3)],
            ["bird", round(random.uniform(0.7, 0.99), 3), round(random.uniform(0.7, 0.99), 3)],
        ],
        headers=False,
    )

    experiment.end()
    return experiment.get_key()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace")
    parser.add_argument("--project", default="view-api-demo")
    parser.add_argument("--count", type=int, default=6, help="experiments to log")
    args = parser.parse_args()

    comet_ml.login()
    workspace = resolve_workspace(args.workspace)
    print("Seeding %s/%s with %d experiment(s)" % (workspace, args.project, args.count))
    for index in range(args.count):
        key = log_one(workspace, args.project, index)
        print("  logged %s" % key)
    print("\nDone. Now run: python 02_build_project_dashboard.py --workspace %s --project %s"
          % (workspace, args.project))


if __name__ == "__main__":
    main()
