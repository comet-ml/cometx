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
#  Copyright (c) 2022 Cometx Development
#      Team. All rights reserved.
# ****************************************

import os
import random

from comet_ml import ExistingExperiment
from comet_ml.api import APIExperiment


def adjust_point(xyz):
    x, y, z = xyz
    # return [x, z, y]
    return [x, y, z]


def log_points(filename, experiment, points, boxes):
    existingExperiment = ExistingExperiment(previous_experiment=experiment.id)
    # log vertices and segments
    print(f"Logging {filename} to {experiment.get_name()}...")
    existingExperiment.log_points_3d(
        os.path.basename(filename) + ".c3d",  # comet 3d file format, zipped jsonl files
        points,
        boxes,
        step=0,
    )  # file type is "3d-image" in comet
    # Log the source data too:
    existingExperiment.log_asset(filename)
    existingExperiment.end()


def log_points_3d_pcd_file(experiment, filename):
    """
    FIELDS x y z                                # XYZ data
    FIELDS x y z rgb                            # XYZ + colors
    FIELDS x y z normal_x normal_y normal_z     # XYZ + surface normals
    FIELDS j1 j2 j3                             # moment invariants
    """
    points = []
    boxes = []
    fields = ["x", "y", "z"]  # default
    with open(filename) as fp:
        line = fp.readline()
        while line:
            line = line.strip()
            if line.startswith("FIELDS"):
                fields = line.split()[1:]
            elif not any(line.startswith(ch) for ch in "#VSTCWHVPD"):
                line = line.split("#", 1)[0]
                row = line.split()
                if len(fields) != len(row):
                    raise Exception(f"row {row} doesn't match fields {fields}")

                if fields == ["x", "y", "z"]:
                    xyz = adjust_point([float(num) for num in row])
                    points.append(xyz)
                elif fields == ["x", "y", "z", "rgb"]:
                    xyz = adjust_point([float(num) for num in row[:3]])
                    color = int(float(row[3]))
                    red = color & 0x0000FF >> 0
                    green = color & 0x00FF00 >> 8
                    blue = color & 0xFF0000 >> 16
                    points.append(xyz + [red, green, blue])
                elif fields == ["x", "y", "z", "normal_x", "normal_y", "normal_z"]:
                    print("ignoring normals on row")
                    xyz = adjust_point([float(num) for num in row[:3]])
                    points.append(xyz)
                elif fields == ["j1", "j2", "j3"]:
                    raise Exception("unable to handle j1, j2, j3 fields format")

            line = fp.readline()

    log_points(filename, experiment, points, boxes)


def log_points_3d_xyz_file(experiment, filename):
    """
    Log an XYZ file.
    """
    points = []
    boxes = []
    with open(filename) as fp:
        line = fp.readline()
        while line:
            line = line.strip()
            if not line.startswith("#"):
                line = line.split("#", 1)[0]
                raw_values = line.split()
                xyz = [float(num) for num in raw_values[0:3]]
                raw_color = raw_values[3:]
                if any("." in c for c in raw_color):
                    color = [int(float(num) * 255) for num in raw_color]
                else:
                    color = [int(num) for num in raw_color]
                if not color:
                    color = [255, 255, 255]
                points.append(adjust_point(xyz) + color)
            line = fp.readline()

    log_points(filename, experiment, points, boxes)


def log_points_3d_off_file(experiment, filename):
    """
    1. First line (optional): the letters OFF to mark the file type.
    2. Second line: the number of vertices, number of faces, and number
    of edges, in order (the latter can be ignored by writing 0 instead).

    Next V lines, number of vertices:

    V. List of vertices: X, Y and Z coordinates.


    V+1: List of faces: number of vertices, followed by the indexes of the
        composing vertices, in order (indexed from zero).

    Optionally, the RGB values for the face color can follow the elements
    of the faces.
    """
    points = []
    boxes = []
    vertices, faces, edges = 0, 0, 0
    mode = None
    with open(filename) as fp:
        line = fp.readline()
        count = 0
        while line:
            line = line.strip()
            if line.startswith("#") or line == "OFF":
                pass
            elif mode is None:
                vertices, faces, edges = [int(num) for num in line.split()]
                count = 0
                if vertices > 0:
                    mode = "vertices"
                elif faces > 0:
                    mode = "faces"
                else:
                    mode = "edges"
            elif mode == "vertices":
                count += 1
                line = line.split("#", 1)[0]
                values = adjust_point([float(num) for num in line.split()])
                points.append(values)
                if count == vertices:
                    count = 0
                    mode = "faces"
            elif mode == "faces":
                count += 1
                line = line.split("#", 1)[0]
                raw_values = line.split()
                vs = int(raw_values[0])
                segment = [int(num) for num in raw_values[1 : 1 + vs]]
                raw_color = raw_values[1 + vs :]
                if any("." in c for c in raw_color):
                    color = [int(float(num) * 255) for num in raw_color]
                else:
                    color = [int(num) for num in raw_color]
                if not color:
                    color = [255, 255, 255]
                box = {
                    "segments": [[points[s] for s in segment] + [points[segment[0]]]],
                    "score": 1,
                    "name": "Layer",
                    "label": "face",
                }
                if color:
                    box["color"] = color
                else:
                    box["color"] = [255, 255, 255]
                boxes.append(box)
                if count == faces:
                    count = 0
                    mode = "edges"
            elif mode == "edges":
                count += 1
                if count == edges:
                    break
            else:
                raise Exception("unknown mode %r" % mode)
            line = fp.readline()

    log_points(filename, experiment, points, boxes)


def log_tensorboard_folder_assets(workspace, project_name, full_path):
    from tensorboard.backend.event_processing import event_accumulator

    for run_dir in os.listdir(full_path):
        # here run_dir is toy.fit.experiment1.10.0.20240930.135419.21510_0
        experiment = APIExperiment(
            workspace=workspace,
            project_name=project_name,
        )
        print("Created new experiment: %s (%s)" % (experiment.url, run_dir))
        experiment.set_name(run_dir)

        print(f"Starting upload for {run_dir}...")
        experiment.log_other("tensorboard-log", run_dir)
        for subdir in os.listdir(os.path.join(full_path, run_dir)):
            newdir = os.path.join(full_path, run_dir, subdir)
            if not os.path.isdir(newdir):
                continue

            # Load the log file
            ea = event_accumulator.EventAccumulator(newdir)
            ea.Reload()  # Load the log data

            # Print available tags (e.g., scalar names, histogram names, etc.) -
            # example sent only has scalar data
            tags = ea.Tags()
            # print("Tags: %s" % tags)

            # Extract scalar metric data (e.g., 'loss' metric)
            if "scalars" in tags:
                if tags["scalars"]:
                    for metric_name in tags["scalars"]:
                        scalar_data = ea.Scalars(metric_name)
                        # Extract step, value, and wall_time for each scalar event
                        data = []
                        for event in scalar_data:
                            step = event.step
                            value = event.value
                            wall_time = event.wall_time
                            data.append(
                                {"value": value, "step": step, "wall_time": wall_time}
                            )
                        if len(data) > 15_000:
                            data = sorted(
                                random.sample(data, 15_000),
                                key=lambda item: item["step"],
                            )
                        for row in data:
                            experiment.log_metric(
                                f"{subdir}/{metric_name}",
                                row["value"],
                                step=row["step"],
                                timestamp=row["wall_time"],
                            )
                else:
                    print("No scalar data for this experiment")
        print(f"Upload complete for {run_dir}")
