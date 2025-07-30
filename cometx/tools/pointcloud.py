# -*- coding: utf-8 -*-
# ****************************************
#                              __
#   _________  ____ ___  ___  / /__  __
#  / ___/ __ \/ __ `__ \/ _ \/ __/ |/_/
# / /__/ /_/ / / / / / /  __/ /__>  <
# \___/\____/_/ /_/ /_/\___/\__/_/|_|
#
#
#  Copyright (c) 2023 Cometx Development
#      Team. All rights reserved.
# ****************************************

import json
import math
import os
import pathlib
import tempfile
from collections import defaultdict

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image, ImageDraw = None, None

## 3D Graphics functions


def identity():
    """
    Return matrix for identity (no transforms).
    """
    return [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]


def rotate_x(angle):
    """
    Return transform matrix for rotation around x axis.
    """
    radians = angle * math.pi / 180
    return [
        [1, 0, 0, 0],
        [0, math.cos(radians), -math.sin(radians), 0],
        [0, math.sin(radians), math.cos(radians), 0],
        [0, 0, 0, 1],
    ]


def rotate_y(angle):
    """
    Return transform matrix for rotation around y axis.
    """
    radians = angle * math.pi / 180
    return [
        [math.cos(radians), 0, math.sin(radians), 0],
        [0, 1, 0, 0],
        [-math.sin(radians), 0, math.cos(radians), 0],
        [0, 0, 0, 1],
    ]


def rotate_z(angle):
    """
    Return transform matrix for rotation around z axis.
    """
    radians = angle * math.pi / 180
    return [
        [math.cos(radians), -math.sin(radians), 0, 0],
        [math.sin(radians), math.cos(radians), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]


def translate_xyz(x, y, z):
    """
    Return transform matrix for translation (linear moving).
    """
    return [
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1],
    ]


def scale_xyz(x, y, z):
    """
    Return transform matrix for scaling.
    """
    return [
        [x, 0, 0, 0],
        [0, y, 0, 0],
        [0, 0, z, 0],
        [0, 0, 0, 1],
    ]


def matmul(a, b):
    """
    Multiply two matrices. Written in Pure Python
    to avoid dependency on numpy.
    """
    c = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    for x in range(4):
        for y in range(4):
            acc = 0
            for n in range(4):
                acc += a[n][x] * b[y][n]
            c[y][x] = acc
    return c


def multiply_point_by_matrix(matrix, point):
    """
    Multiply a point by a matrix. Written in Pure Python
    to avoid dependency on numpy.
    """
    return [
        (point[0] * matrix[0][0])
        + (point[1] * matrix[0][1])
        + (point[2] * matrix[0][2])
        + (1 * matrix[0][3]),
        (point[0] * matrix[1][0])
        + (point[1] * matrix[1][1])
        + (point[2] * matrix[1][2])
        + (1 * matrix[1][3]),
        (point[0] * matrix[2][0])
        + (point[1] * matrix[2][1])
        + (point[2] * matrix[2][2])
        + (1 * matrix[2][3]),
    ]


def point_to_canvas(size, point, z=False):
    """
    Convert to screen coordinates (flip horizontally)
    Only return the first two values [x, y] of point
    """
    if z:
        return [int(size[0] - point[0]), int(point[1]), point[2]]
    else:
        return [int(size[0] - point[0]), int(point[1])]


def draw_line(size, canvas, transform, a, b, color):
    """
    Draw a line on the canvas given two points and transform.
    """
    ta = point_to_canvas(size, multiply_point_by_matrix(transform, a))
    tb = point_to_canvas(size, multiply_point_by_matrix(transform, b))
    canvas.line(ta + tb, fill=color)


def draw_point(size, canvas, transform, point, color):
    """
    Draw a point on the canvas given the transform.
    """
    p = point_to_canvas(size, multiply_point_by_matrix(transform, point))
    canvas.point(p, fill=color)


def draw_point_fake(size, fcanvas, transform, point, color):
    """
    Draw a point on the canvas given the transform.
    """
    p = point_to_canvas(size, multiply_point_by_matrix(transform, point), z=True)
    location = fcanvas[(p[0], p[1])]
    if location is None or location["z"] < p[2]:
        fcanvas[(p[0], p[1])] = {"z": p[2], "color": color}


def render(
    points_filename, boxes_filename, x, y, z, min_max_x, min_max_y, min_max_z, size
):
    """
    Given to files, points and boxes, rotations (in degrees) on x, y ,z, and ranges,
    create an image.
    """
    if Image is None:
        raise Exception("Python Image Library is not installed; pip install PIL")

    background_color = (51, 51, 77)  # Skybox color

    image = Image.new("RGB", size, background_color)
    canvas = ImageDraw.Draw(image)

    midpoint = [
        (min_max_x[0] + min_max_x[1]) / 2,
        (min_max_y[0] + min_max_y[1]) / 2,
        (min_max_z[0] + min_max_z[1]) / 2,
    ]

    scale = min(
        size[0] / abs(min_max_x[0] - min_max_x[1]),
        size[1] / abs(min_max_y[0] - min_max_y[1]),
    )
    transform = identity()
    # First, center it around zero:
    transform = matmul(transform, translate_xyz(*[-n for n in midpoint]))
    # Now, apply rotations:
    transform = matmul(transform, rotate_z(z))
    transform = matmul(transform, rotate_x(x))
    transform = matmul(transform, rotate_y(y))
    # And then scale
    transform = matmul(transform, scale_xyz(scale, scale, scale))
    # Finally, put it in center of window:
    transform = matmul(transform, translate_xyz(size[0] / 2, size[1] / 2, 0))

    # Fake canvas:
    fcanvas = defaultdict(lambda: None)

    # Draw points first
    with open(points_filename) as fp:
        line = fp.readline()
        while line:
            data = json.loads(line)
            # Each data can be [x, y, z] or [x, y, z, r, g, b]
            # r, g, b is given between 0 and 255 (floats are ok)
            point = data[:3]
            if len(data) > 3:
                color = tuple([int(round(c)) for c in data[3:]])
            else:
                # Default color is white
                color = (255, 255, 255)
            draw_point_fake(size, fcanvas, transform, point, color)
            line = fp.readline()

    # draw fake on canvas
    if fcanvas:
        for x, y in fcanvas:
            color = fcanvas[(x, y)]["color"]
            canvas.point((x, y), fill=color)

    # Draw boxes last to show on top of points
    with open(boxes_filename) as fp:
        line = fp.readline()
        while line:
            data = json.loads(line)
            # Each data is {"segments": [...], "name": "prediction", "color": [r, g, b],
            # "score": Number, "label": "pedestrian"}
            # Each segment is a list of lines,  which is a list of points, which is [x, y, z]
            if "color" in data and data["color"]:
                color = tuple(data["color"])
            else:
                color = (255, 255, 255)  ## default color is white
            for points in data["segments"]:
                point1 = points[0]
                for point2 in points[1:]:
                    draw_line(size, canvas, transform, point1, point2, color)
                    point1 = point2
            line = fp.readline()

    return image


def create_image(
    points=None,
    boxes=None,
    x=45,
    y=0,
    z=45,
    output_filename="pointcloud.gif",
    swap_yz=True,
    x_incr=0,
    y_incr=0,
    z_incr=0,
    steps=0,
    size=(250, 250),
):
    """ """
    min_max_x = [float("inf"), float("-inf")]
    min_max_y = [float("inf"), float("-inf")]
    min_max_z = [float("inf"), float("-inf")]

    if isinstance(points, (str, pathlib.Path)):
        points = (json.loads(line) for line in open(points))
    elif points is None:
        points = []

    if isinstance(boxes, (str, pathlib.Path)):
        boxes = (json.loads(line) for line in open(boxes))
    elif boxes is None:
        boxes = []

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as points_fp:
        for point in points:
            if swap_yz:
                point[1], point[2] = point[2], point[1]
            min_max_x = min(point[0], min_max_x[0]), max(point[0], min_max_x[1])
            min_max_y = min(point[1], min_max_y[0]), max(point[1], min_max_y[1])
            min_max_z = min(point[2], min_max_z[0]), max(point[2], min_max_z[1])
            points_fp.write(json.dumps(point))
            points_fp.write("\n")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as boxes_fp:
        for box in boxes:
            for points in box["segments"]:
                for point in points:
                    if swap_yz:
                        point[1], point[2] = point[2], point[1]
                    min_max_x = min(point[0], min_max_x[0]), max(point[0], min_max_x[1])
                    min_max_y = min(point[1], min_max_y[0]), max(point[1], min_max_y[1])
                    min_max_z = min(point[2], min_max_z[0]), max(point[2], min_max_z[1])
            boxes_fp.write(json.dumps(box))
            boxes_fp.write("\n")

    if steps == 0:
        image = render(
            points_fp.name,
            boxes_fp.name,
            x,
            y,
            z,
            min_max_x,
            min_max_y,
            min_max_z,
            size,
        )
        print(f"Saving to '{output_filename}'...")
        image.save(output_filename)
        return image
    else:
        images = []
        for step in range(steps):
            image = render(
                points_fp.name,
                boxes_fp.name,
                x,
                y,
                z,
                min_max_x,
                min_max_y,
                min_max_z,
                size,
            )
            images.append(image)
            x += x_incr
            y += y_incr
            z += z_incr

        for step in range(steps):
            image = render(
                points_fp.name,
                boxes_fp.name,
                x,
                y,
                z,
                min_max_x,
                min_max_y,
                min_max_z,
                size,
            )
            images.append(image)
            x -= x_incr
            y -= y_incr
            z -= z_incr

        print(f"Saving animation to '{output_filename}'...")
        images[0].save(
            output_filename,
            save_all=True,
            append_images=images[1:],
            optimize=True,
            duration=80,
            loop=0,
        )
        return images
