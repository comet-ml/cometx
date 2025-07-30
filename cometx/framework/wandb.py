# -*- coding: utf-8 -*-
# ****************************************
#                               __
#    _________  ____ ___  ___  / /__  __
#   / ___/ __ \/ __ `__ \/ _ \/ __/ |/ /
#  / /__/ /_/ / / / / / /  __/ /_ >   <
#  \___/\____/_/ /_/ /_/\___/\__//_/|_/
#
#     Copyright (c) 2023-2024 Cometx
#  Development Team. All rights reserved.
# ****************************************

import json
import math
import os
import re
import shutil
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import unquote

import comet_ml
import wandb
from comet_ml.annotations import Box, Layer
from comet_ml.cli_args_parse import _parse_cmd_args, _parse_cmd_args_naive
from comet_ml.data_structure import Histogram

from ..utils import download_url, remove_extra_slashes

MAX_METRIC_SAMPLES = 15_000


def clean_for_filename(name):
    return name.replace("/", "-").replace(":", "")


def get_json_value(item):
    if hasattr(item, "_json_dict"):
        return item._json_dict
    else:
        return item


class DownloadManager:
    def __init__(
        self,
        include=None,
        ignore=None,
        output=None,
        list_items=False,
        flat=False,
        ask=False,
        filename=None,
        asset_type=None,
        sync="all",
        debug=False,
        query=None,
        max_workers=1,
    ):
        self.api = wandb.Api(timeout=60)

        self.root = output if output is not None else os.getcwd()
        self.debug = debug
        self.flat = flat
        self.ask = ask
        self.filename = filename
        self.asset_type = asset_type
        self.sync = sync
        if max_workers > 1:
            self.queue = ThreadPoolExecutor(max_workers=max_workers)
        else:
            self.queue = None
        self.ignore = ignore if ignore else []
        self.include_experiments = None

    def download_file_task(self, path, file, doit=False):
        def task():
            with tempfile.TemporaryDirectory() as tmpdir:
                try:
                    file.download(root=tmpdir)
                    shutil.copy(os.path.join(tmpdir, file.name), path)
                except Exception as exc:
                    print(exc)
                    print(
                        "Unable to download %r to %r; skipping..." % (file.name, path)
                    )

        if self.queue is None or doit:
            # Do it now:
            task()
        else:
            # add to queue
            self.queue.submit(task)

    def end(self):
        if self.queue is not None:
            self.queue.shutdown(wait=True)

    def reset_run(self):
        self.asset_metadata = []
        self.annotations = []
        self.parameters = []

    def download(self, PATH):
        path = remove_extra_slashes(PATH)
        path_parts = path.split("/")

        if len(path_parts) == 1:
            projects = []
            workspace = path_parts[0]
            for wandb_project in self.api.projects(workspace):
                projects.append(wandb_project.name)
            self.include_experiments = None
        elif len(path_parts) == 2:
            workspace, project = path_parts
            projects = [project]
            self.include_experiments = None
        elif len(path_parts) == 3:
            workspace, project, experiment = path_parts
            projects = [project]
            self.include_experiments = [experiment]
        elif len(path_parts) == 4 and path_parts[2] == "runs":
            workspace, project, _, experiment = path_parts
            self.include_experiments = [experiment]
            projects = [project]
        else:
            raise Exception("invalid PATH: %r" % PATH)

        # Download items:
        for project in projects:
            # FIXME: PDF only show first exposed charts; ignore for now
            # if "reports" not in self.ignore:
            #    self.download_reports(workspace, project)
            self.download_runs(workspace, project)

    def get_path(self, run, *subdirs, filename):
        if self.flat:
            path = self.root
        else:
            workspace, project, experiment = run.path
            path = os.path.join(self.root, workspace, project, experiment, *subdirs)
        os.makedirs(path, exist_ok=True)
        if filename:
            path = os.path.join(path, filename)
            # Add to asset metadata:
            if (
                filename != "assets_metadata.jsonl"
                and len(subdirs) > 1
                and subdirs[0] == "assets"
            ):
                step = None
                log_as_filename = None
                base_filename, ext = os.path.splitext(filename)
                parts = base_filename.rsplit("_", 2)
                if len(parts) == 3 and parts[1].isdigit() and parts[0] != "boxes":
                    log_as_filename, step, _ = parts
                    log_as_filename += ext
                    step = int(step)
                self.asset_metadata.append(
                    {
                        "fileName": filename,
                        "logAsFileName": log_as_filename,
                        "type": subdirs[1],
                        "step": step,
                    }
                )
        return path

    def get_file_path(self, wandb_file):
        return "/".join(wandb_file.name.split("/")[:-1])

    def get_file_name(self, wandb_file):
        file_name = clean_for_filename(wandb_file.name.split("/")[-1])
        return file_name

    def download_cmd_parameters(self, run, args):
        print("    downloading parameters...")
        try:
            args = _parse_cmd_args(args)
        except ValueError:
            args = _parse_cmd_args_naive(args)

        if args:
            self.parameters = []
            for key, value in args.items():
                self.parameters.append(
                    {
                        "name": key,
                        "valueMax": value,
                        "valueMin": value,
                        "valueCurrent": value,
                        "editable": False,
                    }
                )

    def download_file(self, path, file):
        with tempfile.TemporaryDirectory() as tmpdir:
            file.download(root=tmpdir)
            shutil.copy(os.path.join(tmpdir, file.name), path)

    def download_data(self, path, data):
        with open(path, "w") as fp:
            fp.write(data + "\n")

    def download_model_graph(self, run, file):
        print("    downloading model graph...")
        path = self.get_path(run, "run", filename="graph_definition.txt")
        self.download_file_task(path, file)

    def download_metadata(self, run, info, workspace, project):
        print("    downloading metadata...")
        metadata = {
            "experimentName": run.name,
            "userName": info.get("username", "unknown"),
            "projectName": project,
            "workspaceName": workspace,
            "filePath": info.get("program", "unknown"),
            "fileName": (
                os.path.basename(info.get("program"))
                if info.get("program")
                else "unknown"
            ),
            "cometDownloadVersion": comet_ml.__version__,
        }
        path = self.get_path(run, filename="metadata.json")
        with open(path, "w") as fp:
            fp.write(json.dumps(metadata) + "\n")

    def download_model(self, run, file):
        print("    downloading model...")
        filename = self.get_file_name(file)
        path = self.get_path(run, "assets", "model-element", filename=filename)
        self.download_file_task(path, file)

    def download_image(self, run, file):
        filename = self.get_file_name(file)
        print("    downloading image %r..." % filename)
        path = self.get_path(run, "assets", "image", filename=filename)
        self.download_file_task(path, file)

    def download_audio(self, run, file):
        filename = self.get_file_name(file)
        print("    downloading audio %r..." % filename)
        path = self.get_path(run, "assets", "audio", filename=filename)
        self.download_file_task(path, file)

    def download_video(self, run, file):
        filename = self.get_file_name(file)
        print("    downloading video %r..." % filename)
        path = self.get_path(run, "assets", "video", filename=filename)
        self.download_file_task(path, file)

    def download_text(self, run, file):
        print("    downloading text...")
        filename = self.get_file_name(file)
        path = self.get_path(run, "assets", "text", filename=filename)
        self.download_file_task(path, file)

    def download_html(self, run, file):
        print("    downloading html...")
        filename = self.get_file_name(file)
        path = self.get_path(run, "assets", "html", filename=filename)
        self.download_file_task(path, file)

    def download_asset(self, run, file):
        print("    downloading asset...")
        filename = self.get_file_name(file)
        path = self.get_path(run, "assets", "asset", filename=filename)
        self.download_file_task(path, file)

    def download_asset_data(self, run, data, filename):
        path = self.get_path(run, "assets", "asset", filename=filename)
        self.download_data(path, data)

    def download_code(self, run, file):
        print("    downloading code...")
        filepath = self.get_file_path(file)
        filename = self.get_file_name(file)
        # NOTE: filepath starts with "code"
        if not filepath.startswith("code"):
            filepath = os.path.join("code", filepath)
        path = self.get_path(run, "run", filepath, filename=filename)
        self.download_file_task(path, file)

    def download_output(self, run, file):
        print("    downloading output...")
        path = self.get_path(run, "run", filename="output.txt")
        self.download_file_task(path, file)

    def download_dependencies(self, run, file):
        print("    downloading dependencies...")
        path = self.get_path(run, "run", filename="requirements.txt")
        self.download_file_task(path, file)

    def convert_annotation(self, data, i, width):
        """
        Convert wandb box_data into a standard format.
        """
        # data: {'box_data': [{'position': {'minX': 633, 'maxX': 734, 'minY': 246, 'maxY': 569}, 'class_id': 2, 'box_caption': 'person (99.720)', 'domain': 'pixel', 'scores': {'score': 99.71950650215149}}, {'position': {'minX': 521, 'maxX': 660, 'minY': 249, 'maxY': 609}, 'class_id': 2, 'box_caption': 'person (99.932)', 'domain': 'pixel', 'scores': {'score': 99.93243217468262}}, {'position': {'minX': 757, 'maxX': 776, 'minY': 320, 'maxY': 369}, 'class_id': 2, 'box_caption': 'person (84.234)', 'domain': 'pixel', 'scores': {'score': 84.23382639884949}}], 'class_labels': {'0': 'car', '1': 'truck', '2': 'person', '3': 'traffic light', '4': 'stop sign', '5': 'bus', '6': 'bicycle', '7': 'motorbike', '8': 'parking meter', '9': 'bench', '10': 'fire hydrant', '11': 'aeroplane', '12': 'boat', '13': 'train'}}
        # order tells which image
        boxes = []
        labels = data["class_labels"]
        for box in data["box_data"]:
            label = labels[str(box["class_id"])]
            score = box["scores"]["score"]
            boxes.append(
                Box(
                    box=(
                        (i * width) + box["position"]["minX"],
                        box["position"]["minY"],
                        box["position"]["maxX"] - box["position"]["minX"],
                        box["position"]["maxY"] - box["position"]["minY"],
                    ),
                    label=label,
                    score=score,
                )
            )
        return boxes

    def load_annotations(self, run, pathname, i, width):
        """
        Given a pathname, get the annotations
        """
        filename = os.path.basename(pathname)
        path = self.get_path(run, "assets", "asset", filename=filename)
        while not os.path.exists(path):
            print("Waiting for file...")
            time.sleep(10)
            # FIXME: may not be completed yet! Let's do this sync
        with open(path) as fp:
            # convert the wandb json boxes to standard format
            return self.convert_annotation(json.load(fp), i, width)

    def process_annotation(self, run, annotation):
        # FIXME: break up image into 1200 x 800 chunks, side-by-side and apply annotations
        # {'_type': 'images', 'count': 5, 'width': 1200, 'format': 'png', 'height': 800, 'all_boxes': [
        # {'predictions': {'path': 'media/metadata/boxes2D/boxes_0_72d6bd64.boxes2D.json', 'size': 793, '_type': 'boxes2D', 'sha256': '72d6bd64ffa727fcfc9c182d48e726c7578b6fd3e3444476e0bfec7ae7fb4d12'}},
        # {'predictions': {'path': 'media/metadata/boxes2D/boxes_0_9b5c0bbd.boxes2D.json', 'size': 3090, '_type': 'boxes2D', 'sha256': '9b5c0bbde53b8b27c01ba08404d0a34793bd851b59b0387ce668bf9028c390e8'}},
        # {'predictions': {'_type': 'boxes2D', 'sha256': '33c03f8630e76ee4d5371b0b7d55fd38d9610e883886da730f74268821d5d38f', 'path': 'media/metadata/boxes2D/boxes_0_33c03f86.boxes2D.json', 'size': 1849}},
        # {'predictions': {'_type': 'boxes2D', 'sha256': '1c5422d07299f48880800748ef98d62533f79bc8821b1782f4b44a6b8f8f5bf6', 'path': 'media/metadata/boxes2D/boxes_0_1c5422d0.boxes2D.json', 'size': 439}},
        # {'predictions': {'path': 'media/metadata/boxes2D/boxes_0_2ef7b7e5.boxes2D.json', 'size': 1305, '_type': 'boxes2D', 'sha256': '2ef7b7e59de532123ecbac12528b65c750a67cb22485848a19d3b7d916e82b8a'}}]}
        layer_boxes = defaultdict(list)
        image_name = None
        for i, layer in enumerate(annotation["all_boxes"]):
            for name in layer.keys():
                image_name, _ = os.path.splitext(os.path.basename(layer[name]["path"]))
                image_name, _ = image_name.rsplit("_", 1)
                boxes = self.load_annotations(
                    run, layer[name]["path"], i, annotation["width"]
                )
                if boxes:
                    layer_boxes[name].extend(boxes)
        # set metadata of image to be string of dict or None
        layers = []
        for key in layer_boxes:
            layers.append(
                Layer(
                    boxes=layer_boxes[key],
                    name=key,
                ).to_dict()
            )
        if layers:
            return image_name, layers
        else:
            return None

    def find_asset_metadata(self, image_name):
        for metadata in self.asset_metadata:
            if metadata["fileName"] == image_name:
                return metadata
        return None

    def download_asset_metadata(self, run):
        path = self.get_path(run, "assets", filename="assets_metadata.jsonl")
        # check to see if an image has annotations
        for annotation in self.annotations:
            results = self.process_annotation(run, annotation)
            if results:
                image_name, annotation = results
                metadata = self.find_asset_metadata(image_name + ".png")
                metadata["metadata"] = json.dumps({"annotations": annotation})
        with open(path, "w") as fp:
            for metadata in self.asset_metadata:
                fp.write(json.dumps(metadata) + "\n")

    def download_others(self, run, data):
        print("    downloading others...")
        filename = self.get_path(run, filename="others.jsonl")
        with open(filename, "w") as fp:
            for key, value in data.items():
                data_json = {
                    "name": key,
                    "valueMax": str(value),
                    "valueMin": str(value),
                    "valueCurrent": str(value),
                    "editable": False,
                }
                fp.write(json.dumps(data_json) + "\n")

    def download_runs(self, workspace, project):
        runs = self.api.runs(f"{workspace}/{project}")
        for run in reversed(runs):
            # Handle all of the basic run items here:
            _, _, experiment = run.path
            # Skip if experiment is not one to download:
            if not (
                self.include_experiments is None
                or experiment in self.include_experiments
            ):
                continue

            # Skip if already downloaded, and not sync:
            if (
                os.path.exists(os.path.join(workspace, project, experiment))
                and self.sync == "experiment"
            ):
                print(
                    f"Sync: skipping experiment {workspace}/{project}/{experiment}..."
                )
                continue

            print(
                f"downloading run '{run.name}' to {workspace}/{project}/{experiment}..."
            )
            self.reset_run()
            others = {
                "Name": run.name,
                "origin": run.url,
                "wandb_workspace": workspace,
                "wandb_project": project,
                "wandb_runid": experiment,
            }
            if "-run-" in run.name:
                group, count = run.name.rsplit("-run-", 1)
                others["Group"] = group
                others["Run"] = "run-" + count
            else:
                others["Group"] = "general"
                others["Run"] = run.name
            # Handle all of the specific run items below:
            if "metrics" not in self.ignore:
                self.download_metrics(run)

            for artifact in run.logged_artifacts():
                if artifact.type == "code":
                    for file in list(artifact.files()):
                        self.download_code(run, file)
                else:
                    for file in list(artifact.files()):
                        self.download_asset(run, file)

            for file in list(run.files()):
                path = self.get_file_path(file)
                name = file.name
                if name.startswith("artifact/"):
                    self.download_artifact(run, file)
                elif path == "media/graph" and "graph" not in self.ignore:
                    self.download_model_graph(run, file)
                elif path == "media/images" and "image" not in self.ignore:
                    # FIXME: bounding boxes, (bb and bit masks are saved in assets)
                    if "asset" not in self.ignore:
                        self.download_image(run, file)
                elif path == "media/audio" and "audio" not in self.ignore:
                    if "asset" not in self.ignore:
                        self.download_audio(run, file)
                elif path == "media/videos" and "video" not in self.ignore:
                    if "asset" not in self.ignore:
                        self.download_video(run, file)
                elif path == "media/text" and "text" not in self.ignore:
                    if "asset" not in self.ignore:
                        self.download_text(run, file)
                elif path == "media/html" and "html" not in self.ignore:
                    self.download_html(run, file)
                elif path.startswith("code") and "code" not in self.ignore:
                    self.download_code(run, file)
                elif name == "output.log" and "output" not in self.ignore:
                    self.download_output(run, file)
                elif name == "requirements.txt":
                    self.download_dependencies(run, file)
                elif name == "wandb-summary.json":
                    with tempfile.TemporaryDirectory() as tmpdirname:
                        summary = json.load(file.download(root=tmpdirname))
                        for item in summary:
                            if (
                                isinstance(summary[item], dict)
                                and "_type" in summary[item]
                            ):
                                if summary[item]["_type"] == "histogram":
                                    self.write_histogram(run, item, summary[item])
                                elif summary[item]["_type"].endswith("-file"):
                                    pass  # This is listed in assets
                                else:
                                    print(
                                        f"Ignoring {summary[item]['_type']} in summary"
                                    )
                        self.download_asset_data(
                            run, json.dumps(summary), "wandb_summary.json"
                        )
                elif name == "wandb-metadata.json":
                    # System info etc; only available to the owner?!
                    self.download_system_details(run, file, workspace, project)
                elif name == "diff.patch" and "git" not in self.ignore:
                    self.download_git_patch(run, file)
                elif (
                    any(
                        extension in name
                        for extension in [
                            ".pb",
                            ".onnx",
                            ".pkl",
                            ".mlmodel",
                            ".pmml",
                            ".pt",
                            ".h5",
                        ]
                    )
                    and "model" not in self.ignore
                ):
                    self.download_model(run, file)
                elif "asset" not in self.ignore:
                    self.download_asset(run, file)

            # After all of the file downloads, log others:
            self.download_others(run, others)
            self.download_hyper_parameters(run)
            self.write_parameters(run)
            self.download_asset_metadata(run)

    def download_git_patch(self, run, file):
        path = self.get_path(run, "run", filename="git_diff.patch")
        self.download_file_task(path, file)

    def download_hyper_parameters(self, run):
        value = dict(run.config)
        self.parameters.append(
            {
                "name": "config",
                "valueMax": value,
                "valueMin": value,
                "valueCurrent": value,
                "editable": False,
            }
        )

    def convert_histogram(self, data):
        if "bins" in data:
            bins = data["bins"][:-1]
        elif "packedBins" in data:
            bins = [data["packedBins"]["min"]]
            for i in range(data["packedBins"]["count"] - 1):
                bins.append(bins[-1] + data["packedBins"]["size"])
        else:
            raise Exception("unknown histogram type: %s" % data)
        return (bins, data["values"])

    def write_histogram(self, run, name, data):
        if "histogram_combined_3d" in self.ignore:
            return
        print("    downloading histogram...")
        values, counts = self.convert_histogram(data)
        histogram = Histogram()
        histogram.add(values=values, counts=counts)
        data_dict = {"histograms": [{"step": 0, "histogram": histogram.to_json()}]}
        name = clean_for_filename(name)
        path = self.get_path(
            run, "assets", "histogram_combined_3d", filename="%s_summary.json" % name
        )
        with open(path, "w") as fp:
            fp.write(json.dumps(data_dict) + "\n")

    def write_parameters(self, run):
        path = self.get_path(run, filename="parameters.json")
        with open(path, "w") as fp:
            fp.write(json.dumps(self.parameters) + "\n")

    def download_system_details(self, run, file, workspace, project):
        with tempfile.TemporaryDirectory() as tmpdirname:
            system_and_os_info = json.load(file.download(root=tmpdirname))

        args = system_and_os_info.get("args", [])
        system_details = {
            "command": (
                [system_and_os_info.get("program", "unknown")] + args
                if args
                else [system_and_os_info.get("program", "unknown")]
            ),
            "env": None,
            "hostname": system_and_os_info.get("host", "unknown"),
            "ip": "",
            "machine": "",
            "os": system_and_os_info.get("os", "unknown"),
            "pid": 0,
            "processor": "",
            "executable": system_and_os_info.get("executable", "unknown"),
            "pythonVersion": system_and_os_info.get("python", "unknown"),
            "user": system_and_os_info.get("username", "unknown"),
        }
        # FIXME: add system details
        """
        "cpu_count": 4,
        "cpu_count_logical": 8,
        "cpu_freq": {
          "current": 2513.417375,
          "min": 400.0,
          "max": 4000.0
        },
        "disk": {
          "/": {
            "total": 462.7395782470703,
            "used": 284.8777313232422
        }
        },
        "memory": {
          "total": 15.332221984863281
        }
        """
        path = self.get_path(run, filename="system_details.json")
        with open(path, "w") as fp:
            fp.write(json.dumps(system_details) + "\n")
        # ---
        self.download_cmd_parameters(run, args)
        self.download_metadata(run, system_and_os_info, workspace, project)
        # ---
        if "git" in system_and_os_info:
            commit = system_and_os_info["git"]["commit"]
            origin = system_and_os_info["git"]["remote"]
            git_metadata = {
                "parent": commit,
                "user": None,
                "root": None,
                "branch": None,
                "origin": origin,
            }
            path = self.get_path(run, "run", filename="git_metadata.json")
            with open(path, "w") as fp:
                fp.write(json.dumps(git_metadata) + "\n")
        """
        # FIXME:
        # Set the filename separately
        ## self.experiment.set_filename(system_and_os_info['program'])
        """
        # Log the entire file as well:
        path = self.get_path(run, "assets", "asset", filename="wandb-metadata.json")
        with open(path, "w") as fp:
            fp.write(json.dumps(system_and_os_info) + "\n")

    def download_artifact(self, run, file):
        _, artifact_id, artifact_name = file.name.split("/", 2)
        artifact_name = clean_for_filename(artifact_name)
        path = self.get_path(run, "artifacts", artifact_id, filename=artifact_name)
        self.download_file_task(path, file)

    def download_artifact_by_name(
        self,
        workspace,
        project,
        artifact_name,
        alias,
    ):
        # Utility, not used here
        """
        Example:

        ```python
        download_table(
            "stacey",
            "mnist-viz",
            "baseline",
            "v4",
        )
        ```
        """
        if self.flat:
            path = self.root
        else:
            path = os.path.join(self.root, workspace, project, "artifacts")

        os.makedirs(path, exist_ok=True)
        artifact = self.api.artifact(f"{workspace}/{project}/{artifact_name}:{alias}")
        artifact.download(path)

    def ignore_metric_name(self, metric):
        """
        Example:

        cometx download --from wandb ... --ignore "metrics:.*/avg"
        """
        if metric.startswith("_"):
            return True
        # Ignore any matches:
        for ignore in self.ignore:
            if ignore.startswith("metrics:"):
                _, regex = ignore.split(":", 1)
                if re.match(regex, metric):
                    return True
        return False

    def download_histograms(self, run, name):
        print("Downloading histograms %r..." % name)
        histograms = run.history(
            keys=[name, "_timestamp"],
            pandas=False,
            samples=MAX_METRIC_SAMPLES,
        )
        data_dict = {"histograms": []}
        for row in histograms:
            step = row.get("_step", None)
            histogram_data = row.get(name, None)
            if histogram_data:
                histogram = Histogram()
                values, counts = self.convert_histogram(histogram_data)
                histogram.add(values=values, counts=counts)
                data_dict["histograms"].append(
                    {"step": step, "histogram": histogram.to_json()}
                )
        name = clean_for_filename(name)
        path = self.get_path(
            run, "assets", "histogram_combined_3d", filename="%s_history.json" % name
        )
        with open(path, "w") as fp:
            fp.write(json.dumps(data_dict) + "\n")

    def download_metric_task(self, metric, run, count):
        def task():
            print("        downloading metric %r..." % metric)
            filename = self.get_path(
                run, "metrics", filename="metric_%05d.jsonl" % count
            )
            with open(filename, "w") as fp:
                for row in run.scan_history(
                    keys=[metric, "_step", "_timestamp", "epoch"]
                ):
                    step = row.get("_step", None)
                    epoch = row.get("epoch", None)
                    timestamp = row.get("_timestamp", None)
                    value = row.get(metric, None)
                    if (
                        metric is not None
                        and value is not None
                        and not math.isnan(value)
                    ):
                        ts = int(timestamp * 1000) if timestamp is not None else None
                        data = {
                            "metricName": metric,
                            "metricValue": value,
                            "timestamp": ts,
                            "step": step,
                            "epoch": epoch,
                            "runContext": None,
                        }
                        fp.write(json.dumps(data) + "\n")

        if self.queue is None:
            # Do it now:
            task()
        else:
            # add to queue
            self.queue.submit(task)

    def download_metrics(self, run):
        metrics_summary_path = self.get_path(run, filename="metrics_summary.jsonl")

        count = 0
        with open(metrics_summary_path, "w") as fp:
            if "system-metrics" not in self.ignore:
                system_metrics = run.history(stream="events", pandas=False)
                system_metric_names = set()
                for line in system_metrics:
                    for key in line.keys():
                        if key.startswith("_"):
                            continue
                        system_metric_names.add(key)

                for system_metric_name in system_metric_names:
                    fp.write(
                        json.dumps({"metric": system_metric_name, "count": count})
                        + "\n"
                    )
                    filename = self.get_path(
                        run, "metrics", filename="metric_%05d.jsonl" % count
                    )
                    with open(filename, "w") as metric_fp:
                        name = (
                            system_metric_name.replace("system.", "system/")
                            if system_metric_name.startswith("system.")
                            else system_metric_name
                        )
                        name = name.replace("\\.", "")
                        print("        downloading system metric %r..." % name)
                        for step, line in enumerate(system_metrics):
                            timestamp = line["_timestamp"]
                            ts = (
                                int(timestamp * 1000) if timestamp is not None else None
                            )
                            data = {
                                "metricName": name,
                                "metricValue": line.get(system_metric_name),
                                "timestamp": ts,
                                "step": step + 1,
                                "epoch": None,
                                "runContext": None,
                            }
                            metric_fp.write(json.dumps(data) + "\n")
                    count += 1

            if "summary-metrics" not in self.ignore:
                # Next, log single-value from summary:
                summary = {}
                for item in run.summary.keys():
                    if item.startswith("_"):
                        continue

                    value = get_json_value(run.summary[item])

                    if isinstance(value, dict):
                        if item == "boxes":
                            self.annotations.append(value)
                            continue
                        if "_type" in value and value["_type"] in ["histogram"]:
                            continue

                    summary[item] = value

                self.download_asset_data(
                    run, json.dumps(summary), "summary_metrics.json"
                )

            print("Gathering metrics...")
            metrics = set()
            histograms = set()
            for row in run.scan_history():
                for metric in row:
                    if self.ignore_metric_name(metric):
                        continue

                    if isinstance(row[metric], dict):
                        if (
                            "_type" in row[metric]
                            and row[metric]["_type"] == "histogram"
                        ):
                            histograms.add(metric)
                    else:
                        if row[metric] is not None:
                            metrics.add(metric)
            print("")
            print("Done gathering metrics")

            if "histogram_combined_3d" not in self.ignore:
                for histogram in histograms:
                    self.download_histograms(run, histogram)

            for metric in metrics:
                fp.write(json.dumps({"metric": metric, "count": count}) + "\n")
                self.download_metric_task(metric, run, count)
                count += 1

    def download_reports(self, workspace, project):
        if self.flat:
            path = self.root
        else:
            path = os.path.join(self.root, workspace, project, "reports")

        os.makedirs(path, exist_ok=True)
        wandb_path = workspace + "/" + project
        reports = self.api.reports(path=wandb_path)
        for report in reports:
            url = report.url
            report_name = unquote(url.split("/")[-1] + ".pdf")
            filepath = os.path.join(path, report_name)
            download_url(url, output_filename=filepath)
            report_data = {
                "workspace": workspace,
                "project": project,
                "url": url,
            }
            with open(os.path.join(path, "reports_metadata.jsonl"), "a+") as fp:
                fp.write(json.dumps(report_data))
                fp.write("\n")
