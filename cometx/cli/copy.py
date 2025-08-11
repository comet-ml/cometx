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
#  Copyright (c) 2023 Cometx Development
#      Team. All rights reserved.
# ****************************************
"""
To copy experiment data to new experiments.

cometx copy [--symlink] SOURCE DESTINATION

where SOURCE is:

* if not --symlink, "WORKSPACE/PROJECT/EXPERIMENT", "WORKSPACE/PROJECT", or "WORKSPACE" folder
* if --symlink, then it is a Comet path to workspace or workspace/project
* "WORKSPACE/panels" or "WORKSPACE/panels/PANEL-ZIP-FILENAME" to copy panels

where DESTINATION is:

* WORKSPACE
* WORKSPACE/PROJECT

Not all combinations are possible:


| Destination:       | WORKSPACE            | WORKSPACE/PROJECT      |
| Source (below)     |                      |                        |
|--------------------|----------------------|------------------------|
| WORKSPACE          | Copies all projects  | N/A                    |
| WORKSPACE/PROJ     | N/A                  | Copies all experiments |
| WORKSPACE/PROJ/EXP | N/A                  | Copies experiment      |

Asset types:

* 3d-image
* 3d-points - deprecated
* audio
* confusion-matrix - may contain assets
* curve
* dataframe
* dataframe-profile
* datagrid
* embeddings - may reference image asset
* histogram2d - not used
* histogram3d - internal only, single histogram, partial logging
* histogram_combined_3d
* image
* llm_data
* model-element
* notebook
* source_code
* tensorflow-model-graph-text - not used
* text-sample
* video

"""

import argparse
import glob
import io
import json
import os
import queue
import re

# Progress UI imports
import signal
import sys
import tempfile
import threading
import time
import urllib.parse
import zipfile
from datetime import datetime, timedelta

from comet_ml import APIExperiment, Artifact, Experiment, OfflineExperiment
from comet_ml._typing import TemporaryFilePath
from comet_ml.file_uploader import GitPatchUploadProcessor
from comet_ml.messages import (
    GitMetadataMessage,
    HtmlMessage,
    InstalledPackagesMessage,
    MetricMessage,
    StandardOutputMessage,
    SystemDetailsMessage,
)
from comet_ml.offline_utils import write_experiment_meta_file
from comet_ml.utils import compress_git_patch
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from ..api import API
from ..utils import remove_extra_slashes
from .copy_utils import upload_single_offline_experiment

ADDITIONAL_ARGS = False

# Global variable to track current CopyManager instance for signal handling
_current_copy_manager = None


def _signal_handler(signum, frame):
    """Signal handler to reset cursor on interruption"""
    global _current_copy_manager
    if _current_copy_manager and hasattr(_current_copy_manager, "progress_ui"):
        try:
            _current_copy_manager.progress_ui.reset_cursor()
        except Exception:
            pass  # Ignore errors during cleanup
    # Re-raise KeyboardInterrupt to allow normal handling
    raise KeyboardInterrupt()


class OfflineExperiment(OfflineExperiment):
    """
    Wrapper to alter start/stop times
    """

    START_TIME = None
    STOP_TIME = None

    def _write_experiment_meta_file(self):
        write_experiment_meta_file(
            tempdir=self.tmpdir,
            experiment_key=self.id,
            workspace=self.workspace,
            project_name=self.project_name,
            start_time=self.START_TIME or self.start_time,
            stop_time=self.STOP_TIME or self.stop_time,
            tags=self.get_tags(),
            resume_strategy=self.resume_strategy,
            customer_error_reported=self.customer_error_reported,
            customer_error_message=self.customer_error_message,
            user_provided_experiment_key=self.user_provided_experiment_key,
            comet_start_sourced=self.comet_start_sourced,
        )


def get_parser_arguments(parser):
    parser.add_argument(
        "COMET_SOURCE",
        help=(
            "The folder containing the experiments to copy: 'workspace', or 'workspace/project' or 'workspace/project/experiment'"
        ),
        type=str,
    )
    parser.add_argument(
        "COMET_DESTINATION",
        help=("The Comet destination: 'WORKSPACE', 'WORKSPACE/PROJECT'"),
        type=str,
    )
    parser.add_argument(
        "-i",
        "--ignore",
        help="Resource(s) (or 'experiments') to ignore.",
        nargs="+",
        default=[],
    )
    parser.add_argument(
        "--debug",
        help="Provide debug info",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--symlink",
        help="Instead of copying, create a link to an experiment in a project",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--sync",
        help="Check to see if experiment name has been created first; if so, skip",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-j",
        "--parallel",
        help="The number of threads to use for parallel uploading; default (None) is based on CPUs",
        type=int,
        default=None,
    )


def copy(parsed_args, remaining=None):
    # Called via `cometx copy ...`
    try:
        copy_manager = CopyManager(
            max_concurrent_uploads=parsed_args.parallel, debug=parsed_args.debug
        )
        copy_manager.copy(
            parsed_args.COMET_SOURCE,
            parsed_args.COMET_DESTINATION,
            parsed_args.symlink,
            parsed_args.ignore,
            parsed_args.debug,
            parsed_args.sync,
        )

        # Wait for all uploads to complete
        copy_manager.wait_for_uploads()

        # Shutdown upload workers
        copy_manager.shutdown_upload_workers()

        # Cleanup resources
        copy_manager.cleanup()

        if parsed_args.debug:
            print("finishing...")

    except KeyboardInterrupt:
        if parsed_args.debug:
            raise
        else:
            # Reset cursor and stop any live displays
            try:
                if hasattr(copy_manager, "progress_ui"):
                    copy_manager.progress_ui.reset_cursor()
            except Exception:
                pass  # Ignore errors during cleanup
            print("Canceled by CONTROL+C")
    except Exception as exc:
        if parsed_args.debug:
            raise
        else:
            print("ERROR: " + str(exc))
    finally:
        # Always cleanup resources
        if "copy_manager" in locals():
            copy_manager.cleanup()


def get_query_dict(url):
    """
    Given a URL, return query items as key/value dict.
    """
    result = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(result.query)
    return {key: values[0] for key, values in query.items()}


class ProgressUI:
    """Progress UI for displaying upload progress with multiple concurrent uploads using rich"""

    def __init__(self, quiet=False, max_workers=None):
        self.quiet = quiet
        self.console = Console()
        self.lock = threading.Lock()
        self.upload_status = {}  # experiment_id -> status info
        self.worker_status = {}  # worker_id -> current task info
        self.worker_mapping = {}  # actual_worker_id -> sequential_number
        self.next_worker_number = 1
        self.max_workers = max_workers
        self.start_time = None
        self.live = None

    def start(self):
        """Start the progress display"""
        if self.quiet:
            return
        self.start_time = datetime.now()
        self.live = Live("", console=self.console, refresh_per_second=4)
        self.live.start()
        self.console.print("Starting upload process...")

    def update_upload_status(
        self,
        experiment_id,
        status,
        url=None,
        error=None,
        progress=None,
        name=None,
        worker_id=None,
    ):
        """Update the status of an upload"""
        with self.lock:
            # Map worker_id to sequential number if not already mapped
            sequential_worker_id = None
            if worker_id is not None:
                if worker_id not in self.worker_mapping:
                    self.worker_mapping[worker_id] = self.next_worker_number
                    self.next_worker_number += 1
                sequential_worker_id = self.worker_mapping[worker_id]

            self.upload_status[experiment_id] = {
                "status": status,
                "url": url,
                "error": error,
                "progress": progress,
                "name": name,
                "worker_id": sequential_worker_id,
                "last_update": datetime.now(),
            }

            # Update worker status using sequential worker ID
            if sequential_worker_id is not None:
                if status == "uploading":
                    self.worker_status[sequential_worker_id] = {
                        "experiment_id": experiment_id,
                        "name": name,
                        "progress": progress,
                        "status": status,
                    }
                elif status in ["completed", "failed"]:
                    # Clear worker status when task is done
                    if sequential_worker_id in self.worker_status:
                        del self.worker_status[sequential_worker_id]
        self._redraw()

    def _redraw(self):
        """Redraw the progress display"""
        if self.quiet or not self.live:
            return

        # Calculate statistics
        total = len(self.upload_status)
        queued = sum(
            1 for info in self.upload_status.values() if info["status"] == "queued"
        )
        uploading = sum(
            1 for info in self.upload_status.values() if info["status"] == "uploading"
        )
        completed = sum(
            1 for info in self.upload_status.values() if info["status"] == "completed"
        )
        failed = sum(
            1 for info in self.upload_status.values() if info["status"] == "failed"
        )

        # Create header panel
        elapsed = datetime.now() - self.start_time if self.start_time else timedelta(0)
        header_text = f"Upload Progress - Elapsed: {elapsed.total_seconds():.1f}s"
        stats_text = f"Total: {total} | Queued: {queued} | Uploading: {uploading} | Completed: {completed} | Failed: {failed}"

        header_panel = Panel(
            f"{header_text}\n{stats_text}",
            title="[bold blue]Upload Status[/bold blue]",
            border_style="blue",
        )

        # Create table for worker status (showing what each process is working on)
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Worker", style="cyan", width=8)
        table.add_column("Status", style="cyan", width=8)
        table.add_column("Experiment Name", style="green", min_width=20, max_width=None)
        table.add_column("Progress Bar", style="yellow", width=25)
        table.add_column("Percentage", style="yellow", width=10)

        # Show worker status (what each process is working on)
        # Only show workers that are currently active or have been used
        active_workers = set(self.worker_status.keys())

        # If no workers are active yet, show a placeholder
        if not active_workers and not self.max_workers:
            table.add_row("N/A", "❓", "No worker info available", "", "")
        elif not active_workers:
            # Show that workers are waiting for tasks
            table.add_row("1", "⏸️", "Waiting for tasks...", "", "")
        else:
            # Show only active workers
            for worker_id in sorted(active_workers):
                worker_info = self.worker_status[worker_id]
                status_icon = "🔄"  # uploading
                display_name = worker_info.get(
                    "name", worker_info.get("experiment_id", "Unknown")
                )
                progress = worker_info.get("progress", 0)
                progress_bar = self._create_progress_bar(progress, 20)
                percentage_text = f"{progress:.1f}%" if progress is not None else "0%"

                table.add_row(
                    f"{worker_id}",
                    status_icon,
                    display_name,
                    progress_bar,
                    percentage_text,
                )

        # Create layout using rich Group
        from rich.console import Group

        layout = Group(header_panel, table)

        # Update the live display
        self.live.update(layout)

    def finish(self):
        """Finish the progress display and show final results"""
        if self.quiet:
            return

        if self.live:
            self.live.stop()
            # Ensure cursor is visible after stopping live display
            self.console.show_cursor()

        self.console.print("\n[bold green]Upload Process Complete![/bold green]")
        self.console.print("=" * 80)

        # Show final results
        for exp_id, info in self.upload_status.items():
            display_name = info.get("name", exp_id)
            if info["status"] == "completed":
                if info["url"]:
                    self.console.print(
                        f"✅ [link={info['url']}]{display_name}[/link] -> {info['url']}"
                    )
                else:
                    self.console.print(
                        f"⚠️  {display_name} -> Upload completed but no URL returned"
                    )
            else:
                self.console.print(f"❌ {display_name} -> {info['error']}")

        # Summary
        total = len(self.upload_status)
        completed = sum(
            1 for info in self.upload_status.values() if info["status"] == "completed"
        )
        failed = sum(
            1 for info in self.upload_status.values() if info["status"] == "failed"
        )

        success_rate = (
            (completed / (completed + failed)) * 100 if (completed + failed) > 0 else 0
        )

        self.console.print(
            f"\n[bold]Summary:[/bold] {completed}/{total} successful, {failed} failed"
        )
        self.console.print(f"[bold]Success Rate:[/bold] {success_rate:.1f}%")

    def reset_cursor(self):
        """Reset cursor and stop live display - used for cleanup on interruption"""
        if self.quiet:
            return

        try:
            if self.live:
                self.live.stop()
            self.console.show_cursor()
        except Exception:
            pass  # Ignore errors during cleanup

    def _create_progress_bar(self, progress, width=20):
        """Create a text-based progress bar"""
        filled = int(width * progress / 100)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"


class CopyManager:
    def __init__(self, max_concurrent_uploads=None, debug=False):
        """
        | Destination:       | WORKSPACE            | WORKSPACE/PROJECT      |
        | Source (below)     |                      |                        |
        |--------------------|----------------------|------------------------|
        | WORKSPACE          | Copies all projects  | N/A                    |
        | WORKSPACE/PROJ     | N/A                  | Copies all experiments |
        | WORKSPACE/PROJ/EXP | N/A                  | Copies experiment      |
        """
        global _current_copy_manager
        _current_copy_manager = self

        self.api = API()
        self.debug = debug
        # Calculate default number of workers based on CPU count, similar to download command
        self.max_concurrent_uploads = (
            min(32, os.cpu_count() + 4)
            if max_concurrent_uploads is None
            else max_concurrent_uploads
        )
        self.upload_queue = queue.Queue()
        self.upload_results = {}
        self.upload_lock = threading.Lock()
        self.upload_threads = []
        self.progress_ui = ProgressUI(
            quiet=debug, max_workers=self.max_concurrent_uploads
        )  # Show UI by default, hide when debug=True

        # Register signal handler for cursor reset on interruption
        if not debug:  # Only register signal handler when not in debug mode
            signal.signal(signal.SIGINT, _signal_handler)

        self._start_upload_workers()

    def _start_upload_workers(self):
        """Start background worker threads for handling uploads"""
        for i in range(self.max_concurrent_uploads):
            thread = threading.Thread(
                target=self._upload_worker, args=(i,), daemon=True
            )
            thread.start()
            self.upload_threads.append(thread)

    def _upload_worker(self, worker_id):
        """Worker thread that processes upload tasks from the queue"""
        while True:
            try:
                task = self.upload_queue.get(timeout=1)
                if task is None:  # Shutdown signal
                    break

                experiment_id, name, archive_path, settings = task

                # Get file size for debug logging
                try:
                    file_size = os.path.getsize(archive_path)
                    if self.debug:
                        size_mb = file_size / (1024 * 1024)
                        print(
                            f"Worker {worker_id} uploading {experiment_id}: {size_mb:.1f} MB"
                        )
                except (OSError, TypeError):
                    if self.debug:
                        print(
                            f"Worker {worker_id} uploading {experiment_id}: size unknown"
                        )

                # Update status to uploading
                self.progress_ui.update_upload_status(
                    experiment_id,
                    "uploading",
                    progress=0,
                    name=name,
                    worker_id=worker_id,
                )

                try:
                    # Start progress updates in a separate thread to avoid blocking the upload
                    progress_stop_event = threading.Event()
                    progress_thread = threading.Thread(
                        target=self._update_progress_periodically,
                        args=(experiment_id, name, worker_id, progress_stop_event),
                        daemon=True,
                    )
                    progress_thread.start()

                    # Perform the actual upload (this is the real work)
                    url = upload_single_offline_experiment(
                        offline_archive_path=archive_path,
                        settings=settings,
                        force_upload=False,
                    )

                    # Stop progress updates and wait for thread to finish
                    progress_stop_event.set()
                    progress_thread.join(
                        timeout=1.0
                    )  # Wait max 1 second for thread to finish

                    # Update status to completed
                    self.progress_ui.update_upload_status(
                        experiment_id,
                        "completed",
                        url=url,
                        progress=100,
                        name=name,
                        worker_id=worker_id,
                    )

                    with self.upload_lock:
                        self.upload_results[experiment_id] = {
                            "url": url,
                            "status": "completed",
                            "error": None,
                            "name": name,
                        }
                except Exception as e:
                    # Update status to failed
                    self.progress_ui.update_upload_status(
                        experiment_id,
                        "failed",
                        error=str(e),
                        name=name,
                        worker_id=worker_id,
                    )

                    with self.upload_lock:
                        self.upload_results[experiment_id] = {
                            "url": None,
                            "status": "failed",
                            "error": str(e),
                        }
                finally:
                    self.upload_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                if self.debug:
                    print(f"Upload worker {worker_id} error: {e}")
                continue

    def _update_progress_periodically(self, experiment_id, name, worker_id, stop_event):
        """Update progress periodically without blocking the main upload thread"""
        start_time = time.time()

        while not stop_event.is_set():
            # Update progress every 0.5 seconds
            if stop_event.wait(timeout=0.5):
                break

            current_time = time.time()
            elapsed_time = current_time - start_time

            # Calculate a more realistic progress based on time elapsed
            # This assumes uploads typically take 5-30 seconds depending on size
            # We'll show progress that accelerates over time to be more realistic
            if elapsed_time < 2.0:
                # First 2 seconds: 0-20%
                progress = min(20, (elapsed_time / 2.0) * 20)
            elif elapsed_time < 8.0:
                # 2-8 seconds: 20-70%
                progress = 20 + ((elapsed_time - 2.0) / 6.0) * 50
            elif elapsed_time < 15.0:
                # 8-15 seconds: 70-90%
                progress = 70 + ((elapsed_time - 8.0) / 7.0) * 20
            else:
                # After 15 seconds: 90-95% (slow crawl to completion)
                progress = min(95, 90 + ((elapsed_time - 15.0) / 10.0) * 5)

            # Only update if progress has increased significantly (avoid too many updates)
            current_progress = self._get_current_progress(experiment_id)
            if progress - current_progress >= 3.0:  # Update every ~3% progress
                self.progress_ui.update_upload_status(
                    experiment_id,
                    "uploading",
                    progress=progress,
                    name=name,
                    worker_id=worker_id,
                )

    def _get_current_progress(self, experiment_id):
        """Get the current progress for an experiment"""
        with self.progress_ui.lock:
            if experiment_id in self.progress_ui.upload_status:
                return self.progress_ui.upload_status[experiment_id].get("progress", 0)
        return 0

    def _queue_upload(self, experiment_id, name, archive_path, settings):
        """Queue an upload task for background processing"""
        with self.upload_lock:
            self.upload_results[experiment_id] = {
                "url": None,
                "status": "queued",
                "error": None,
                "name": name,
            }
        # Update progress UI to show queued status (no worker_id yet since it's queued)
        self.progress_ui.update_upload_status(experiment_id, "queued", name=name)
        self.upload_queue.put((experiment_id, name, archive_path, settings))

    def wait_for_uploads(self):
        """Wait for all queued uploads to complete"""
        total_uploads = len(self.upload_results)

        if total_uploads > 0:
            # Start the progress UI
            self.progress_ui.start()

            # Wait for all tasks to finish
            self.upload_queue.join()

            # Finish the progress UI
            self.progress_ui.finish()

    def shutdown_upload_workers(self):
        """Shutdown upload worker threads"""
        for _ in self.upload_threads:
            self.upload_queue.put(None)  # Shutdown signal
        for thread in self.upload_threads:
            thread.join(timeout=5)

    def cleanup(self):
        """Cleanup resources and reset signal handler"""
        global _current_copy_manager
        if _current_copy_manager == self:
            _current_copy_manager = None
        # Restore default signal handler
        signal.signal(signal.SIGINT, signal.default_int_handler)

    def copy(self, source, destination, symlink, ignore, debug, sync):
        """ """
        self.ignore = ignore
        self.debug = debug
        self.sync = sync
        self.copied_reports = False
        comet_destination = remove_extra_slashes(destination)
        comet_destination = comet_destination.replace("\\", "/").split("/")
        if len(comet_destination) == 2:
            workspace_dst, project_dst = comet_destination
        elif len(comet_destination) == 1:
            workspace_dst = comet_destination[0]
            project_dst = None
        else:
            raise Exception("invalid COMET_DESTINATION: %r" % destination)

        comet_source = remove_extra_slashes(source)
        comet_source = comet_source.replace("\\", "/").split("/")

        if len(comet_source) == 3:
            workspace_src, project_src, experiment_src = comet_source
        elif len(comet_source) == 2:
            workspace_src, project_src = comet_source
            experiment_src = "*"
        elif len(comet_source) == 1:
            workspace_src = comet_source[0]
            project_src, experiment_src = "*", "*"
        else:
            raise Exception("invalid COMET_SOURCE: %r" % source)

        # First check to make sure workspace_dst exists:
        workspaces = self.api.get_workspaces()
        if workspace_dst not in workspaces:
            raise Exception(
                f"{workspace_dst} does not exist; use the Comet UI to create it"
            )

        if project_src == "panels":
            # experiment_src may be "*" or filename
            for filename in glob.glob(
                os.path.join(workspace_src, project_src, experiment_src)
            ):
                print("Uploading panel zip: %r to %r..." % (filename, workspace_dst))
                self.api.upload_panel_zip(workspace_dst, filename)
            return

        # For checking if the project_dst exists below:
        projects = self.api.get_projects(workspace_dst)
        for experiment_folder in self.get_experiment_folders(
            workspace_src, project_src, experiment_src
        ):
            # Normalize path separators for cross-platform compatibility
            normalized_path = experiment_folder.replace("\\", "/")
            if normalized_path.count("/") >= 2:
                folder_workspace, folder_project, folder_experiment = (
                    normalized_path.rsplit("/", 2)
                )
            else:
                print("Unknown folder: %r; ignoring" % experiment_folder)
                continue
            if folder_experiment in ["project_metadata.json"]:
                continue
            temp_project_dst = project_dst
            if temp_project_dst is None:
                temp_project_dst = folder_project

            # Next, check if the project_dst exists:
            if temp_project_dst not in projects:
                project_metadata_path = os.path.join(
                    workspace_src, project_src, "project_metadata.json"
                )
                if os.path.exists(project_metadata_path):
                    with open(project_metadata_path) as fp:
                        project_metadata = json.load(fp)
                    self.api.create_project(
                        workspace_dst,
                        temp_project_dst,
                        project_description=project_metadata["projectDescription"],
                        public=project_metadata["public"],
                    )
                projects.append(temp_project_dst)

            if symlink:
                print(
                    f"Creating symlink from {workspace_src}/{project_src}/{experiment_src} to {workspace_dst}/{temp_project_dst}"
                )
                experiment = APIExperiment(previous_experiment=experiment_src)
                experiment.create_symlink(temp_project_dst)
                symlink_url = f"{self.api._get_url_server()}/{workspace_dst}/{temp_project_dst}/{experiment_src}"
                print(
                    f"    New symlink created: [link={symlink_url}]{symlink_url}[/link]"
                )
            elif "experiments" not in self.ignore:
                self.copy_experiment_to(
                    experiment_folder, workspace_dst, temp_project_dst
                )

    def create_experiment(self, workspace_dst, project_dst, offline=True):
        """
        Create an experiment in destination workspace
        and project, and return an Experiment.
        """
        if self.debug:
            print("Creating experiment...")

        ExperimentClass = OfflineExperiment if offline else Experiment
        experiment = ExperimentClass(
            project_name=project_dst,
            workspace=workspace_dst,
            log_code=False,
            log_graph=False,
            auto_param_logging=False,
            auto_metric_logging=False,
            parse_args=False,
            auto_output_logging="simple",
            log_env_details=False,
            log_git_metadata=False,
            log_git_patch=False,
            disabled=False,
            log_env_gpu=False,
            log_env_host=False,
            display_summary=None,
            log_env_cpu=False,
            log_env_network=False,
            display_summary_level=1,
            optimizer_data=None,
            auto_weight_logging=None,
            auto_log_co2=False,
            auto_metric_step_rate=10,
            auto_histogram_tensorboard_logging=False,
            auto_histogram_epoch_rate=1,
            auto_histogram_weight_logging=False,
            auto_histogram_gradient_logging=False,
            auto_histogram_activation_logging=False,
            experiment_key=None,
        )

        def filter_messages(method):
            def filtered_method(message):
                if hasattr(message, "context") and message.context == "ignore":
                    return
                method(message)

            return filtered_method

        experiment.streamer.put_message_in_q = filter_messages(
            experiment.streamer.put_message_in_q
        )
        return experiment

    def get_experiment_folders(self, workspace_src, project_src, experiment_src):
        for path in glob.iglob(f"{workspace_src}/{project_src}/{experiment_src}"):
            if any(
                [path.endswith("~"), path.endswith(".json"), path.endswith(".jsonl")]
            ):
                continue
            else:
                yield path

    def copy_experiment_to(self, experiment_folder, workspace_dst, project_dst):
        title = experiment_folder
        experiment_name = None
        # See if there is a name:
        filename = os.path.join(experiment_folder, "others.jsonl")
        if os.path.isfile(filename):
            with open(filename) as fp:
                line = fp.readline()
                while line:
                    others_json = json.loads(line)
                    if others_json["name"] == "Name":
                        experiment_name = others_json["valueCurrent"]
                        title = (
                            f"{experiment_folder} (\"{others_json['valueCurrent']}\")"
                        )
                        break
                    line = fp.readline()
        if self.debug:
            print(f"Copying from {title} to {workspace_dst}/{project_dst}...")

        # Copy other project-level items to an experiment:
        if "reports" not in self.ignore and not self.copied_reports:
            experiment = None
            workspace_src, project_src, _ = experiment_folder.replace("\\", "/").split(
                "/"
            )
            reports = os.path.join(workspace_src, project_src, "reports", "*")
            for filename in glob.glob(reports):
                if filename.endswith("reports_metadata.jsonl"):
                    continue
                basename = os.path.basename(filename)
                artifact = Artifact(basename, "Report")
                artifact.add(filename)
                if experiment is None:
                    experiment = self.create_experiment(
                        workspace_dst, project_dst, offline=False
                    )
                    experiment.log_other("Name", "Reports")
                experiment.log_artifact(artifact)
            if experiment:
                experiment.end()
            self.copied_reports = True

        if self.sync:
            if experiment_name is not None:
                experiment = self.api.get_experiment(
                    workspace_dst, project_dst, experiment_name
                )
                if experiment is not None:
                    if self.debug:
                        print("    Experiment exists; skipping due to --sync")
                    return
                else:
                    if self.debug:
                        print("   Experiment doesn't exist on destination; copying...")
            else:
                if self.debug:
                    print("    Can't sync because source has no name; copying...")

        experiment = self.create_experiment(workspace_dst, project_dst)
        # copy experiment_folder stuff to experiment
        # copy all resources to existing or new experiment
        self.log_all(experiment, experiment_folder)
        experiment.end()

        archive_path = os.path.join(
            experiment.offline_directory,
            experiment._get_offline_archive_file_name(),
        )
        if self.debug:
            try:
                file_size = os.path.getsize(archive_path)
                size_mb = file_size / (1024 * 1024)
                print(f"Queuing upload for {archive_path} ({size_mb:.1f} MB)")
            except (OSError, TypeError):
                print(f"Queuing upload for {archive_path} (size unknown)")
        self._queue_upload(
            experiment.id,
            f"{workspace_dst}/{project_dst}/{experiment_name or experiment.id}",
            archive_path,
            self.api.config,
        )

    def log_metadata(self, experiment, filename):
        if self.debug:
            with experiment.context_manager("ignore"):
                print("log_metadata...")
        if os.path.exists(filename):
            metadata = json.load(open(filename))
            experiment.add_tags(metadata.get("tags", []))
            if metadata.get("fileName", None):
                experiment.set_filename(metadata["fileName"])

            OfflineExperiment.START_TIME = metadata.get("startTimeMillis")
            OfflineExperiment.STOP_TIME = metadata.get("endTimeMillis")

    def log_system_details(self, experiment, filename):
        if self.debug:
            with experiment.context_manager("ignore"):
                print("log_system_details...")
        if os.path.exists(filename):
            system = json.load(open(filename))

            # System info:
            message = SystemDetailsMessage(
                command=system.get("command") or [],
                env=system.get("env") or {},
                hostname=system.get("hostname") or "",
                ip=system.get("ip") or "",
                machine=system.get("machine") or "",
                os_release=system.get("osRelease") or "",
                os_type=system.get("osType") or "",
                os=system.get("os") or "",
                pid=system.get("pid", None) or 0,
                processor=system.get("processor") or "",
                python_exe=system.get("executable") or "",
                python_version_verbose=system.get("pythonVersionVerbose") or "",
                python_version=system.get("pythonVersion") or "",
                user=system.get("user") or "",
            )
            experiment._enqueue_message(message)

    def log_graph(self, experiment, filename):
        if self.debug:
            with experiment.context_manager("ignore"):
                print("log_graph...")
        if os.path.exists(filename):
            experiment.set_model_graph(open(filename).read())

    def _log_asset_filename(
        self, experiment, asset_type, metadata, filename, step, log_filename
    ):
        if isinstance(filename, io.BytesIO):
            binary_io = filename
        else:
            binary_io = open(filename, "rb")

        # If the filename has a sequence number:
        sequence = re.search(r"\((\d+)\)$", log_filename)
        if sequence:
            log_filename = log_filename.rsplit("(", 1)[0].strip()

        result = experiment._log_asset(
            binary_io,
            file_name=log_filename,
            copy_to_tmp=True,
            asset_type=asset_type,
            metadata=metadata,
            step=step,
        )
        return result

    def update_datagrid_contents(
        self,
        experiment,
        asset_map,
        metadata,
        step,
        zip_file_path,
        log_as_filename,
        old_asset_id,
    ):
        """
        Take a datagrid zip, and replace the asset IDs with new
        asset IDs.
        """
        basename = os.path.basename(zip_file_path)
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_to_edit = None
            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                for file in zip_ref.namelist():
                    file_to_edit = file
                zip_ref.extractall(tmp_dir)

            file_path = os.path.join(tmp_dir, file_to_edit)
            print(file_path)
            with open(file_path, "r") as fp:
                json_data = json.load(fp)
                asset_columns = []
                for i, column in enumerate(json_data["columns"]):
                    if json_data["columns"][column] == "IMAGE-ASSET":
                        asset_columns.append(i)
                for row in json_data["rows"]:
                    for column in asset_columns:
                        old = row[column]["asset_id"]
                        row[column]["asset_id"] = asset_map[old]

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, basename)
            with open(file_path, "wb") as fp:
                with zipfile.ZipFile(fp, "w", zipfile.ZIP_DEFLATED) as zfp:
                    zfp.writestr(file_to_edit, json.dumps(json_data))
            result = self._log_asset_filename(
                experiment,
                "datagrid",
                metadata,
                file_path,
                step,
                log_as_filename,
            )
            asset_map[old_asset_id] = result["assetId"]

    def _log_asset(
        self, experiment, path, asset_type, log_filename, assets_metadata, asset_map
    ):
        log_as_filename = assets_metadata[log_filename].get(
            "logAsFileName",
            None,
        )
        step = assets_metadata[log_filename].get("step")
        epoch = assets_metadata[log_filename].get("epoch")
        old_asset_id = assets_metadata[log_filename].get("assetId")
        if asset_type in self.ignore:
            return
        if log_filename.startswith("/"):
            filename = os.path.join(path, asset_type, log_filename[1:])
        else:
            filename = os.path.join(path, asset_type, log_filename)

        filename = filename.replace(":", "-")

        if not os.path.isfile(filename):
            with experiment.context_manager("ignore"):
                print("Missing file %r: unable to copy" % filename)
            return

        metadata = assets_metadata[log_filename].get("metadata")
        metadata = json.loads(metadata) if metadata else {}

        if asset_type == "notebook":
            result = experiment.log_notebook(filename)  # done!
            asset_map[old_asset_id] = result["assetId"]
        elif asset_type == "embeddings":
            # This will come after contained assets
            with open(filename) as fp:
                em_json = json.load(fp)
            # go though JSON, replace asset_ids with new asset_ids
            # {"embeddings":
            #    [{"tensorName": "Comet Embedding",
            #      "tensorShape": [240, 5],
            #      "tensorPath": "/api/asset/download?assetId=b6edbf11e548417580af163b20d7fd23&experimentKey=fe5ed0231e4e4425a13b7c25ea82c51f",
            #      "metadataPath": "/api/asset/download?assetId=fcac2559f7cc42f8a14d20ebed4f8da1&experimentKey=fe5ed0231e4e4425a13b7c25ea82c51f",
            #      "sprite": {
            #         "imagePath": "/api/image/download?imageId=2052efea88b24d4b9111e0e4b0bdb003&experimentKey=fe5ed0231e4e4425a13b7c25ea82c51f",
            #         "singleImageDim": [6, 6]
            #      }
            #     }]
            # }
            for embedding in em_json["embeddings"]:
                if embedding.get("tensorPath"):
                    args = get_query_dict(embedding["tensorPath"])
                    new_args = {
                        "experimentKey": experiment.id,
                        "assetId": asset_map[args.get("assetId", args.get("imageId"))],
                    }
                    embedding["tensorPath"] = (
                        "/api/asset/download?assetId={assetId}&experimentKey={experimentKey}".format(
                            **new_args
                        )
                    )
                if embedding.get("metadataPath"):
                    args = get_query_dict(embedding["metadataPath"])
                    new_args = {
                        "experimentKey": experiment.id,
                        "assetId": asset_map[args.get("assetId", args.get("imageId"))],
                    }
                    embedding["metadataPath"] = (
                        "/api/asset/download?assetId={assetId}&experimentKey={experimentKey}".format(
                            **new_args
                        )
                    )
                if embedding.get("sprite"):
                    if embedding["sprite"].get("imagePath"):
                        args = get_query_dict(embedding["sprite"]["imagePath"])
                        new_args = {
                            "experimentKey": experiment.id,
                            "assetId": asset_map[
                                args.get("assetId", args.get("imageId"))
                            ],
                        }
                        embedding["sprite"]["imagePath"] = (
                            "/api/asset/download?assetId={assetId}&experimentKey={experimentKey}".format(
                                **new_args
                            )
                        )
            binary_io = io.BytesIO(json.dumps(em_json).encode())
            result = self._log_asset_filename(
                experiment,
                asset_type,
                metadata,
                binary_io,
                step,
                log_as_filename or log_filename,
            )
            asset_map[old_asset_id] = result["assetId"]
        elif asset_type == "datagrid":
            self.update_datagrid_contents(
                experiment,
                asset_map,
                metadata,
                step,
                filename,
                log_as_filename or log_filename,
                old_asset_id,
            )
        elif asset_type == "confusion-matrix":
            # This will come after contained assets
            with open(filename) as fp:
                cm_json = json.load(fp)
            # go though JSON, replace asset_ids with new asset_ids
            for row in cm_json["sampleMatrix"]:
                if row:
                    for cols in row:
                        if cols:
                            for cell in cols:
                                if cell and isinstance(cell, dict):
                                    old_cell_asset_id = cell["assetId"]
                                    new_cell_asset_id = asset_map[old_cell_asset_id]
                                    cell["assetId"] = new_cell_asset_id

            binary_io = io.BytesIO(json.dumps(cm_json).encode())
            result = self._log_asset_filename(
                experiment,
                asset_type,
                metadata,
                binary_io,
                step,
                log_as_filename or log_filename,
            )
            asset_map[old_asset_id] = result["assetId"]
        elif asset_type == "video":
            name = os.path.basename(filename)
            binary_io = open(filename, "rb")
            result = experiment.log_video(
                binary_io, name=log_as_filename or name, step=step, epoch=epoch
            )  # done!
            asset_map[old_asset_id] = result["assetId"]
        elif asset_type == "model-element":
            name = os.path.basename(filename)
            result = experiment.log_model(name, filename)
            asset_map[old_asset_id] = result["assetId"]
        else:
            result = self._log_asset_filename(
                experiment,
                asset_type,
                metadata,
                filename,
                step,
                log_as_filename or log_filename,
            )
            asset_map[old_asset_id] = result["assetId"]

    def log_assets(self, experiment, path, assets_metadata):
        if self.debug:
            with experiment.context_manager("ignore"):
                print("log_assets...")
        # Create mapping from old asset id to new asset id
        asset_map = {}
        # Process all of the non-nested assets first:
        for log_filename in assets_metadata:
            asset_type = assets_metadata[log_filename].get("type", "asset") or "asset"
            if asset_type not in ["confusion-matrix", "embeddings", "datagrid"]:
                if (
                    "remote" in assets_metadata[log_filename]
                    and assets_metadata[log_filename]["remote"]
                ):
                    asset = assets_metadata[log_filename]
                    experiment.log_remote_asset(
                        uri=asset["link"],
                        remote_file_name=asset["fileName"],
                        step=asset["step"],
                        metadata=asset["metadata"],
                    )
                else:
                    self._log_asset(
                        experiment,
                        path,
                        asset_type,
                        log_filename,
                        assets_metadata,
                        asset_map,
                    )
        # Process all nested assets:
        for log_filename in assets_metadata:
            asset_type = assets_metadata[log_filename].get("type", "asset") or "asset"
            if asset_type in ["confusion-matrix", "embeddings", "datagrid"]:
                self._log_asset(
                    experiment,
                    path,
                    asset_type,
                    log_filename,
                    assets_metadata,
                    asset_map,
                )

    def log_code(self, experiment, filename):
        """ """
        if self.debug:
            with experiment.context_manager("ignore"):
                print("log_code...")
        if os.path.exists(filename):
            if os.path.isfile(filename):
                experiment.log_code(str(filename))
            elif os.path.isdir(filename):
                experiment.log_code(folder=str(filename))

    def log_requirements(self, experiment, filename):
        """
        Requirements (pip packages)
        """
        if self.debug:
            with experiment.context_manager("ignore"):
                print("log_requirements...")
        if os.path.exists(filename):
            installed_packages_list = [package.strip() for package in open(filename)]
            if installed_packages_list is None:
                return
            message = InstalledPackagesMessage(
                installed_packages=installed_packages_list,
            )
            experiment._enqueue_message(message)

    def log_metrics(self, experiment, filename):
        """ """
        if os.path.exists(filename):
            if self.debug:
                with experiment.context_manager("ignore"):
                    print("log_metrics %s..." % filename)

            for line in open(filename):
                dict_line = json.loads(line)
                name = dict_line["metricName"]
                if name.startswith("sys.") and "system-metrics" in self.ignore:
                    continue
                value = dict_line.get("metricValue", None)
                if value is None:
                    continue
                step = dict_line.get("step", None)
                epoch = dict_line.get("epoch", None)
                context = dict_line.get("runContext", None)
                timestamp = dict_line.get("timestamp", None)
                message = MetricMessage(
                    context=context,
                    timestamp=timestamp,
                )
                message.set_metric(name, value, step=step, epoch=epoch)
                experiment._enqueue_message(message)

    def log_metrics_split(self, experiment, folder):
        """ """
        summary_filename = os.path.join(folder, "metrics_summary.jsonl")
        if os.path.exists(summary_filename):
            if self.debug:
                with experiment.context_manager("ignore"):
                    print("log_metrics from %s..." % summary_filename)

            for line in open(summary_filename):
                metric_summary = json.loads(line)
                self.log_metrics(
                    experiment,
                    os.path.join(
                        folder, "metrics", "metric_%05d.jsonl" % metric_summary["count"]
                    ),
                )

    def _prepare_parameter_value(self, value):
        if isinstance(value, list):
            return str(value)
        else:
            return value

    def log_parameters(self, experiment, filename):
        """ """
        if self.debug:
            with experiment.context_manager("ignore"):
                print("log_parameters...")
        if os.path.exists(filename):
            parameters = json.load(open(filename))
            parameter_dictionary = {
                parameter["name"]: self._prepare_parameter_value(
                    parameter["valueCurrent"]
                )
                for parameter in parameters
            }
            experiment.log_parameters(parameter_dictionary, nested_support=True)

    def log_others(self, experiment, filename):
        """ """
        if self.debug:
            with experiment.context_manager("ignore"):
                print("log_others...")
        if os.path.exists(filename):
            for line in open(filename):
                dict_line = json.loads(line)
                name = dict_line["name"]
                value = dict_line["valueCurrent"]
                experiment.log_other(key=name, value=value)

    def log_output(self, experiment, output_file):
        """ """
        if self.debug:
            with experiment.context_manager("ignore"):
                print("log_output...")
        if os.path.exists(output_file):
            for line in open(output_file):
                message = StandardOutputMessage(
                    output=line,
                    stderr=False,
                )
                experiment._enqueue_message(message)

    def log_html(self, experiment, filename):
        if self.debug:
            with experiment.context_manager("ignore"):
                print("log_html...")
        if os.path.exists(filename):
            html = open(filename).read()
            message = HtmlMessage(
                html=html,
            )
            experiment._enqueue_message(message)

    def log_git_metadata(self, experiment, filename):
        if os.path.exists(filename):
            with open(filename) as fp:
                metadata = json.load(fp)

            git_metadata = {
                "parent": metadata.get("parent", None),
                "repo_name": None,
                "status": None,
                "user": metadata.get("user", None),
                "root": metadata.get("root", None),
                "branch": metadata.get("branch", None),
                "origin": metadata.get("origin", None),
            }
            message = GitMetadataMessage(
                git_metadata=git_metadata,
            )
            experiment._enqueue_message(message)

    def log_git_patch(self, experiment, filename):
        if os.path.exists(filename):
            with open(filename) as fp:
                git_patch = fp.read()

            _, zip_path = compress_git_patch(git_patch)
            processor = GitPatchUploadProcessor(
                TemporaryFilePath(zip_path),
                experiment.asset_upload_limit,
                url_params=None,
                metadata=None,
                copy_to_tmp=False,
                error_message_identifier=None,
                tmp_dir=experiment.tmpdir,
                critical=False,
            )
            upload_message = processor.process()
            if upload_message:
                experiment._enqueue_message(upload_message)

    def log_all(self, experiment, experiment_folder):
        """ """
        # FIXME: missing notes (edited by human, not logged programmatically)
        if "metrics" not in self.ignore:
            # All together, in one file:
            self.log_metrics(
                experiment, os.path.join(experiment_folder, "metrics.jsonl")
            )
            # In separate files:
            self.log_metrics_split(experiment, experiment_folder)

        if "metadata" not in self.ignore:
            self.log_metadata(
                experiment, os.path.join(experiment_folder, "metadata.json")
            )

        if "parameters" not in self.ignore:
            self.log_parameters(
                experiment, os.path.join(experiment_folder, "parameters.json")
            )

        if "others" not in self.ignore:
            self.log_others(experiment, os.path.join(experiment_folder, "others.jsonl"))

        if "assets" not in self.ignore:
            assets_metadata_filename = os.path.join(
                experiment_folder, "assets", "assets_metadata.jsonl"
            )
            assets_metadata = {}
            if os.path.exists(assets_metadata_filename):
                for line in open(assets_metadata_filename):
                    data = json.loads(line)
                    assets_metadata[data["fileName"]] = data

                self.log_assets(
                    experiment,
                    os.path.join(experiment_folder, "assets"),
                    assets_metadata,
                )

        if "output" not in self.ignore:
            self.log_output(
                experiment, os.path.join(experiment_folder, "run/output.txt")
            )

        if "requirements" not in self.ignore:
            self.log_requirements(
                experiment, os.path.join(experiment_folder, "run/requirements.txt")
            )

        if "model-graph" not in self.ignore:
            self.log_graph(
                experiment, os.path.join(experiment_folder, "run/graph_definition.txt")
            )

        if "html" not in self.ignore:
            # NOTE: also logged as html asset
            html_filenames = os.path.join(experiment_folder, "assets", "html", "*")
            for html_filename in glob.glob(html_filenames):
                self.log_html(experiment, html_filename)
            # Deprecated:
            self.log_html(
                experiment,
                os.path.join(experiment_folder, "experiment.html"),
            )

        if "system-details" not in self.ignore:
            self.log_system_details(
                experiment, os.path.join(experiment_folder, "system_details.json")
            )

        if "git" not in self.ignore:
            self.log_git_metadata(
                experiment, os.path.join(experiment_folder, "run", "git_metadata.json")
            )
            self.log_git_patch(
                experiment, os.path.join(experiment_folder, "run", "git_diff.patch")
            )

        if "code" not in self.ignore:
            code_folder = os.path.join(experiment_folder, "run", "code")
            self.log_code(experiment, code_folder)
            # Deprecated:
            self.log_code(
                experiment, os.path.join(experiment_folder, "run", "script.py")
            )


def main(args):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    get_parser_arguments(parser)
    parsed_args = parser.parse_args(args)
    copy(parsed_args)


if __name__ == "__main__":
    # Called via `python -m cometx.cli.copy ...`
    main(sys.argv[1:])
