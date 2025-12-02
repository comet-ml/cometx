"""
Prerequisites:
- AWS credentials configured
"""

import argparse
import os
import queue
import random
import sys
import tempfile
import threading
import time
from datetime import datetime

import boto3
from comet_ml import API, Optimizer
from comet_ml.config import get_config, get_optimizer_address
from comet_ml.exceptions import NotFound, Unauthorized
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from .admin_optimizer_report import fetch_dashboard_data
from .copy_utils import upload_single_offline_experiment

ADDITIONAL_ARGS = False


class ProgressUI:
    """Progress UI for displaying upload progress with multiple concurrent workers using rich"""

    def __init__(self, max_workers, optimizer_id, quiet=False):
        self.max_workers = max_workers
        self.optimizer_id = optimizer_id
        self.quiet = quiet
        self.console = Console()
        self.lock = threading.Lock()
        self.worker_status = {}  # worker_id -> status info
        self.live = None

    def start(self):
        """Start the progress display"""
        if self.quiet:
            return
        self.live = Live("", console=self.console, refresh_per_second=4)
        self.live.start()

    def update_worker_status(self, worker_id, status, uploaded_id=None, url=None):
        """Update the status of a worker"""
        with self.lock:
            self.worker_status[worker_id] = {
                "status": status,
                "uploaded_id": uploaded_id,
                "url": url,
                "last_update": datetime.now(),
            }
        self._redraw()

    def _redraw(self):
        """Redraw the progress display"""
        if self.quiet or not self.live:
            return

        # Create header panel
        header_text = (
            f"Comet Optimizer Experiment Upload\n\nOptimizer ID: {self.optimizer_id}"
        )
        header_panel = Panel(
            header_text,
            title="[bold blue]Upload Status[/bold blue]",
            border_style="blue",
        )

        # Create table for worker status
        table = Table(show_header=True, header_style="bold magenta", expand=True)
        table.add_column("Worker", style="cyan", width=8, no_wrap=True)
        table.add_column("Status", style="green", min_width=50)

        # Show all workers
        for worker_id in range(1, self.max_workers + 1):
            if worker_id in self.worker_status:
                worker_info = self.worker_status[worker_id]
                status = worker_info.get("status", "waiting")
                uploaded_id = worker_info.get("uploaded_id")
                url = worker_info.get("url")

                if status == "waiting":
                    status_text = "[bold blue]waiting[/bold blue]"
                elif status == "processing":
                    if uploaded_id and url:
                        status_text = f"[bold yellow]processing {uploaded_id} to {url}...[/bold yellow]"
                    elif uploaded_id:
                        status_text = (
                            f"[bold yellow]processing {uploaded_id}...[/bold yellow]"
                        )
                    else:
                        status_text = "[bold yellow]processing...[/bold yellow]"
                elif status == "completed":
                    status_text = "[bold green]completed[/bold green]"
                else:
                    status_text = f"[bold red]{status}[/bold red]"

                table.add_row(f"Worker {worker_id}", status_text)
            else:
                # Show waiting worker
                table.add_row(f"Worker {worker_id}", "[bold blue]waiting[/bold blue]")

        # Create layout with header and table
        layout = Group(header_panel, table)

        # Update the live display
        self.live.update(layout)

    def finish(self):
        """Finish the progress display"""
        if self.quiet:
            return

        if self.live:
            self._redraw()
            time.sleep(0.5)
            self.live.stop()
            self.console.show_cursor()

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


class Experiment:
    """
    An experiment stub for status updating.
    """

    def __init__(self, pid, trial):
        self.optimizer = {
            "pid": pid,
            "trial": trial,
        }


def get_parser_arguments(parser):
    parser.add_argument(
        "optimizer_id",
        metavar="optimizer-id",
        help="The Comet Optimizer ID",
        type=str,
    )
    parser.add_argument(
        "--bucket-name",
        help="The name of S3 bucket, if experiments are in S3",
        type=str,
    )
    parser.add_argument(
        "--object-key",
        help="The S3 object key, if experiments are in S3",
        type=str,
        default="",
    )
    parser.add_argument(
        "--path",
        help="The path to offline experiments, if not using S3",
        type=str,
        default="./.cometml-runs",
    )
    parser.add_argument(
        "--debug",
        help=("Show debugging information"),
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--parallel",
        "-j",
        help=("How many parallel experiments to upload"),
        type=int,
        default=20,
    )


class Uploader:
    def __init__(self, args):
        self.optimizer_id = args.optimizer_id
        self.bucket_name = args.bucket_name
        self.object_key = args.object_key
        self.path = args.path
        self.debug = args.debug
        self.max_threads = args.parallel
        self.s3_client = None
        self.api = API(cache=False)
        self.config = get_config()
        self.api_key = self.config["comet.api_key"]
        self.optimizer_url = get_optimizer_address(self.config)
        self.optimizer = Optimizer(self.optimizer_id)

        # Thread pool setup
        self.task_queue = queue.Queue()
        self.worker_threads = []
        self.progress_ui = ProgressUI(
            max_workers=self.max_threads,
            optimizer_id=self.optimizer_id,
            quiet=self.debug,
        )
        self.fetch_thread = None
        self.stop_fetching = threading.Event()
        self._start_workers()

    def _start_workers(self):
        """Start background worker threads for processing uploads"""
        for i in range(self.max_threads):
            thread = threading.Thread(target=self._worker, args=(i + 1,), daemon=True)
            thread.start()
            self.worker_threads.append(thread)

    def _worker(self, worker_id):
        """Worker thread that processes upload tasks from the queue"""
        # Initialize worker as waiting
        self.progress_ui.update_worker_status(worker_id, "waiting")

        while True:
            try:
                task = self.task_queue.get(timeout=1)
                if task is None:  # Shutdown signal
                    break

                item = task
                experiment = Experiment(item["pid"], item["trial"])
                self.optimizer.update(experiment, "processing", {})
                experiment_key = item["metadata"]["experiment_id"]

                try:
                    # Update status to processing (before download)
                    self.progress_ui.update_worker_status(worker_id, "processing")

                    # Download or get file path
                    if self.bucket_name is not None and self.object_key is not None:
                        filename = self.download_from_s3(experiment_key)
                    else:
                        filename = os.path.join(self.path, (experiment_key + ".zip"))

                    # Upload experiment
                    (url, uploaded_id) = self.upload_experiment(filename)

                    # Update status with uploaded_id and url
                    self.progress_ui.update_worker_status(
                        worker_id, "processing", uploaded_id=uploaded_id, url=url
                    )

                    if self.debug:
                        print(
                            f"Worker {worker_id}: Uploading {experiment_key} to {url}..."
                        )

                    # Give a changes to get started:
                    time.sleep(20)
                    # Now we check to see if done:
                    status = self.get_processing_status(uploaded_id)
                    if status is None:
                        print("Backend does not support endpoint, or invalid key")
                        time.sleep(random.randint(30, 60))
                    else:
                        while status:
                            time.sleep(10)
                            status = self.get_processing_status(uploaded_id)

                    # when processing is complete:
                    self.optimizer.update(
                        experiment, "processed", {"uploaded_id": uploaded_id}
                    )

                    # Update status to completed briefly, then back to waiting
                    self.progress_ui.update_worker_status(
                        worker_id, "completed", uploaded_id=uploaded_id, url=url
                    )
                    time.sleep(0.5)  # Brief display of completed status
                    self.progress_ui.update_worker_status(worker_id, "waiting")

                except Exception as e:
                    if self.debug:
                        print(f"Worker {worker_id} error: {e}")
                    self.progress_ui.update_worker_status(worker_id, f"error: {str(e)}")
                    time.sleep(1)  # Brief display of error
                    self.progress_ui.update_worker_status(worker_id, "waiting")
                finally:
                    self.task_queue.task_done()

            except queue.Empty:
                # Worker is waiting for tasks
                self.progress_ui.update_worker_status(worker_id, "waiting")
                continue
            except Exception as e:
                if self.debug:
                    print(f"Worker {worker_id} error: {e}")
                self.progress_ui.update_worker_status(worker_id, "waiting")
                continue

    def download_from_s3(self, experiment_id):
        """
        Download file from S3 to temporary location.

        Args:
            experiment_id: experiment key

        Returns:
            filename of downloaded file
        """
        if self.s3_client is None:
            self.s3_client = boto3.client("s3")

        s3_key = f"{self.object_key}/{experiment_id}.zip"

        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
            tmp_file_path = tmp_file.name

        self.s3_client.download_file(self.bucket_name, s3_key, tmp_file_path)

        return tmp_file_path

    def upload_experiment(self, filename):
        url = upload_single_offline_experiment(
            offline_archive_path=filename,
            settings=self.api.config,
            force_upload=True,
        )
        base_url, uploaded_key = url.rsplit("/", 1)
        return (url, uploaded_key)

    def get_processing_status(self, key):
        params = {"experimentKey": key}
        endpoint = "experiment/processing/status"
        try:
            results = self.api._client.get_from_endpoint(endpoint, params)
        except Unauthorized:
            print(f"You are not authorized to access this experiment key: {key}")
            return None
        except NotFound:
            print(f"Your Comet ML backend does not support this endpoint: {endpoint}")
            return None

        return results.get("processing")

    def _fetch_jobs_continuously(self):
        """Background thread that continuously fetches jobs and adds them to the queue"""
        page = 1
        empty_fetch_wait_time = 10  # Wait 10 seconds when we get empty results

        while not self.stop_fetching.is_set():
            try:
                # Fetch a batch of jobs
                data, error = fetch_dashboard_data(
                    self.optimizer_id,
                    self.api_key,
                    self.optimizer_url,
                    status_filter=["completed"],
                    page=page,
                    page_size=self.max_threads,
                )

                # Check if we have jobs to process
                if not data.get("jobs") or len(data["jobs"]) == 0:
                    # No jobs in this page, reset to page 1 and wait before checking again
                    # This allows new jobs to appear and be picked up
                    page = 1
                    if self.stop_fetching.wait(timeout=empty_fetch_wait_time):
                        break
                    continue

                # Queue all jobs from this batch
                for item in data["jobs"]:
                    if self.stop_fetching.is_set():
                        break
                    self.task_queue.put(item)

                # Move to next page for next fetch
                page += 1

                # Brief pause before fetching next batch
                # This allows workers to process jobs while we fetch more
                if self.stop_fetching.wait(timeout=1):
                    break

            except Exception as e:
                if self.debug:
                    print(f"Error fetching jobs: {e}")
                # Wait a bit before retrying, then reset to page 1
                page = 1
                if self.stop_fetching.wait(timeout=5):
                    break

    def upload_from_optimizer(self):
        # Start progress UI
        if not self.debug:
            self.progress_ui.start()

        # Start background thread to continuously fetch jobs
        self.fetch_thread = threading.Thread(
            target=self._fetch_jobs_continuously, daemon=True
        )
        self.fetch_thread.start()

        try:
            # Keep running until interrupted
            # The fetch thread will continuously check for new jobs
            # Workers will process jobs as they become available
            while True:
                # Check if fetch thread is still alive
                if not self.fetch_thread.is_alive():
                    # Fetch thread died (shouldn't happen, but handle it)
                    if self.debug:
                        print("Fetch thread stopped unexpectedly")
                    break

                # Wait a bit and check again
                # This allows the main thread to be interruptible
                time.sleep(1)

        except KeyboardInterrupt:
            if not self.debug:
                self.progress_ui.reset_cursor()
            print("\nCanceled by CONTROL+C")
        finally:
            # Signal fetch thread to stop
            self.stop_fetching.set()
            if self.fetch_thread and self.fetch_thread.is_alive():
                self.fetch_thread.join(timeout=5)

            # Wait for all remaining tasks in queue to complete
            self.task_queue.join()

            # Shutdown workers
            self.shutdown_workers()
            if not self.debug:
                self.progress_ui.finish()

    def shutdown_workers(self):
        """Shutdown worker threads"""
        for _ in self.worker_threads:
            self.task_queue.put(None)  # Shutdown signal
        for thread in self.worker_threads:
            thread.join(timeout=5)


def main(args):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    get_parser_arguments(parser)
    parsed_args = parser.parse_args(args)
    upload_optimizer_experiments(parsed_args)


def upload_optimizer_experiments(parsed_args):
    uploader = Uploader(parsed_args)
    uploader.upload_from_optimizer()


if __name__ == "__main__":
    # Called via `python -m cometx.cli.upload_optimizer_experiments ...`
    main(sys.argv[1:])
