"""
Charts:

1. avg(metric) vs workspace (bar)
2. Time series: legend - project (line)
   max(metric) vs month by workspace

"""

import datetime
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

from comet_ml import API
from comet_ml.query import Metadata
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

console = Console()
api = None  # Will be initialized in main()


def search(workspace, project_name, query):
    columns = api._client.get_project_columns(workspace, project_name)
    predicates = query.get_predicates_for_search(columns)

    project_json = api.get_project(workspace, project_name)
    if not project_json:
        return []

    project_id = project_json["projectId"]
    results = api._client.search(
        workspace, project_id, predicates, archived=False, page=1, page_size=1_000_000
    )
    return results.json()


def get_experiment_data(workspace, project_name, date):
    results = search(
        workspace,
        project_name,
        Metadata("start_server_timestamp") > datetime.datetime(*date),
    )
    # Return a list of dictionaries, one per experiment
    return [
        {key: data[key] for key in ["experimentKey", "server_timestamp"]}
        for data in results.get("experiments", [])
    ]


def process_workspace_project(
    workspace: str, project_name: str, date: Tuple[int, int, int]
) -> Tuple[str, str, List[Dict[str, Any]]]:
    """Process a single workspace-project pair and return the result."""
    try:
        experiment_data = get_experiment_data(workspace, project_name, date)
        return (workspace, project_name, experiment_data)
    except Exception as e:
        console.print(f"[red]Error processing {workspace}/{project_name}: {e}[/red]")
        return (workspace, project_name, [])


def collect_experiment_data(workspaces, max_workers, cutoff_date):
    """Collect experiment data from workspaces in parallel."""
    all_data = []

    # First, collect all workspace-project pairs
    console.print("[dim]Collecting project list from workspaces...[/dim]")
    workspace_project_pairs = []
    for workspace in workspaces:
        try:
            projects = api.get_projects(workspace)
            for project_name in projects:
                workspace_project_pairs.append((workspace, project_name))
        except Exception as e:
            console.print(
                f"[red]Error getting projects for workspace {workspace}: {e}[/red]"
            )

    total_pairs = len(workspace_project_pairs)
    if total_pairs == 0:
        console.print("[yellow]No projects found in any workspace.[/yellow]")
        return []

    console.print(
        f"[green]Found {total_pairs} workspace-project pairs to process[/green]"
    )

    # Process in parallel
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "[cyan]Processing workspace-project pairs in parallel...", total=total_pairs
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            date = cutoff_date
            future_to_pair = {
                executor.submit(
                    process_workspace_project, workspace, project_name, date
                ): (workspace, project_name)
                for workspace, project_name in workspace_project_pairs
            }

            # Process completed tasks as they finish
            for future in as_completed(future_to_pair):
                workspace, project_name = future_to_pair[future]
                try:
                    ws, proj, experiment_data = future.result()
                    if experiment_data:  # experiment_data is now a list
                        ws_project = {"workspace": ws, "project_name": proj}
                        experiment_data = [
                            {**data, **ws_project} for data in experiment_data
                        ]
                        all_data.extend(experiment_data)  # Use extend instead of append
                    progress.update(
                        task,
                        description=f"[cyan]Processed: [bold]{workspace}/{proj}[/bold] ({progress.tasks[task].completed}/{total_pairs})",
                    )
                except Exception as e:
                    console.print(
                        f"[red]Error processing {workspace}/{project_name}: {e}[/red]"
                    )
                finally:
                    progress.advance(task)
    return all_data


def process_metric_batch(
    chunk: List[str], batch_index: int, total_batches: int
) -> Tuple[int, Dict[str, Any]]:
    """Process a single batch of experiment keys and return the metrics."""
    try:
        metrics = api.get_metrics_for_chart(chunk, SYSTEM_METRICS_TO_TRACK)
        return (batch_index, metrics)
    except Exception as e:
        console.print(
            f"[red]Error processing batch {batch_index + 1}/{total_batches}: {e}[/red]"
        )
        return (batch_index, {})


def get_metric_data(experiment_keys, max_workers):
    """Fetch metric data for experiments in parallel batches."""
    BATCH_SIZE = 250
    chunks = [
        experiment_keys[i : i + BATCH_SIZE]
        for i in range(0, len(experiment_keys), BATCH_SIZE)
    ]

    console.print(
        f"\n[bold green]Processing {len(experiment_keys)} experiments in {len(chunks)} batches (parallel processing with {max_workers} workers)[/bold green]"
    )

    if len(chunks) == 0:
        console.print("[yellow]No experiment keys to process.[/yellow]")
        return []

    # Create a list to store results in order
    all_metrics = [None] * len(chunks)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        batch_task = progress.add_task(
            "[magenta]Fetching metrics in parallel...", total=len(chunks)
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batch tasks
            future_to_batch = {
                executor.submit(process_metric_batch, chunk, i, len(chunks)): i
                for i, chunk in enumerate(chunks)
            }

            # Process completed tasks as they finish
            for future in as_completed(future_to_batch):
                batch_index = future_to_batch[future]
                try:
                    idx, metrics = future.result()
                    # Transform list of dicts into dict keyed by experimentKey
                    metrics_list = list(metrics.values())
                    all_metrics[idx] = {
                        item["experimentKey"]: {
                            k: v for k, v in item.items() if k != "experimentKey"
                        }
                        for item in metrics_list
                    }
                    progress.update(
                        batch_task,
                        description=f"[magenta]Batch {idx + 1}/{len(chunks)} completed ({progress.tasks[batch_task].completed}/{len(chunks)})",
                    )
                except Exception as e:
                    console.print(
                        f"[red]Error processing batch {batch_index + 1}: {e}[/red]"
                    )
                    all_metrics[batch_index] = {}
                finally:
                    progress.advance(batch_task)

    # Merge all batch dictionaries into a single dictionary
    merged_metrics = {}
    for batch_metrics in all_metrics:
        if batch_metrics:  # Skip None or empty dictionaries
            merged_metrics.update(batch_metrics)

    return merged_metrics


WORKSPACES = [
    "chasefortier",
    "comet-demos",
    "examples",
    "fetch-issue-test",
    "stevenm-workspace",
]
MAX_WORKERS = min(
    32, os.cpu_count() + 4
)  # Number of parallel threads for processing workspace-project pairs and metric batches
SYSTEM_METRICS_TO_TRACK = [
    "sys.gpu.0.gpu_utilization",  # percent
    "sys.gpu.0.memory_utilization",  # percent
    "sys.gpu.0.used_memory",  # gb
    "sys.gpu.0.power_usage",  # watt
    "sys.gpu.0.temperature",  # celsius
]


def main():
    global api
    console.print(
        Panel.fit(
            "[bold blue]GPU Usage Data Collection[/bold blue]\n"
            f"Workspaces: {len(WORKSPACES)}\n"
            f"Metrics to track: {len(SYSTEM_METRICS_TO_TRACK)}",
            border_style="blue",
        )
    )

    api = API()

    console.print(
        f"\n[bold]Step 1:[/bold] Collecting experiment data from workspaces (parallel processing with {MAX_WORKERS} workers)..."
    )
    all_data = collect_experiment_data(
        WORKSPACES, max_workers=MAX_WORKERS, cutoff_date=(2020, 10, 20)
    )
    experiment_map = {data["experimentKey"]: data for data in all_data}

    if not all_data:
        console.print("[yellow]No experiment data found.[/yellow]")
        return

    experiment_keys = [data["experimentKey"] for data in all_data]

    # Display summary table
    table = Table(title="Collection Summary", box=box.ROUNDED)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    table.add_row("Total experiments found", str(len(experiment_keys)))
    table.add_row("Workspaces processed", str(len(WORKSPACES)))
    table.add_row("Metrics per experiment", str(len(SYSTEM_METRICS_TO_TRACK)))

    console.print("\n")
    console.print(table)

    console.print(
        f"\n[bold]Step 2:[/bold] Fetching metric data for experiments (parallel processing with {MAX_WORKERS} workers)..."
    )
    all_metrics = get_metric_data(experiment_keys, max_workers=MAX_WORKERS)

    console.print(
        f"\n[bold green]✓[/bold green] [green]Completed! Processed {len(all_metrics)} metric batches.[/green]"
    )

    for key in all_metrics:
        ws_project = experiment_map[key]
        all_metrics[key].update(
            {
                "workspace": ws_project["workspace"],
                "project_name": ws_project["project_name"],
            }
        )

    return all_metrics


if __name__ == "__main__":
    all_metrics = main()
