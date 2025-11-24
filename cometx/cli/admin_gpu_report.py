"""
Charts:

1. avg(metric) vs workspace (bar)
2. Time series: legend - project (line)
   max(metric) vs month by workspace

"""

import datetime
import os
import warnings
import webbrowser
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime as dt
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse
from xml.sax.saxutils import escape

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from comet_ml import API
from comet_ml.query import Metadata
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as RLImage
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer
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

from .admin_utils import save_chart

# Suppress matplotlib warnings about non-GUI backend
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

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


def get_experiment_data(workspace, project_name, start_date, end_date):
    # Build query with date range (inclusive start, exclusive end)
    query = Metadata("start_server_timestamp") >= datetime.datetime(*start_date)
    if end_date:
        query = query & (
            Metadata("start_server_timestamp") < datetime.datetime(*end_date)
        )
    results = search(workspace, project_name, query)
    # Return a list of dictionaries, one per experiment
    return [
        {key: data[key] for key in ["experimentKey", "server_timestamp"]}
        for data in results.get("experiments", [])
    ]


def process_workspace_project(
    workspace: str,
    project_name: str,
    start_date: Tuple[int, int, int],
    end_date: Tuple[int, int, int] = None,
) -> Tuple[str, str, List[Dict[str, Any]]]:
    """Process a single workspace-project pair and return the result."""
    try:
        experiment_data = get_experiment_data(
            workspace, project_name, start_date, end_date
        )
        return (workspace, project_name, experiment_data)
    except Exception as e:
        console.print(f"[red]Error processing {workspace}/{project_name}: {e}[/red]")
        return (workspace, project_name, [])


def collect_experiment_data(
    workspace_project_pairs, max_workers, start_date, end_date=None
):
    """Collect experiment data from workspace-project pairs in parallel."""
    all_data = []

    total_pairs = len(workspace_project_pairs)
    if total_pairs == 0:
        console.print("[yellow]No workspace-project pairs to process.[/yellow]")
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
            future_to_pair = {
                executor.submit(
                    process_workspace_project,
                    workspace,
                    project_name,
                    start_date,
                    end_date,
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
    chunk: List[str], batch_index: int, total_batches: int, metrics_to_track: List[str]
) -> Tuple[int, Dict[str, Any]]:
    """Process a single batch of experiment keys and return the metrics."""
    try:
        metrics = api.get_metrics_for_chart(chunk, metrics_to_track)
        return (batch_index, metrics)
    except Exception as e:
        console.print(
            f"[red]Error processing batch {batch_index + 1}/{total_batches}: {e}[/red]"
        )
        return (batch_index, {})


def get_metric_data(experiment_keys, max_workers, metrics_to_track):
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
                executor.submit(
                    process_metric_batch, chunk, i, len(chunks), metrics_to_track
                ): i
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


MAX_WORKERS = min(
    32, os.cpu_count() + 4
)  # Number of parallel threads for processing workspace-project pairs and metric batches
DEFAULT_SYSTEM_METRICS_TO_TRACK = [
    "sys.gpu.0.gpu_utilization",  # percent
    "sys.gpu.0.memory_utilization",  # percent
    "sys.gpu.0.used_memory",  # gb
    "sys.gpu.0.power_usage",  # watt
    "sys.gpu.0.temperature",  # celsius
]


def parse_date(date_str):
    """Parse a date string in YYYY-MM-DD format to a tuple (year, month, day)."""
    try:
        dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        return (dt.year, dt.month, dt.day)
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Expected YYYY-MM-DD")


def extract_metric_value(metric_data, metric_name):
    """
    Extract the maximum value for a metric from metric data.

    Args:
        metric_data: Dictionary containing metric data from get_metrics_for_chart
        metric_name: Name of the metric to extract

    Returns:
        float or None: Maximum value of the metric, or None if not found
    """
    if not metric_data or "metrics" not in metric_data:
        return None

    for metric in metric_data.get("metrics", []):
        if metric.get("metricName") == metric_name:
            values = metric.get("values", [])
            if values:
                return max(values)
    return None


def process_metrics_for_charts(all_metrics, experiment_map, metrics_to_track):
    """
    Process metric data to prepare for chart generation.

    Args:
        all_metrics: Dictionary keyed by experiment key with metric data
        experiment_map: Dictionary keyed by experiment key with experiment metadata
        metrics_to_track: List of metric names to process

    Returns:
        tuple: (workspace_avg_data, monthly_max_data)
            workspace_avg_data: Dict[metric_name][workspace] = average_value
            monthly_max_data: Dict[metric_name][month_key][workspace] = max_value
    """
    # Structure: metric_name -> workspace -> list of values
    workspace_values = defaultdict(lambda: defaultdict(list))

    # Structure: metric_name -> month_key -> workspace -> list of values
    monthly_values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for exp_key, metric_data in all_metrics.items():
        if exp_key not in experiment_map:
            continue

        workspace = experiment_map[exp_key].get("workspace", "Unknown")
        server_timestamp = experiment_map[exp_key].get("server_timestamp")

        # Determine month key from server_timestamp
        month_key = None
        if server_timestamp:
            try:
                # server_timestamp might be a timestamp (milliseconds) or datetime
                if isinstance(server_timestamp, (int, float)):
                    # If it's a large number, assume milliseconds, otherwise seconds
                    if server_timestamp > 1e10:
                        dt = datetime.datetime.fromtimestamp(server_timestamp / 1000)
                    else:
                        dt = datetime.datetime.fromtimestamp(server_timestamp)
                elif isinstance(server_timestamp, str):
                    # Try parsing as ISO format
                    dt = datetime.datetime.fromisoformat(
                        server_timestamp.replace("Z", "+00:00")
                    )
                else:
                    dt = server_timestamp
                month_key = dt.strftime("%Y-%m")
            except (ValueError, TypeError, AttributeError):
                pass

        # Extract values for each metric
        for metric_name in metrics_to_track:
            value = extract_metric_value(metric_data, metric_name)
            if value is not None:
                workspace_values[metric_name][workspace].append(value)
                if month_key:
                    monthly_values[metric_name][month_key][workspace].append(value)

    # Calculate averages per workspace
    workspace_avg_data = {}
    for metric_name in metrics_to_track:
        workspace_avg_data[metric_name] = {}
        for workspace, values in workspace_values[metric_name].items():
            if values:
                workspace_avg_data[metric_name][workspace] = sum(values) / len(values)

    # Calculate max per month per workspace
    monthly_max_data = {}
    for metric_name in metrics_to_track:
        monthly_max_data[metric_name] = {}
        for month_key in sorted(monthly_values[metric_name].keys()):
            monthly_max_data[metric_name][month_key] = {}
            for workspace, values in monthly_values[metric_name][month_key].items():
                if values:
                    monthly_max_data[metric_name][month_key][workspace] = max(values)

    return workspace_avg_data, monthly_max_data


def create_workspace_avg_chart(
    workspace_avg_data, metric_name, png_filename=None, debug=False
):
    """
    Create a bar chart showing average metric value per workspace.

    Args:
        workspace_avg_data: Dictionary of metric_name -> workspace -> average_value
        metric_name: Name of the metric to chart
        png_filename: Optional filename for the PNG
        debug: If True, print debug information

    Returns:
        str: The filename of the saved PNG chart, or None if no data
    """
    if metric_name not in workspace_avg_data or not workspace_avg_data[metric_name]:
        if debug:
            print(f"No data for metric {metric_name}")
        return None

    data = workspace_avg_data[metric_name]

    if png_filename is None:
        safe_metric = metric_name.replace(".", "_").replace("/", "_")
        png_filename = f"gpu_report_avg_{safe_metric}_by_workspace.png"

    # Sort workspaces by average value (descending)
    sorted_workspaces = sorted(data.items(), key=lambda x: x[1], reverse=True)
    workspaces = [ws for ws, _ in sorted_workspaces]
    values = [val for _, val in sorted_workspaces]

    # Create bar chart
    fig, ax = plt.subplots(figsize=(14, 8))

    bars = ax.bar(
        range(len(workspaces)),
        values,
        color="steelblue",
        edgecolor="navy",
        alpha=0.7,
        width=0.8,
    )

    # Customize the chart
    ax.set_title(
        f"Average {metric_name} by Workspace",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Workspace", fontsize=14)
    ax.set_ylabel(f"Average {metric_name}", fontsize=14)

    # Set x-axis labels
    ax.set_xticks(range(len(workspaces)))
    ax.set_xticklabels(workspaces, rotation=45, ha="right")

    # Add grid
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5, axis="y")

    # Add value labels on bars
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    return save_chart(png_filename, fig, debug=debug)


def create_monthly_max_chart(
    monthly_max_data, metric_name, png_filename=None, debug=False
):
    """
    Create a time series line chart showing max metric value per month with workspace legend.

    Args:
        monthly_max_data: Dictionary of metric_name -> month_key -> workspace -> max_value
        metric_name: Name of the metric to chart
        png_filename: Optional filename for the PNG
        debug: If True, print debug information

    Returns:
        str: The filename of the saved PNG chart, or None if no data
    """
    if metric_name not in monthly_max_data or not monthly_max_data[metric_name]:
        if debug:
            print(f"No data for metric {metric_name}")
        return None

    data = monthly_max_data[metric_name]

    if png_filename is None:
        safe_metric = metric_name.replace(".", "_").replace("/", "_")
        png_filename = f"gpu_report_max_{safe_metric}_by_month.png"

    # Collect all unique workspaces and existing months
    all_workspaces = set()
    existing_months = sorted(data.keys())

    for month_data in data.values():
        all_workspaces.update(month_data.keys())

    all_workspaces = sorted(all_workspaces)

    if not all_workspaces or not existing_months:
        if debug:
            print(f"No workspace or month data for metric {metric_name}")
        return None

    # Generate complete list of months from earliest to latest
    if len(existing_months) > 0:
        # Parse first and last month
        start_year, start_month = map(int, existing_months[0].split("-"))
        end_year, end_month = map(int, existing_months[-1].split("-"))

        # Generate all months between start and end
        all_months = []
        current_year = start_year
        current_month = start_month

        while True:
            month_key = f"{current_year:04d}-{current_month:02d}"
            all_months.append(month_key)

            if current_year == end_year and current_month == end_month:
                break

            # Move to next month
            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1
    else:
        all_months = existing_months

    # Create line chart
    fig, ax = plt.subplots(figsize=(14, 8))

    # Generate distinct colors for each workspace
    colors_list = cm.tab10(range(len(all_workspaces)))

    # Plot line for each workspace
    for i, workspace in enumerate(all_workspaces):
        values = []
        for month in all_months:
            value = data.get(month, {}).get(workspace)
            values.append(value if value is not None else None)

        ax.plot(
            range(len(all_months)),
            values,
            label=workspace,
            color=colors_list[i],
            marker="o",
            linewidth=2,
            markersize=6,
        )

    # Customize the chart
    ax.set_title(
        f"Maximum {metric_name} by Month",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Month", fontsize=14)
    ax.set_ylabel(f"Maximum {metric_name}", fontsize=14)

    # Set x-axis labels - show every Nth to avoid crowding
    max_labels = 20
    step = max(1, len(all_months) // max_labels)
    x_ticks = range(0, len(all_months), step)
    x_labels = [all_months[i] for i in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")

    # Add grid
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    # Add legend
    ax.legend(
        loc="best",
        ncol=min(len(all_workspaces), 3),
        fontsize=10,
        framealpha=0.9,
    )

    return save_chart(png_filename, fig, debug=debug)


def extract_website_name(base_url):
    """
    Extract website name (domain) from base_url.

    Example:
        'https://www.comet.com/api/rest/v2/' -> 'www.comet.com'
        'https://app.comet.ml/api/rest/v2/' -> 'app.comet.ml'

    Args:
        base_url: Full URL string

    Returns:
        str: Domain name (e.g., 'www.comet.com') or 'Unknown' if parsing fails
    """
    try:
        parsed = urlparse(base_url)
        return parsed.netloc or "Unknown"
    except Exception:
        return "Unknown"


def add_chart_to_flowables(flowables, png_file, title=None, debug=False):
    """
    Add a chart image to the list of flowables for ReportLab.

    Args:
        flowables: List of ReportLab flowables to append to
        png_file: Path to PNG image file
        title: Optional title for the chart
        debug: If True, print debug information
    """
    if not os.path.exists(png_file):
        if debug:
            print(f"Chart file not found: {png_file}")
        return

    # Add title if provided
    if title:
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontSize=14,
            textColor=colors.HexColor("#000000"),
            spaceAfter=12,
            alignment=1,  # Center alignment
        )
        flowables.append(Paragraph(escape(title), title_style))
        flowables.append(Spacer(1, 0.2 * inch))

    # Calculate dimensions to fit image on page with margins
    img = Image.open(png_file)
    img_width, img_height = img.size
    aspect_ratio = img_width / img_height

    # Scale to fit page with margins
    max_width = 7.5 * inch  # Leave margin
    max_height = 9.0 * inch  # Leave margin

    if aspect_ratio > max_width / max_height:
        # Image is wider
        display_width = max_width
        display_height = max_width / aspect_ratio
    else:
        # Image is taller
        display_height = max_height
        display_width = max_height * aspect_ratio

    # Add image to flowables
    rl_image = RLImage(png_file, width=display_width, height=display_height)
    flowables.append(rl_image)
    flowables.append(Spacer(1, 0.3 * inch))

    if debug:
        print(f"Added chart {png_file} to PDF")


def add_statistics_to_flowables(
    flowables,
    result_data,
    workspace_projects_input,
    website_name=None,
    debug=False,
):
    """
    Create statistics content and add it to the list of flowables for ReportLab.

    Args:
        flowables: List of ReportLab flowables to append to
        result_data: Dictionary with result data from main()
        workspace_projects_input: List of workspace/project strings that were requested
        website_name: Optional website name (domain) to display in the title
        debug: If True, print debug information
    """
    if not result_data:
        return

    # Get styles
    styles = getSampleStyleSheet()

    # Create custom styles
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=18,
        textColor=colors.HexColor("#000000"),
        spaceAfter=24,
        alignment=1,  # Center alignment
    )

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=12,
        textColor=colors.HexColor("#000000"),
        spaceAfter=12,
        spaceBefore=12,
    )

    body_style = ParagraphStyle(
        "BodyText",
        parent=styles["Normal"],
        fontSize=10,
        spaceAfter=6,
        leftIndent=0,
    )

    # Title
    flowables.append(
        Paragraph(escape("GPU Usage Report - Summary Statistics"), title_style)
    )
    if website_name:
        flowables.append(Paragraph(escape(website_name), title_style))
    flowables.append(Spacer(1, 0.3 * inch))

    # Overall Statistics section
    flowables.append(Paragraph("Overall Statistics", heading_style))

    all_metrics = result_data.get("metrics", {})
    workspace_avg_data = result_data.get("workspace_avg_data", {})

    num_experiments = len(all_metrics)
    flowables.append(
        Paragraph(
            escape(f"Total Number of Experiments: {num_experiments:,}"), body_style
        )
    )

    # Get unique workspaces
    unique_workspaces = set()
    for exp_data in all_metrics.values():
        workspace = exp_data.get("workspace", "Unknown")
        unique_workspaces.add(workspace)

    flowables.append(
        Paragraph(
            escape(f"Total Number of Workspaces: {len(unique_workspaces):,}"),
            body_style,
        )
    )

    # Get metrics tracked
    metrics_tracked = list(workspace_avg_data.keys())
    flowables.append(
        Paragraph(escape(f"Metrics Tracked: {len(metrics_tracked)}"), body_style)
    )

    if metrics_tracked:
        flowables.append(Spacer(1, 0.1 * inch))
        flowables.append(Paragraph("Metrics:", body_style))
        for metric in metrics_tracked:
            flowables.append(Paragraph(escape(f"  • {metric}"), body_style))

    flowables.append(Spacer(1, 0.2 * inch))

    # Breakdown by workspace (if multiple workspaces)
    if len(unique_workspaces) > 1:
        flowables.append(Paragraph("Breakdown by Workspace", heading_style))

        for workspace in sorted(unique_workspaces):
            workspace_experiments = [
                exp for exp in all_metrics.values() if exp.get("workspace") == workspace
            ]
            exp_count = len(workspace_experiments)

            flowables.append(Paragraph(escape(f"{workspace}:"), body_style))
            flowables.append(
                Paragraph(escape(f"  Experiments: {exp_count:,}"), body_style)
            )

            # Add average metrics for this workspace
            for metric_name in metrics_tracked:
                avg_value = workspace_avg_data.get(metric_name, {}).get(workspace)
                if avg_value is not None:
                    flowables.append(
                        Paragraph(
                            escape(f"  Average {metric_name}: {avg_value:.2f}"),
                            body_style,
                        )
                    )

            flowables.append(Spacer(1, 0.1 * inch))


def generate_pdf_report(
    result_data,
    workspace_projects,
    pdf_filename=None,
    api=None,
    debug=False,
):
    """
    Generate a PDF report from the GPU report data.

    Args:
        result_data: Dictionary with result data from main()
        workspace_projects: List of workspace/project strings
        pdf_filename: Optional filename for the PDF
        api: Optional API instance to extract website name
        debug: If True, print debug information

    Returns:
        str: The filename of the generated PDF
    """
    if not result_data or not result_data.get("charts"):
        if debug:
            print("No charts to include in PDF")
        return None

    if pdf_filename is None:
        filename_parts = [wp.replace("/", "_") for wp in workspace_projects]
        pdf_filename = f"gpu_report_{'_'.join(filename_parts)}.pdf"

    try:
        # Generate timestamp for footer
        timestamp = dt.now().strftime("Generated: %Y-%m-%d %H:%M:%S")

        # Create document with margins
        doc = SimpleDocTemplate(
            pdf_filename,
            pagesize=letter,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=1.0 * inch,  # Extra space for footer
        )

        # Define footer drawing function
        def draw_footer(canvas, doc):
            """Draw footer on every page"""
            canvas.saveState()
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(colors.HexColor("#808080"))
            footer_y = 0.5 * inch
            canvas.drawCentredString(letter[0] / 2.0, footer_y, timestamp)
            canvas.restoreState()

        # Assign footer callbacks
        doc.onFirstPage = draw_footer
        doc.onLaterPages = draw_footer

        # Build list of flowables
        flowables = []

        # Extract website name from API base_url
        website_name = None
        if api:
            try:
                base_url = api._client.base_url
                website_name = extract_website_name(base_url)
            except Exception:
                pass

        # Add statistics content first
        add_statistics_to_flowables(
            flowables,
            result_data,
            workspace_projects,
            website_name=website_name,
            debug=debug,
        )

        # Add charts
        chart_files = result_data.get("charts", [])

        # Group charts by type (avg vs monthly)
        avg_charts = [f for f in chart_files if "avg" in f]
        monthly_charts = [f for f in chart_files if "max" in f]

        # Add workspace average charts
        if avg_charts:
            flowables.append(PageBreak())
            flowables.append(
                Paragraph(
                    escape("Average Metrics by Workspace"),
                    ParagraphStyle(
                        "SectionHeading",
                        parent=getSampleStyleSheet()["Heading1"],
                        fontSize=16,
                        textColor=colors.HexColor("#000000"),
                        spaceAfter=12,
                        spaceBefore=12,
                        alignment=1,
                    ),
                )
            )
            flowables.append(Spacer(1, 0.2 * inch))

            for chart_file in avg_charts:
                # Extract metric name from filename
                metric_name = (
                    chart_file.replace("gpu_report_avg_", "")
                    .replace("_by_workspace.png", "")
                    .replace("_", ".")
                )
                add_chart_to_flowables(
                    flowables,
                    chart_file,
                    title=f"Average {metric_name} by Workspace",
                    debug=debug,
                )
                flowables.append(PageBreak())

        # Add monthly max charts
        if monthly_charts:
            if avg_charts:
                flowables.append(PageBreak())
            flowables.append(
                Paragraph(
                    escape("Maximum Metrics by Month"),
                    ParagraphStyle(
                        "SectionHeading",
                        parent=getSampleStyleSheet()["Heading1"],
                        fontSize=16,
                        textColor=colors.HexColor("#000000"),
                        spaceAfter=12,
                        spaceBefore=12,
                        alignment=1,
                    ),
                )
            )
            flowables.append(Spacer(1, 0.2 * inch))

            for chart_file in monthly_charts:
                # Extract metric name from filename
                metric_name = (
                    chart_file.replace("gpu_report_max_", "")
                    .replace("_by_month.png", "")
                    .replace("_", ".")
                )
                add_chart_to_flowables(
                    flowables,
                    chart_file,
                    title=f"Maximum {metric_name} by Month",
                    debug=debug,
                )
                if chart_file != monthly_charts[-1]:
                    flowables.append(PageBreak())

        # Build the PDF
        if debug:
            print("Building PDF...")
        doc.build(flowables)
        print(f"\nPDF report generated successfully: {pdf_filename}")
        return pdf_filename

    except Exception as e:
        print(f"Error generating PDF: {e}")
        if debug:
            import traceback

            traceback.print_exc()
        return None


def open_pdf(pdf_path, debug=False):
    """Open PDF file using the default system application"""
    if os.path.exists(pdf_path):
        webbrowser.open(f"file://{os.path.abspath(pdf_path)}")
        if debug:
            print(f"Opening PDF: {pdf_path}")
    else:
        print(f"PDF file not found: {pdf_path}")


def parse_workspace_projects(api, workspace_projects):
    """
    Parse workspace/project arguments into a list of (workspace, project) tuples.

    Args:
        api: Comet API instance
        workspace_projects: List of strings in format "workspace" or "workspace/project"

    Returns:
        List of (workspace, project) tuples
    """
    workspace_project_pairs = []
    for workspace_project in workspace_projects:
        if "/" in workspace_project:
            workspace, project = workspace_project.split("/", 1)
            workspace_project_pairs.append((workspace, project))
        else:
            workspace = workspace_project
            try:
                projects = api.get_projects(workspace)
                for project in projects:
                    workspace_project_pairs.append((workspace, project))
            except Exception as e:
                console.print(
                    f"[red]Error getting projects for workspace {workspace}: {e}[/red]"
                )
    return workspace_project_pairs


def main(workspace_projects, start_date, end_date=None, metrics=None, max_workers=None):
    """
    Main function for GPU report generation.

    Args:
        workspace_projects: List of workspace/project strings
        start_date: Start date as tuple (year, month, day) or string YYYY-MM-DD
        end_date: End date as tuple (year, month, day) or string YYYY-MM-DD (optional)
        metrics: List of metric names to track (optional, uses defaults if not provided)
        max_workers: Number of parallel workers (optional, uses default if not provided)

    Returns:
        Dictionary of metrics keyed by experiment key
    """
    global api

    # Parse dates if they're strings
    if isinstance(start_date, str):
        start_date = parse_date(start_date)
    if isinstance(end_date, str):
        end_date = parse_date(end_date)

    # Use defaults if not provided
    if metrics is None:
        metrics = DEFAULT_SYSTEM_METRICS_TO_TRACK
    if max_workers is None:
        max_workers = MAX_WORKERS

    api = API()

    # Parse workspace/project pairs
    workspace_project_pairs = parse_workspace_projects(api, workspace_projects)

    if not workspace_project_pairs:
        console.print("[yellow]No workspace-project pairs found.[/yellow]")
        return {}

    console.print(
        Panel.fit(
            "[bold blue]GPU Usage Data Collection[/bold blue]\n"
            f"Workspace-project pairs: {len(workspace_project_pairs)}\n"
            f"Metrics to track: {len(metrics)}\n"
            f"Date range: {start_date[0]}-{start_date[1]:02d}-{start_date[2]:02d}"
            + (
                f" to {end_date[0]}-{end_date[1]:02d}-{end_date[2]:02d}"
                if end_date
                else " onwards"
            ),
            border_style="blue",
        )
    )

    console.print(
        f"\n[bold]Step 1:[/bold] Collecting experiment data from workspaces (parallel processing with {max_workers} workers)..."
    )
    all_data = collect_experiment_data(
        workspace_project_pairs,
        max_workers=max_workers,
        start_date=start_date,
        end_date=end_date,
    )
    experiment_map = {data["experimentKey"]: data for data in all_data}

    if not all_data:
        console.print("[yellow]No experiment data found.[/yellow]")
        return {}

    experiment_keys = [data["experimentKey"] for data in all_data]

    # Display summary table
    table = Table(title="Collection Summary", box=box.ROUNDED)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    table.add_row("Total experiments found", str(len(experiment_keys)))
    table.add_row(
        "Workspace-project pairs processed", str(len(workspace_project_pairs))
    )
    table.add_row("Metrics per experiment", str(len(metrics)))

    console.print("\n")
    console.print(table)

    console.print(
        f"\n[bold]Step 2:[/bold] Fetching metric data for experiments (parallel processing with {max_workers} workers)..."
    )
    all_metrics = get_metric_data(
        experiment_keys, max_workers=max_workers, metrics_to_track=metrics
    )

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

    # Generate charts
    console.print(
        f"\n[bold]Step 3:[/bold] Generating charts for {len(metrics)} metrics..."
    )

    workspace_avg_data, monthly_max_data = process_metrics_for_charts(
        all_metrics, experiment_map, metrics
    )

    chart_files = []

    for metric_name in metrics:
        # Create workspace average chart
        avg_chart = create_workspace_avg_chart(
            workspace_avg_data, metric_name, debug=False
        )
        if avg_chart:
            chart_files.append(avg_chart)
            console.print(
                f"[green]✓[/green] Created workspace average chart: {avg_chart}"
            )

        # Create monthly max chart
        monthly_chart = create_monthly_max_chart(
            monthly_max_data, metric_name, debug=False
        )
        if monthly_chart:
            chart_files.append(monthly_chart)
            console.print(
                f"[green]✓[/green] Created monthly max chart: {monthly_chart}"
            )

    console.print(
        f"\n[bold green]✓[/bold green] [green]Chart generation completed! Created {len(chart_files)} charts.[/green]"
    )

    result_data = {
        "metrics": all_metrics,
        "charts": chart_files,
        "workspace_avg_data": workspace_avg_data,
        "monthly_max_data": monthly_max_data,
    }

    # Generate PDF report
    console.print("\n[bold]Step 4:[/bold] Generating PDF report...")
    pdf_filename = generate_pdf_report(
        result_data,
        workspace_projects,
        api=api,
        debug=False,
    )

    if pdf_filename:
        result_data["pdf_file"] = pdf_filename
        console.print(
            f"\n[bold green]✓[/bold green] [green]PDF report generated: {pdf_filename}[/green]"
        )

    return result_data
