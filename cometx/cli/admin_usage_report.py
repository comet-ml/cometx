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
#  Copyright (c) 2024 Cometx Development
#      Team. All rights reserved.
# ****************************************
"""
Usage report functionality for admin commands

Generate usage reports with experiment counts by month for one or more workspaces/projects.
Multiple workspaces/projects are combined into a single chart with a legend.

Examples:
    cometx admin usage-report WORKSPACE
    cometx admin usage-report WORKSPACE WORKSPACE
    cometx admin usage-report WORKSPACE/PROJECT WORKSPACE/PROJECT
    cometx admin usage-report WORKSPACE WORKSPACE/PROJECT

"""

import os
import warnings
import webbrowser
from collections import defaultdict
from datetime import datetime
from xml.sax.saxutils import escape

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from comet_ml import API
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as RLImage
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable=None, desc=None, total=None, disable=False):
        if iterable is None:

            class FakeProgressBar:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

                def update(self, n=1):
                    pass

            return FakeProgressBar()
        return iterable


# Suppress matplotlib warnings about non-GUI backend
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Maximum number of datasets (workspaces/projects) per chart
# If more datasets are provided, multiple charts will be generated
MAX_DATASETS_PER_CHART = 5


def debug_print(debug, *args, **kwargs):
    """Print only if debug is True"""
    if debug:
        print(*args, **kwargs)


def add_chart_to_flowables(
    flowables,
    png_file,
    workspace_project=None,
    chart_num=None,
    total_charts=None,
    debug=False,
):
    """
    Add a chart image to the list of flowables for ReportLab.

    Args:
        flowables: List of ReportLab flowables to append to
        png_file: Path to PNG image file
        workspace_project: Optional workspace/project string for title
        chart_num: Optional chart number (1-based)
        total_charts: Optional total number of charts
    """
    if os.path.exists(png_file):
        # Add title if we have workspace_project info
        if workspace_project:
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                "CustomTitle",
                parent=styles["Heading1"],
                fontSize=14,
                textColor=colors.HexColor("#000000"),
                spaceAfter=12,
                alignment=1,  # Center alignment
            )
            # Build title with chart number if available
            if chart_num is not None and total_charts is not None and total_charts > 1:
                title = f"Experiments by Month ({chart_num} of {total_charts})"
            else:
                title = "Experiments by Month"
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

        debug_print(debug, f"Added chart {png_file} to PDF")


def generate_experiment_chart(api, workspace, project, debug=False):
    """
    Collect experiment data for a workspace/project.

    Returns a dictionary with the collected data, ready for chart generation.
    Does not generate any charts.
    """
    workspace_project = f"{workspace}/{project}"
    debug_print(
        debug, f"Fetching experiments for workspace: {workspace}, project: {project}"
    )
    project_data = api._client.get_project_experiments(workspace, project)

    if not project_data:
        debug_print(debug, f"API returned None for {workspace_project}")
        return {}

    if not isinstance(project_data, dict):
        debug_print(
            debug,
            f"API returned unexpected data type: {type(project_data)} for {workspace_project}",
        )
        return {}

    if "experiments" not in project_data:
        debug_print(
            debug, f"No 'experiments' key in API response for {workspace_project}"
        )
        debug_print(
            debug,
            f"Available keys: {list(project_data.keys()) if project_data else 'None'}",
        )
        return {}

    experiments = project_data["experiments"]
    debug_print(debug, f"Found {len(experiments)} experiments for {workspace_project}")

    # Get GPU data:
    column_data = api._client.get_project_columns(workspace, project)
    experiment_keys = [exp["experimentKey"] for exp in experiments]
    metric_names = [
        item["name"]
        for item in column_data["columns"]
        if item["name"].startswith("sys.gpu")
    ]
    metric_data = api.get_metrics_for_chart(experiment_keys, metric_names)
    for experiment_key in metric_data:
        for metric in metric_data[experiment_key]["metrics"]:
            print(metric["metricName"])
            print(metric["values"])
            print(metric["timestamps"])
            print(metric["durations"])

    # Group experiments by month/year based on startTimeMillis
    monthly_counts = defaultdict(int)
    # Track run times for statistics
    total_run_time_seconds = 0
    run_time_count = 0

    for exp in experiments:
        if "startTimeMillis" in exp and exp["startTimeMillis"]:
            # Convert milliseconds to seconds, then to datetime
            start_time = datetime.fromtimestamp(exp["startTimeMillis"] / 1000)
            # Format as YYYY-MM for grouping
            month_key = start_time.strftime("%Y-%m")
            monthly_counts[month_key] += 1

            # Calculate run time if both start and end times are available
            if "endTimeMillis" in exp and exp["endTimeMillis"]:
                run_time_ms = exp["endTimeMillis"] - exp["startTimeMillis"]
                if run_time_ms > 0:
                    run_time_seconds = run_time_ms / 1000
                    total_run_time_seconds += run_time_seconds
                    run_time_count += 1

    if not monthly_counts:
        debug_print(debug, "No experiments with valid start times found")
        return {}

    # Create a complete range of months from first to last experiment
    all_months = sorted(monthly_counts.keys())
    if not all_months:
        debug_print(debug, "No valid months found")
        return {}

    # Generate all months between first and last
    start_date = datetime.strptime(all_months[0], "%Y-%m")
    end_date = datetime.strptime(all_months[-1], "%Y-%m")

    complete_months = []
    current_date = start_date
    while current_date <= end_date:
        month_key = current_date.strftime("%Y-%m")
        complete_months.append(month_key)
        # Move to next month
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1)

    # Fill in counts for all months (0 for months with no experiments)
    counts = [monthly_counts[month] for month in complete_months]

    # Display summary statistics (only in debug mode)
    debug_print(debug, f"\nSummary for {workspace_project}:")
    debug_print(debug, f"Total experiments: {sum(counts)}")
    debug_print(debug, f"Date range: {complete_months[0]} to {complete_months[-1]}")
    debug_print(debug, f"Months with experiments: {sum(1 for c in counts if c > 0)}")
    debug_print(
        debug, f"Months with zero experiments: {sum(1 for c in counts if c == 0)}"
    )
    debug_print(debug, f"Average experiments per month: {sum(counts)/len(counts):.1f}")

    return {
        "total_experiments": sum(counts),
        "monthly_counts": dict(monthly_counts),
        "complete_monthly_counts": dict(zip(complete_months, counts)),
        "complete_months": complete_months,
        "counts": counts,
        "date_range": (complete_months[0], complete_months[-1]),
        "workspace_project": workspace_project,
        "total_run_time_seconds": total_run_time_seconds,
        "run_time_count": run_time_count,
        "experiments": experiments,  # Store raw experiments for statistics
    }


def create_chart_from_data(data, png_filename=None, debug=False):
    """
    Create a bar chart from experiment data and save it as PNG.

    Args:
        data: Dictionary with experiment data (from generate_experiment_chart)
        png_filename: Optional filename for the PNG. If not provided, generates from workspace_project.

    Returns:
        str: The filename of the saved PNG chart
    """
    if not data:
        return None

    workspace_project = data.get("workspace_project", "Unknown")
    complete_months = data.get("complete_months", [])
    counts = data.get("counts", [])

    if not complete_months or not counts:
        debug_print(debug, f"No data to create chart for {workspace_project}")
        return None

    if png_filename is None:
        png_filename = (
            f"experiment_usage_report_{workspace_project.replace('/', '_')}.png"
        )

    # Create bar chart with complete timeline
    plt.figure(figsize=(14, 8))

    # Create bars with different colors for zero vs non-zero values
    bar_colors = ["lightcoral" if count == 0 else "steelblue" for count in counts]
    bars = plt.bar(
        range(len(complete_months)),
        counts,
        color=bar_colors,
        edgecolor="navy",
        alpha=0.7,
        width=0.8,
    )

    # Customize the chart
    plt.title(
        f"Experiment Count by Month - {workspace_project}",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Month", fontsize=14)
    plt.ylabel("Number of Experiments", fontsize=14)

    # Set x-axis labels - show every 3rd month to avoid crowding
    step = max(1, len(complete_months) // 20)  # Show about 20 labels max
    x_ticks = range(0, len(complete_months), step)
    x_labels = [complete_months[i] for i in x_ticks]
    plt.xticks(x_ticks, x_labels, rotation=45, ha="right")

    # Add grid for better readability
    plt.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    # Add value labels on bars (only for non-zero values to avoid clutter)
    for i, (bar, count) in enumerate(zip(bars, counts)):
        if count > 0:  # Only label non-zero bars
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                str(count),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the chart as PNG
    plt.savefig(png_filename, dpi=300, bbox_inches="tight")
    debug_print(debug, f"Bar chart saved as: {png_filename}")

    plt.close()

    return png_filename


def create_combined_chart_from_data(
    data_list, png_filename=None, date_range=None, debug=False
):
    """
    Create a combined bar chart from multiple experiment datasets with a legend.

    Args:
        data_list: List of dictionaries with experiment data (from generate_experiment_chart)
        png_filename: Optional filename for the PNG. If not provided, generates a default name.
        date_range: Optional tuple (start_month, end_month) as strings in "YYYY-MM" format.
                    If provided, all datasets will be aligned to this date range.
                    If None, the date range is calculated from the datasets.

    Returns:
        str: The filename of the saved PNG chart
    """
    if not data_list:
        return None

    # Filter out empty data
    data_list = [
        d for d in data_list if d and d.get("complete_months") and d.get("counts")
    ]

    if not data_list:
        debug_print(debug, "No valid data to create combined chart")
        return None

    if png_filename is None:
        workspace_projects = [d.get("workspace_project", "Unknown") for d in data_list]
        filename_parts = [wp.replace("/", "_") for wp in workspace_projects]
        png_filename = (
            f"experiment_usage_report_combined_{'_'.join(filename_parts[:3])}.png"
        )
        if len(filename_parts) > 3:
            png_filename = png_filename.replace(
                ".png", f"_and_{len(filename_parts)-3}_more.png"
            )

    # Determine the date range to use
    if date_range:
        # Use provided date range
        overall_start = datetime.strptime(date_range[0], "%Y-%m")
        overall_end = datetime.strptime(date_range[1], "%Y-%m")
    else:
        # Find the complete date range across all datasets
        all_date_ranges = [
            data.get("date_range") for data in data_list if data.get("date_range")
        ]
        if not all_date_ranges:
            debug_print(debug, "No date ranges found in data")
            return None

        # Find the earliest start and latest end date
        start_dates = [datetime.strptime(dr[0], "%Y-%m") for dr in all_date_ranges]
        end_dates = [datetime.strptime(dr[1], "%Y-%m") for dr in all_date_ranges]
        overall_start = min(start_dates)
        overall_end = max(end_dates)

    # Generate complete list of months from overall start to end
    all_months = []
    current_date = overall_start
    while current_date <= overall_end:
        month_key = current_date.strftime("%Y-%m")
        all_months.append(month_key)
        # Move to next month
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1)

    if not all_months:
        debug_print(debug, "No months found in data")
        return None

    # For each dataset, align counts to the complete month list
    aligned_data = []
    for data in data_list:
        complete_monthly_counts = data.get("complete_monthly_counts", {})
        counts = [complete_monthly_counts.get(month, 0) for month in all_months]
        aligned_data.append(
            {
                "workspace_project": data.get("workspace_project", "Unknown"),
                "counts": counts,
            }
        )

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Generate distinct colors for each project
    colors = cm.tab10(range(len(aligned_data)))

    # Set up bar positions for grouped bars
    num_projects = len(aligned_data)
    num_months = len(all_months)
    bar_width = 0.8 / num_projects if num_projects > 1 else 0.8
    x_positions = range(num_months)

    # Create grouped bars
    bars_list = []
    for i, data in enumerate(aligned_data):
        # Calculate x offset for each group of bars
        offset = (i - (num_projects - 1) / 2) * bar_width
        x_pos = [x + offset for x in x_positions]

        bars = ax.bar(
            x_pos,
            data["counts"],
            width=bar_width,
            label=data["workspace_project"],
            color=colors[i],
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
        )
        bars_list.append(bars)

        # Add value labels on bars (only for non-zero values)
        for bar, count in zip(bars, data["counts"]):
            if count > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    str(count),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

    # Customize the chart
    ax.set_title(
        "Experiment Count by Month - Combined Projects",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Month", fontsize=14)
    ax.set_ylabel("Number of Experiments", fontsize=14)

    # Set x-axis labels - show every Nth month to avoid crowding
    step = max(1, num_months // 20)  # Show about 20 labels max
    x_ticks = range(0, num_months, step)
    x_labels = [all_months[i] for i in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5, axis="y")

    # Add legend
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the chart as PNG
    plt.savefig(png_filename, dpi=300, bbox_inches="tight")
    debug_print(debug, f"Combined bar chart saved as: {png_filename}")

    plt.close()

    return png_filename


def format_time(seconds):
    """Format seconds into a human-readable string (days, hours, minutes, seconds)"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f} hours"
    else:
        days = seconds / 86400
        hours = (seconds % 86400) / 3600
        if hours > 0:
            return f"{days:.1f} days, {hours:.1f} hours"
        return f"{days:.1f} days"


def add_statistics_to_flowables(
    flowables, all_results, workspace_projects_input, debug=False
):
    """
    Create statistics content and add it to the list of flowables for ReportLab.

    Args:
        flowables: List of ReportLab flowables to append to
        all_results: List of dictionaries with experiment data
        workspace_projects_input: List of workspace/project strings that were requested
    """
    if not all_results:
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

    indent_body_style = ParagraphStyle(
        "IndentBodyText",
        parent=styles["Normal"],
        fontSize=10,
        spaceAfter=6,
        leftIndent=20,
    )

    bold_body_style = ParagraphStyle(
        "BoldBodyText",
        parent=styles["Normal"],
        fontSize=10,
        fontName="Helvetica-Bold",
        spaceAfter=6,
        leftIndent=0,
    )

    # Calculate overall statistics
    total_experiments = sum(r.get("total_experiments", 0) for r in all_results)
    total_run_time_seconds = sum(
        r.get("total_run_time_seconds", 0) for r in all_results
    )
    run_time_count = sum(r.get("run_time_count", 0) for r in all_results)

    # Find date range across all results
    all_start_dates = []
    all_end_dates = []
    for result in all_results:
        date_range = result.get("date_range")
        if date_range:
            all_start_dates.append(datetime.strptime(date_range[0], "%Y-%m"))
            all_end_dates.append(datetime.strptime(date_range[1], "%Y-%m"))

    earliest_date = min(all_start_dates) if all_start_dates else None
    latest_date = max(all_end_dates) if all_end_dates else None

    # Find peak month
    combined_monthly_counts = defaultdict(int)
    for result in all_results:
        monthly_counts = result.get("monthly_counts", {})
        for month, count in monthly_counts.items():
            combined_monthly_counts[month] += count

    peak_month = (
        max(combined_monthly_counts.items(), key=lambda x: x[1])
        if combined_monthly_counts
        else None
    )

    # Calculate average run time
    avg_run_time_seconds = (
        total_run_time_seconds / run_time_count if run_time_count > 0 else 0
    )

    # Title
    flowables.append(Paragraph("Usage Report - Summary Statistics", title_style))
    flowables.append(Spacer(1, 0.3 * inch))

    # Overall Statistics section
    flowables.append(Paragraph("Overall Statistics", heading_style))

    flowables.append(
        Paragraph(
            escape(f"Total Number of Experiments: {total_experiments:,}"), body_style
        )
    )

    if total_run_time_seconds > 0:
        flowables.append(
            Paragraph(
                escape(f"Total Run Time: {format_time(total_run_time_seconds)}"),
                body_style,
            )
        )
        flowables.append(
            Paragraph(
                escape(f"Experiments with Run Time Data: {run_time_count:,}"),
                body_style,
            )
        )
        if run_time_count > 0:
            flowables.append(
                Paragraph(
                    escape(
                        f"Average Run Time per Experiment: {format_time(avg_run_time_seconds)}"
                    ),
                    body_style,
                )
            )

    if earliest_date and latest_date:
        flowables.append(
            Paragraph(
                escape(
                    f"Date Range: {earliest_date.strftime('%Y-%m')} to {latest_date.strftime('%Y-%m')}"
                ),
                body_style,
            )
        )
        # Calculate span in months
        months_span = (
            (latest_date.year - earliest_date.year) * 12
            + (latest_date.month - earliest_date.month)
            + 1
        )
        flowables.append(
            Paragraph(escape(f"Time Span: {months_span} months"), body_style)
        )

    flowables.append(
        Paragraph(
            escape(f"Number of Workspaces/Projects: {len(all_results)}"), body_style
        )
    )

    if peak_month:
        flowables.append(
            Paragraph(
                escape(f"Peak Month: {peak_month[0]} ({peak_month[1]:,} experiments)"),
                body_style,
            )
        )

    flowables.append(Spacer(1, 0.2 * inch))

    # Breakdown section
    flowables.append(Paragraph("Breakdown by Workspace/Project", heading_style))

    for i, result in enumerate(all_results):
        wp = result.get("workspace_project", "Unknown")
        exp_count = result.get("total_experiments", 0)
        run_time = result.get("total_run_time_seconds", 0)
        run_time_exp_count = result.get("run_time_count", 0)

        # Use KeepTogether to prevent breaking a workspace/project across pages
        workspace_content = [
            Paragraph(escape(f"{wp}:"), bold_body_style),
            Paragraph(escape(f"Experiments: {exp_count:,}"), indent_body_style),
        ]

        if run_time > 0:
            workspace_content.append(
                Paragraph(
                    escape(f"Total Run Time: {format_time(run_time)}"),
                    indent_body_style,
                )
            )
            if run_time_exp_count > 0:
                avg = run_time / run_time_exp_count
                workspace_content.append(
                    Paragraph(
                        escape(f"Average Run Time: {format_time(avg)}"),
                        indent_body_style,
                    )
                )

        flowables.append(KeepTogether(workspace_content))
        flowables.append(Spacer(1, 0.1 * inch))

    # Note: Footer is now drawn on every page via canvas callback, not as a flowable

    debug_print(debug, "Added statistics content to PDF")


def open_pdf(pdf_path, debug=False):
    """Open PDF file using the default system application"""
    if os.path.exists(pdf_path):
        webbrowser.open(f"file://{os.path.abspath(pdf_path)}")
        debug_print(debug, f"Opening PDF: {pdf_path}")
    else:
        print(f"PDF file not found: {pdf_path}")


def generate_usage_report(
    api, workspace_projects, no_open=False, max_datasets_per_chart=None, debug=False
):
    """
    Get usage report for one or more workspace/project combinations.

    Args:
        api: Comet API instance
        workspace_projects: String or list of strings in format "workspace" or "workspace/project"
        no_open: If True, don't automatically open the generated PDF file
        max_datasets_per_chart: Maximum number of datasets per chart. If None, uses MAX_DATASETS_PER_CHART constant.

    Returns:
        dict: The usage report data
    """
    if api is None:
        api = API()

    # Normalize to a list
    if isinstance(workspace_projects, str):
        workspace_projects = [workspace_projects]

    if not workspace_projects:
        print("ERROR: At least one workspace/project is required")
        return {}

    # Collect all data first from all workspace/project combinations
    all_results = []

    # First, collect all workspace/project pairs to process
    workspace_project_pairs = []
    for workspace_project in workspace_projects:
        if "/" in workspace_project:
            workspace, project = workspace_project.split("/", 1)
            projects = [project]
        else:
            workspace = workspace_project
            try:
                projects = api.get_projects(workspace)
            except Exception as e:
                print(f"ERROR: Could not get projects for workspace {workspace}: {e}")
                continue

        for project in projects:
            workspace_project_pairs.append((workspace, project))

    # Collect data with progress bar
    if workspace_project_pairs:
        with tqdm(
            total=len(workspace_project_pairs), desc="Collecting experiment data"
        ) as pbar:
            for workspace, project in workspace_project_pairs:
                results = generate_experiment_chart(
                    api, workspace, project, debug=debug
                )
                if results:
                    all_results.append(results)
                pbar.update(1)

    if not all_results:
        print("No data collected. Cannot create charts or PDF.")
        return {}

    # Determine max datasets per chart (use parameter if provided, otherwise use constant)
    max_per_chart = (
        max_datasets_per_chart
        if max_datasets_per_chart is not None
        else MAX_DATASETS_PER_CHART
    )

    # Validate max_per_chart
    if max_per_chart < 1:
        print(
            f"WARNING: max_datasets_per_chart must be at least 1, using 1 instead of {max_per_chart}"
        )
        max_per_chart = 1

    # Calculate overall date range across all results (for consistent chart scaling)
    all_date_ranges = [r.get("date_range") for r in all_results if r.get("date_range")]
    if all_date_ranges:
        start_dates = [datetime.strptime(dr[0], "%Y-%m") for dr in all_date_ranges]
        end_dates = [datetime.strptime(dr[1], "%Y-%m") for dr in all_date_ranges]
        overall_date_range = (
            min(start_dates).strftime("%Y-%m"),
            max(end_dates).strftime("%Y-%m"),
        )
    else:
        overall_date_range = None

    # Split results into chunks if we have more than max_per_chart
    num_results = len(all_results)
    chart_chunks = []

    if num_results <= max_per_chart:
        # Single chart with all results
        chart_chunks = [all_results]
    else:
        # Split into multiple charts
        num_charts = (num_results + max_per_chart - 1) // max_per_chart
        debug_print(
            debug,
            f"Creating {num_charts} charts (max {max_per_chart} datasets per chart)",
        )
        for i in range(0, num_results, max_per_chart):
            chunk = all_results[i : i + max_per_chart]
            chart_chunks.append(chunk)

    # Generate charts for each chunk with progress bar
    chart_info = []  # List of (chart_filename, chunk_index) tuples
    with tqdm(total=len(chart_chunks), desc="Generating charts") as pbar:
        for i, chunk in enumerate(chart_chunks):
            # Generate filename for this chart
            if len(chart_chunks) > 1:
                workspace_projects = [
                    r.get("workspace_project", "Unknown") for r in chunk
                ]
                filename_parts = [wp.replace("/", "_") for wp in workspace_projects]
                chunk_filename = f"experiment_usage_report_chart_{i+1}_of_{len(chart_chunks)}_{'_'.join(filename_parts[:2])}.png"
                if len(filename_parts) > 2:
                    chunk_filename = chunk_filename.replace(
                        ".png", f"_and_{len(filename_parts)-2}_more.png"
                    )
            else:
                chunk_filename = None  # Use default naming

            # Create chart with consistent date range
            chart_filename = create_combined_chart_from_data(
                chunk,
                png_filename=chunk_filename,
                date_range=overall_date_range,
                debug=debug,
            )
            if chart_filename:
                chart_info.append(
                    (chart_filename, i)
                )  # Store filename with its chunk index
            pbar.update(1)

    if not chart_info:
        print("No charts generated. Cannot create PDF.")
        return {}

    # Generate PDF filename from all workspace/projects
    filename_parts = [wp.replace("/", "_") for wp in workspace_projects]
    pdf_filename = f"usage_report_{'_'.join(filename_parts)}.pdf"

    # Create PDF using ReportLab
    try:
        # Generate timestamp for footer
        timestamp = datetime.now().strftime("Generated: %Y-%m-%d %H:%M:%S")

        # Create document with margins (leave extra space at bottom for footer)
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
            # Position footer at bottom of page, accounting for margin
            footer_y = 0.5 * inch
            canvas.drawCentredString(
                letter[0] / 2.0, footer_y, timestamp  # Center of page width
            )
            canvas.restoreState()

        # Assign footer callbacks
        doc.onFirstPage = draw_footer
        doc.onLaterPages = draw_footer

        # Build list of flowables (content elements)
        flowables = []

        # Add statistics content first
        add_statistics_to_flowables(
            flowables, all_results, workspace_projects, debug=debug
        )

        # Add charts (with page breaks between them if multiple)
        for chart_idx, (chart_filename, chunk_idx) in enumerate(chart_info):
            # Add page break before each chart (maintaining same behavior as original code)
            flowables.append(PageBreak())

            # Determine workspace/projects for this chart using the chunk index
            chunk = chart_chunks[chunk_idx]
            if len(chart_info) > 1:
                workspace_projects_str = ", ".join(
                    [r.get("workspace_project", "Unknown") for r in chunk]
                )
                workspace_projects_str = f"Chart {chart_idx+1} of {len(chart_info)}: {workspace_projects_str}"
            elif len(chart_chunks) > 1:
                # Multiple chunks but only one chart generated (edge case)
                workspace_projects_str = ", ".join(
                    [r.get("workspace_project", "Unknown") for r in chunk]
                )
            else:
                workspace_projects_str = ", ".join(
                    [r.get("workspace_project", "Unknown") for r in all_results]
                )

            add_chart_to_flowables(
                flowables,
                chart_filename,
                workspace_projects_str,
                chart_num=chart_idx + 1,
                total_charts=len(chart_info),
                debug=debug,
            )

        # Build the PDF
        debug_print(debug, "Building PDF...")
        doc.build(flowables)
        print(f"\nPDF report generated successfully: {pdf_filename}")
    except Exception as e:
        print(f"Error generating PDF: {e}")
        import traceback

        traceback.print_exc()
        return {}

    # Open PDF file unless --no-open flag is set
    if not no_open:
        open_pdf(pdf_filename, debug=debug)

    return {
        "pdf_file": pdf_filename,
        "results": all_results,
    }
