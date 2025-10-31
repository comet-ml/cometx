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

Generate usage reports with experiment counts by time unit (month, week, day, or hour) for one or more workspaces/projects.
Multiple workspaces/projects are combined into a single chart with a legend.

Examples:
    cometx admin usage-report WORKSPACE
    cometx admin usage-report WORKSPACE WORKSPACE
    cometx admin usage-report WORKSPACE/PROJECT WORKSPACE/PROJECT
    cometx admin usage-report WORKSPACE WORKSPACE/PROJECT
    cometx admin usage-report WORKSPACE --units week
    cometx admin usage-report WORKSPACE --units day --max-experiments-per-chart 10
    cometx admin usage-report WORKSPACE/PROJECT --no-open

Options:
    WORKSPACE_PROJECT (required, one or more)
        One or more WORKSPACE or WORKSPACE/PROJECT to run usage report for.
        If WORKSPACE is provided without a project, all projects in that workspace will be included.

    --units {month,week,day,hour}
        Time unit for grouping experiments (default: month).
        - month: Group by month (YYYY-MM format)
        - week: Group by ISO week (YYYY-WW format)
        - day: Group by day (YYYY-MM-DD format)
        - hour: Group by hour (YYYY-MM-DD-HH format)

    --max-experiments-per-chart N
        Maximum number of workspaces/projects per chart (default: 100).
        If more workspaces/projects are provided, multiple charts will be generated.

    --no-open
        Don't automatically open the generated PDF file after generation.

    --debug
        Enable debug output for troubleshooting.

    --api-key KEY
        Set the COMET_API_KEY for authentication.

    --url-override URL
        Set the COMET_URL_OVERRIDE for custom Comet server.

    --host URL
        Override the HOST URL.

Output:
    Generates a PDF report containing:
    - Summary statistics (total experiments, users, run times, GPU utilization)
    - Experiment count charts by time unit
    - GPU utilization charts (if GPU data is available)
    - GPU memory utilization charts (if GPU data is available)

"""

import json
import os
import warnings
import webbrowser
from collections import defaultdict
from datetime import datetime, timedelta
from urllib.parse import urlparse
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
from tqdm import tqdm

# Suppress matplotlib warnings about non-GUI backend
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Maximum number of datasets (workspaces/projects) per chart
# If more datasets are provided, multiple charts will be generated
MAX_DATASETS_PER_CHART = 100


def debug_print(debug, *args, **kwargs):
    """Print only if debug is True"""
    if debug:
        print(*args, **kwargs)


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


def format_time_key(dt, unit):
    """
    Format a datetime object as a time key based on the specified unit.

    Args:
        dt: datetime object
        unit: One of "month", "week", "day", "hour"

    Returns:
        str: Formatted time key
    """
    if unit == "month":
        return dt.strftime("%Y-%m")
    elif unit == "week":
        # ISO week format: YYYY-WW
        year, week, _ = dt.isocalendar()
        return f"{year}-W{week:02d}"
    elif unit == "day":
        return dt.strftime("%Y-%m-%d")
    elif unit == "hour":
        return dt.strftime("%Y-%m-%d-%H")
    else:
        raise ValueError(f"Unknown unit: {unit}")


def parse_time_key(time_key, unit):
    """
    Parse a time key string back to a datetime object.

    Args:
        time_key: Time key string (e.g., "2024-01", "2024-W01", "2024-01-01", "2024-01-01-12")
        unit: One of "month", "week", "day", "hour"

    Returns:
        datetime: Parsed datetime object
    """
    if unit == "month":
        return datetime.strptime(time_key, "%Y-%m")
    elif unit == "week":
        # Parse ISO week format: YYYY-WW
        year_str, week_str = time_key.split("-W")
        year = int(year_str)
        week = int(week_str)
        # Create datetime for January 4th of the year (which is always in week 1)
        jan4 = datetime(year, 1, 4)
        # Get the Monday of week 1
        days_since_monday = jan4.weekday()
        week1_monday = jan4 - timedelta(days=days_since_monday)
        # Add weeks to get to the target week
        target_monday = week1_monday + timedelta(weeks=(week - 1))
        return target_monday
    elif unit == "day":
        return datetime.strptime(time_key, "%Y-%m-%d")
    elif unit == "hour":
        return datetime.strptime(time_key, "%Y-%m-%d-%H")
    else:
        raise ValueError(f"Unknown unit: {unit}")


def get_next_time_key(time_key, unit):
    """
    Get the next time key after the given one.

    Args:
        time_key: Current time key string
        unit: One of "month", "week", "day", "hour"

    Returns:
        str: Next time key
    """
    dt = parse_time_key(time_key, unit)
    if unit == "month":
        if dt.month == 12:
            next_dt = dt.replace(year=dt.year + 1, month=1)
        else:
            next_dt = dt.replace(month=dt.month + 1)
    elif unit == "week":
        next_dt = dt + timedelta(weeks=1)
    elif unit == "day":
        next_dt = dt + timedelta(days=1)
    elif unit == "hour":
        next_dt = dt + timedelta(hours=1)
    else:
        raise ValueError(f"Unknown unit: {unit}")
    return format_time_key(next_dt, unit)


def get_unit_label(unit):
    """
    Get a human-readable label for a time unit.

    Args:
        unit: One of "month", "week", "day", "hour"

    Returns:
        str: Label (e.g., "Month", "Week", "Day", "Hour")
    """
    labels = {
        "month": "Month",
        "week": "Week",
        "day": "Day",
        "hour": "Hour",
    }
    return labels.get(unit, unit.capitalize())


def get_unit_label_plural(unit):
    """
    Get a human-readable plural label for a time unit.

    Args:
        unit: One of "month", "week", "day", "hour"

    Returns:
        str: Plural label (e.g., "Months", "Weeks", "Days", "Hours")
    """
    labels = {
        "month": "Months",
        "week": "Weeks",
        "day": "Days",
        "hour": "Hours",
    }
    return labels.get(unit, unit.capitalize() + "s")


def add_chart_to_flowables(
    flowables,
    png_file,
    workspace_project=None,
    chart_num=None,
    total_charts=None,
    chart_type="experiments",
    time_unit="month",
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
        chart_type: Type of chart - "experiments", "gpu", or "memory"
        time_unit: Time unit for the chart ("month", "week", "day", or "hour")
        debug: If True, print debug information
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
            # Get unit label for titles
            unit_label = get_unit_label(time_unit)
            # Build title with chart number if available
            if chart_type == "experiments":
                if (
                    chart_num is not None
                    and total_charts is not None
                    and total_charts > 1
                ):
                    title = (
                        f"Experiments by {unit_label} ({chart_num} of {total_charts})"
                    )
                else:
                    title = f"Experiments by {unit_label}"
            elif chart_type == "gpu":
                if (
                    chart_num is not None
                    and total_charts is not None
                    and total_charts > 1
                ):
                    title = f"GPU Utilization by {unit_label} ({chart_num} of {total_charts})"
                else:
                    title = f"GPU Utilization by {unit_label}"
            elif chart_type == "memory":
                if (
                    chart_num is not None
                    and total_charts is not None
                    and total_charts > 1
                ):
                    title = f"GPU Memory Utilization by {unit_label} ({chart_num} of {total_charts})"
                else:
                    title = f"GPU Memory Utilization by {unit_label}"
            else:
                title = workspace_project
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


def generate_experiment_chart(api, workspace, project, units="month", debug=False):
    """
    Collect experiment data for a workspace/project.

    Args:
        api: Comet API instance
        workspace: Workspace name
        project: Project name
        units: Time unit for grouping ("month", "week", "day", or "hour")
        debug: If True, print debug information

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

    # Get GPU data: collect utilization metrics for each GPU separately
    column_data = api._client.get_project_columns(workspace, project)
    experiment_keys = [exp["experimentKey"] for exp in experiments]

    # Find GPU metric names (memory and utilization)
    # Format: sys.gpu.0.gpu_utilization, sys.gpu.0.memory_utilization, sys.gpu.1.gpu_utilization, etc.
    gpu_metric_names = []
    if column_data and "columns" in column_data:
        gpu_metric_names = [
            item["name"]
            for item in column_data["columns"]
            if item["name"].startswith("sys.gpu")
            and (
                item["name"].endswith(".gpu_utilization")
                or item["name"].endswith(".memory_utilization")
            )
        ]

    # Dictionary to store max GPU utilization per experiment per GPU
    # Key: experiment_key, Value: {gpu_number: {"gpu_utilization": max, "memory_utilization": max}}
    experiment_gpu_data = {}

    # Helper function to extract GPU number from metric name
    # e.g., "sys.gpu.0.gpu_utilization" -> 0
    def extract_gpu_number(metric_name):
        """Extract GPU number from metric name like sys.gpu.0.gpu_utilization"""
        try:
            # Split by '.' and find the number after "gpu"
            parts = metric_name.split(".")
            if "gpu" in parts:
                gpu_idx = parts.index("gpu")
                if gpu_idx + 1 < len(parts):
                    return int(parts[gpu_idx + 1])
        except (ValueError, IndexError):
            pass
        return None

    # Track total GPU duration (in milliseconds) for all gpu_utilization metrics
    total_gpu_duration_ms = 0

    if gpu_metric_names and experiment_keys:
        try:
            metric_data = api.get_metrics_for_chart(experiment_keys, gpu_metric_names)

            # Debug: inspect the structure of metric_data
            if debug and metric_data:
                first_key = list(metric_data.keys())[0] if metric_data else None
                if first_key:
                    debug_print(debug, f"Sample experiment key structure: {first_key}")
                    debug_print(
                        debug,
                        f"Sample experiment data keys: {list(metric_data[first_key].keys())}",
                    )
                    if (
                        "metrics" in metric_data[first_key]
                        and metric_data[first_key]["metrics"]
                    ):
                        sample_metric = metric_data[first_key]["metrics"][0]
                        debug_print(
                            debug,
                            f"Sample metric structure keys: {list(sample_metric.keys())}",
                        )

            for experiment_key in metric_data:
                if experiment_key not in experiment_gpu_data:
                    experiment_gpu_data[experiment_key] = {}

                # Check if duration exists at experiment level (alternative structure)
                experiment_duration = metric_data[experiment_key].get("duration", None)
                if debug and experiment_duration is not None:
                    debug_print(
                        debug,
                        f"Found experiment-level duration for {experiment_key}: {experiment_duration}",
                    )

                for metric in metric_data[experiment_key].get("metrics", []):
                    metric_name = metric["metricName"]
                    values = metric.get("values", [])

                    # Extract duration from durations array
                    # durations array contains duration between timesteps, so sum them all
                    # Also check for single duration field as fallback
                    durations = metric.get("durations", [])
                    if durations:
                        # Sum all durations to get total duration (durations are intervals between timesteps)
                        duration = sum(durations) if durations else 0
                    else:
                        # Fallback to single duration field
                        duration = metric.get("duration", 0)
                        # Also check alternative field names
                        if duration == 0:
                            duration = metric.get("durationMs", 0)
                        if duration == 0:
                            duration = metric.get("duration_ms", 0)
                        if duration == 0:
                            duration = metric.get("time", 0)

                    # Debug: log gpu_utilization metrics and their duration
                    if metric_name.endswith(".gpu_utilization") and debug:
                        durations_preview = (
                            f"[{durations[0]}, {durations[1]}, ... {durations[-1]}]"
                            if len(durations) > 2
                            else str(durations)
                        )
                        debug_print(
                            debug,
                            f"GPU metric {metric_name}: duration={duration} ms (sum of {len(durations)} intervals from durations array: {durations_preview}), has_values={bool(values)}",
                        )

                    if values:
                        max_value = max(values)
                        gpu_num = extract_gpu_number(metric_name)

                        if gpu_num is not None:
                            if gpu_num not in experiment_gpu_data[experiment_key]:
                                experiment_gpu_data[experiment_key][gpu_num] = {
                                    "gpu_utilization": None,
                                    "memory_utilization": None,
                                }

                            if metric_name.endswith(".gpu_utilization"):
                                # GPU utilization
                                if (
                                    experiment_gpu_data[experiment_key][gpu_num][
                                        "gpu_utilization"
                                    ]
                                    is None
                                ):
                                    experiment_gpu_data[experiment_key][gpu_num][
                                        "gpu_utilization"
                                    ] = max_value
                                else:
                                    experiment_gpu_data[experiment_key][gpu_num][
                                        "gpu_utilization"
                                    ] = max(
                                        experiment_gpu_data[experiment_key][gpu_num][
                                            "gpu_utilization"
                                        ],
                                        max_value,
                                    )
                                # Sum duration for gpu_utilization metrics
                                # duration is the sum of all duration intervals from the durations array
                                if duration is not None and duration > 0:
                                    total_gpu_duration_ms += duration
                                    debug_print(
                                        debug,
                                        f"Added {duration} ms for {metric_name} (total now: {total_gpu_duration_ms} ms)",
                                    )
                                elif debug:
                                    debug_print(
                                        debug,
                                        f"Skipping duration for {metric_name}: duration={duration} (None or <= 0)",
                                    )
                            elif metric_name.endswith(".memory_utilization"):
                                # GPU memory utilization
                                if (
                                    experiment_gpu_data[experiment_key][gpu_num][
                                        "memory_utilization"
                                    ]
                                    is None
                                ):
                                    experiment_gpu_data[experiment_key][gpu_num][
                                        "memory_utilization"
                                    ] = max_value
                                else:
                                    experiment_gpu_data[experiment_key][gpu_num][
                                        "memory_utilization"
                                    ] = max(
                                        experiment_gpu_data[experiment_key][gpu_num][
                                            "memory_utilization"
                                        ],
                                        max_value,
                                    )
        except Exception as e:
            debug_print(debug, f"Error collecting GPU metrics: {e}")
            # Continue without GPU data
            experiment_gpu_data = {}

    # Group experiments by time unit based on startTimeMillis
    time_unit_counts = defaultdict(int)
    # Track GPU utilization by time unit - average across all GPUs per experiment
    # Structure: time_key -> [avg_max_gpu_util_per_exp, ...] or time_key -> [avg_max_memory_util_per_exp, ...]
    time_unit_gpu_utilizations = defaultdict(
        list
    )  # time_key -> [avg_utilizations across all GPUs per experiment]
    time_unit_memory_utilizations = defaultdict(
        list
    )  # time_key -> [avg_utilizations across all GPUs per experiment]
    # Track run times for statistics
    total_run_time_seconds = 0
    run_time_count = 0
    # Track unique users
    unique_users = set()

    for exp in experiments:
        exp_key = exp.get("experimentKey")
        if "startTimeMillis" in exp and exp["startTimeMillis"]:
            # Convert milliseconds to seconds, then to datetime
            start_time = datetime.fromtimestamp(exp["startTimeMillis"] / 1000)
            # Format based on time unit for grouping
            time_key = format_time_key(start_time, units)
            time_unit_counts[time_key] += 1

            # Track GPU utilization for this experiment - average across all GPUs
            if exp_key and exp_key in experiment_gpu_data:
                # Collect all GPU utilizations for this experiment
                gpu_utils = []
                memory_utils = []

                for gpu_num, gpu_data in experiment_gpu_data[exp_key].items():
                    if gpu_data["gpu_utilization"] is not None:
                        gpu_utils.append(gpu_data["gpu_utilization"])
                    if gpu_data["memory_utilization"] is not None:
                        memory_utils.append(gpu_data["memory_utilization"])

                # Calculate average across all GPUs for this experiment
                if gpu_utils:
                    avg_gpu_util = sum(gpu_utils) / len(gpu_utils)
                    time_unit_gpu_utilizations[time_key].append(avg_gpu_util)

                if memory_utils:
                    avg_memory_util = sum(memory_utils) / len(memory_utils)
                    time_unit_memory_utilizations[time_key].append(avg_memory_util)

            # Calculate run time if both start and end times are available
            if "endTimeMillis" in exp and exp["endTimeMillis"]:
                run_time_ms = exp["endTimeMillis"] - exp["startTimeMillis"]
                if run_time_ms > 0:
                    run_time_seconds = run_time_ms / 1000
                    total_run_time_seconds += run_time_seconds
                    run_time_count += 1

        # Track unique users (check all experiments, not just those with startTimeMillis)
        if "userName" in exp and exp["userName"]:
            unique_users.add(exp["userName"])

    if not time_unit_counts:
        debug_print(debug, "No experiments with valid start times found")
        return {}

    # Create a complete range of time units from first to last experiment
    all_time_keys = sorted(time_unit_counts.keys())
    if not all_time_keys:
        debug_print(debug, f"No valid {units} found")
        return {}

    # Generate all time units between first and last
    start_date = parse_time_key(all_time_keys[0], units)
    end_date = parse_time_key(all_time_keys[-1], units)

    complete_time_keys = []
    current_key = all_time_keys[0]
    while True:
        complete_time_keys.append(current_key)
        if current_key == all_time_keys[-1]:
            break
        current_key = get_next_time_key(current_key, units)

    # Fill in counts for all time units (0 for time units with no experiments)
    counts = [time_unit_counts[key] for key in complete_time_keys]

    # Calculate average GPU utilization per time unit
    # For each time unit, average the experiment-level averages (which are already averages across all GPUs)
    avg_gpu_utilizations = []
    avg_memory_utilizations = []

    for time_key in complete_time_keys:
        gpu_utils = time_unit_gpu_utilizations.get(time_key, [])
        memory_utils = time_unit_memory_utilizations.get(time_key, [])

        if gpu_utils:
            # Average of experiment-level averages (each experiment avg is across all its GPUs)
            avg_gpu_utilizations.append(sum(gpu_utils) / len(gpu_utils))
        else:
            avg_gpu_utilizations.append(None)

        if memory_utils:
            avg_memory_utilizations.append(sum(memory_utils) / len(memory_utils))
        else:
            avg_memory_utilizations.append(None)

    # Display summary statistics (only in debug mode)
    unit_label = get_unit_label(unit=units)
    unit_label_plural = get_unit_label_plural(unit=units)
    debug_print(debug, f"\nSummary for {workspace_project}:")
    debug_print(debug, f"Total experiments: {sum(counts)}")
    debug_print(
        debug, f"Date range: {complete_time_keys[0]} to {complete_time_keys[-1]}"
    )
    debug_print(
        debug,
        f"{unit_label_plural} with experiments: {sum(1 for c in counts if c > 0)}",
    )
    debug_print(
        debug,
        f"{unit_label_plural} with zero experiments: {sum(1 for c in counts if c == 0)}",
    )
    debug_print(
        debug,
        f"Average experiments per {unit_label.lower()}: {sum(counts)/len(counts):.1f}",
    )

    # Debug GPU stats
    gpu_time_units_with_data = sum(1 for u in avg_gpu_utilizations if u is not None)
    memory_time_units_with_data = sum(
        1 for u in avg_memory_utilizations if u is not None
    )
    debug_print(
        debug,
        f"{unit_label_plural} with GPU utilization data: {gpu_time_units_with_data}",
    )
    debug_print(
        debug,
        f"{unit_label_plural} with memory utilization data: {memory_time_units_with_data}",
    )
    debug_print(
        debug,
        f"Total GPU duration: {format_gpu_hours(total_gpu_duration_ms)} ({total_gpu_duration_ms} ms)",
    )

    return {
        "total_experiments": sum(counts),
        "monthly_counts": dict(
            time_unit_counts
        ),  # Keep key name for backward compatibility
        "complete_monthly_counts": dict(
            zip(complete_time_keys, counts)
        ),  # Keep key name for backward compatibility
        "complete_months": complete_time_keys,  # Keep key name for backward compatibility
        "time_unit": units,  # Store the time unit used
        "counts": counts,
        "date_range": (complete_time_keys[0], complete_time_keys[-1]),
        "workspace_project": workspace_project,
        "total_run_time_seconds": total_run_time_seconds,
        "run_time_count": run_time_count,
        "experiments": experiments,  # Store raw experiments for statistics
        "unique_users": unique_users,  # Set of unique usernames
        # GPU utilization data - combined across all GPUs
        "monthly_gpu_utilizations": dict(
            time_unit_gpu_utilizations
        ),  # Keep key name for backward compatibility
        "monthly_memory_utilizations": dict(
            time_unit_memory_utilizations
        ),  # Keep key name for backward compatibility
        "avg_gpu_utilizations": avg_gpu_utilizations,  # List aligned with complete_time_keys
        "avg_memory_utilizations": avg_memory_utilizations,  # List aligned with complete_time_keys
        "total_gpu_duration_ms": total_gpu_duration_ms,  # Total GPU duration in milliseconds
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
    complete_time_keys = data.get("complete_months", [])  # Backward compatibility key
    counts = data.get("counts", [])
    time_unit = data.get("time_unit", "month")

    if not complete_time_keys or not counts:
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
        range(len(complete_time_keys)),
        counts,
        color=bar_colors,
        edgecolor="navy",
        alpha=0.7,
        width=0.8,
    )

    # Get unit label for chart
    unit_label = get_unit_label(time_unit)

    # Customize the chart
    plt.title(
        f"Experiment Count by {unit_label} - {workspace_project}",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel(unit_label, fontsize=14)
    plt.ylabel("Number of Experiments", fontsize=14)

    # Set x-axis labels - show every Nth to avoid crowding (adjust based on unit)
    max_labels = 50 if time_unit == "hour" else 30 if time_unit == "day" else 20
    step = max(1, len(complete_time_keys) // max_labels)
    x_ticks = range(0, len(complete_time_keys), step)
    x_labels = [complete_time_keys[i] for i in x_ticks]
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
    data_list, png_filename=None, date_range=None, time_unit="month", debug=False
):
    """
    Create a combined bar chart from multiple experiment datasets with a legend.

    Args:
        data_list: List of dictionaries with experiment data (from generate_experiment_chart)
        png_filename: Optional filename for the PNG. If not provided, generates a default name.
        date_range: Optional tuple (start_key, end_key) as strings in the format for the time_unit.
                    If provided, all datasets will be aligned to this date range.
                    If None, the date range is calculated from the datasets.
        time_unit: Time unit for grouping ("month", "week", "day", or "hour")
        debug: If True, print debug information

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
        # Use provided date range - parse according to time unit
        overall_start = parse_time_key(date_range[0], time_unit)
        overall_end = parse_time_key(date_range[1], time_unit)
    else:
        # Find the complete date range across all datasets
        all_date_ranges = [
            data.get("date_range") for data in data_list if data.get("date_range")
        ]
        if not all_date_ranges:
            debug_print(debug, "No date ranges found in data")
            return None

        # Find the earliest start and latest end date - parse according to time unit
        start_dates = [parse_time_key(dr[0], time_unit) for dr in all_date_ranges]
        end_dates = [parse_time_key(dr[1], time_unit) for dr in all_date_ranges]
        overall_start = min(start_dates)
        overall_end = max(end_dates)

    # Generate complete list of time keys from overall start to end
    all_time_keys = []
    current_key = format_time_key(overall_start, time_unit)
    end_key = format_time_key(overall_end, time_unit)
    while True:
        all_time_keys.append(current_key)
        if current_key == end_key:
            break
        current_key = get_next_time_key(current_key, time_unit)

    if not all_time_keys:
        debug_print(debug, f"No {time_unit}s found in data")
        return None

    # For each dataset, align counts to the complete time key list
    aligned_data = []
    for data in data_list:
        complete_monthly_counts = data.get("complete_monthly_counts", {})
        counts = [complete_monthly_counts.get(key, 0) for key in all_time_keys]
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
    num_time_units = len(all_time_keys)
    bar_width = 0.8 / num_projects if num_projects > 1 else 0.8
    x_positions = range(num_time_units)

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

    # Get unit label for chart
    unit_label = get_unit_label(time_unit)

    # Customize the chart - adjust title based on number of projects
    if len(aligned_data) == 1:
        # Single project - show project name
        project_name = aligned_data[0]["workspace_project"]
        title = f"Experiment Count by {unit_label} - {project_name}"
    else:
        # Multiple projects - show "Combined Projects"
        title = f"Experiment Count by {unit_label} - Combined Projects"

    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel(unit_label, fontsize=14)
    ax.set_ylabel("Number of Experiments", fontsize=14)

    # Set x-axis labels - show every Nth to avoid crowding (adjust based on unit)
    max_labels = 50 if time_unit == "hour" else 30 if time_unit == "day" else 20
    step = max(1, num_time_units // max_labels)
    x_ticks = range(0, num_time_units, step)
    x_labels = [all_time_keys[i] for i in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5, axis="y")

    # Add legend below the chart
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=min(len(aligned_data), 3),  # Arrange in up to 3 columns
        fontsize=10,
        framealpha=0.9,
    )

    # Adjust layout to prevent label cutoff and make room for legend
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    # Save the chart as PNG
    plt.savefig(png_filename, dpi=300, bbox_inches="tight")
    debug_print(debug, f"Combined bar chart saved as: {png_filename}")

    plt.close()

    return png_filename


def create_combined_gpu_chart_from_data(
    data_list,
    png_filename=None,
    date_range=None,
    utilization_type="gpu",
    time_unit="month",
    debug=False,
):
    """
    Create a combined bar chart from multiple experiment datasets showing GPU utilization per GPU.

    Args:
        data_list: List of dictionaries with experiment data (from generate_experiment_chart)
        png_filename: Optional filename for the PNG. If not provided, generates a default name.
        date_range: Optional tuple (start_key, end_key) as strings in the format for the time_unit.
                    If provided, all datasets will be aligned to this date range.
                    If None, the date range is calculated from the datasets.
        utilization_type: "gpu" for GPU utilization or "memory" for memory utilization
        time_unit: Time unit for grouping ("month", "week", "day", or "hour")
        debug: If True, print debug information

    Returns:
        str: The filename of the saved PNG chart, or None if no data
    """
    if not data_list:
        return None

    # Filter out empty data - check for GPU data
    data_list = [
        d
        for d in data_list
        if d
        and d.get("complete_months")
        and d.get(f"avg_{utilization_type}_utilizations")
    ]

    if not data_list:
        debug_print(
            debug, f"No valid GPU {utilization_type} utilization data to create chart"
        )
        return None

    if png_filename is None:
        workspace_projects = [d.get("workspace_project", "Unknown") for d in data_list]
        filename_parts = [wp.replace("/", "_") for wp in workspace_projects]
        util_type_name = "gpu" if utilization_type == "gpu" else "memory"
        png_filename = f"gpu_{util_type_name}_utilization_report_combined_{'_'.join(filename_parts[:3])}.png"
        if len(filename_parts) > 3:
            png_filename = png_filename.replace(
                ".png", f"_and_{len(filename_parts)-3}_more.png"
            )

    # Determine the date range to use
    if date_range:
        # Use provided date range - parse according to time unit
        overall_start = parse_time_key(date_range[0], time_unit)
        overall_end = parse_time_key(date_range[1], time_unit)
    else:
        # Find the complete date range across all datasets
        all_date_ranges = [
            data.get("date_range") for data in data_list if data.get("date_range")
        ]
        if not all_date_ranges:
            debug_print(debug, "No date ranges found in data")
            return None

        # Find the earliest start and latest end date - parse according to time unit
        start_dates = [parse_time_key(dr[0], time_unit) for dr in all_date_ranges]
        end_dates = [parse_time_key(dr[1], time_unit) for dr in all_date_ranges]
        overall_start = min(start_dates)
        overall_end = max(end_dates)

    # Generate complete list of time keys from overall start to end
    all_time_keys = []
    current_key = format_time_key(overall_start, time_unit)
    end_key = format_time_key(overall_end, time_unit)
    while True:
        all_time_keys.append(current_key)
        if current_key == end_key:
            break
        current_key = get_next_time_key(current_key, time_unit)

    if not all_time_keys:
        debug_print(debug, f"No {time_unit}s found in data")
        return None

    # For each dataset, align utilization data to the complete time key list
    aligned_data = []
    for data in data_list:
        complete_time_keys = data.get("complete_months", [])
        utilizations = data.get(f"avg_{utilization_type}_utilizations", [])

        # Create a mapping from time key to utilization
        time_key_to_util = dict(zip(complete_time_keys, utilizations))

        # Align to all_time_keys
        aligned_utils = [time_key_to_util.get(key, None) for key in all_time_keys]

        aligned_data.append(
            {
                "workspace_project": data.get("workspace_project", "Unknown"),
                "utilizations": aligned_utils,
            }
        )

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Generate distinct colors for each project
    chart_colors = cm.tab10(range(len(aligned_data)))

    # Set up bar positions for grouped bars
    num_projects = len(aligned_data)
    num_time_units = len(all_time_keys)
    bar_width = 0.8 / num_projects if num_projects > 1 else 0.8
    x_positions = range(num_time_units)

    # Create grouped bars
    bars_list = []
    for i, data in enumerate(aligned_data):
        # Calculate x offset for each group of bars
        offset = (i - (num_projects - 1) / 2) * bar_width
        x_pos = [x + offset for x in x_positions]

        # Convert None to 0 for plotting
        plot_values = [float(u) if u is not None else 0.0 for u in data["utilizations"]]

        bars = ax.bar(
            x_pos,
            plot_values,
            width=bar_width,
            label=data["workspace_project"],
            color=chart_colors[i],
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
        )
        bars_list.append(bars)

        # Add value labels on bars (only for non-zero values)
        for bar, value in zip(bars, data["utilizations"]):
            if value is not None and value > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

    # Get unit label for chart
    unit_label = get_unit_label(time_unit)

    # Customize the chart - adjust title based on number of projects
    util_type_display = "GPU" if utilization_type == "gpu" else "Memory"
    if len(aligned_data) == 1:
        # Single project - show project name
        project_name = aligned_data[0]["workspace_project"]
        title = f"Average Max {util_type_display} Utilization by {unit_label} - {project_name}"
    else:
        # Multiple projects - show "Combined Projects"
        title = f"Average Max {util_type_display} Utilization by {unit_label} - Combined Projects"

    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel(unit_label, fontsize=14)
    ax.set_ylabel(f"Average Max {util_type_display} Utilization (%)", fontsize=14)

    # Set x-axis labels - show every Nth to avoid crowding (adjust based on unit)
    max_labels = 50 if time_unit == "hour" else 30 if time_unit == "day" else 20
    step = max(1, num_time_units // max_labels)
    x_ticks = range(0, num_time_units, step)
    x_labels = [all_time_keys[i] for i in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5, axis="y")

    # Set y-axis to show percentages (0-100)
    ax.set_ylim(0, 100)

    # Add legend below the chart
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=min(len(aligned_data), 3),  # Arrange in up to 3 columns
        fontsize=10,
        framealpha=0.9,
    )

    # Adjust layout to prevent label cutoff and make room for legend
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    # Save the chart as PNG
    plt.savefig(png_filename, dpi=300, bbox_inches="tight")
    debug_print(
        debug,
        f"Combined {utilization_type} utilization bar chart saved as: {png_filename}",
    )

    plt.close()

    return png_filename


def format_time(seconds):
    """Format seconds into a human-readable string (hours, minutes, seconds)"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"


def format_gpu_hours(milliseconds):
    """Format milliseconds into GPU hours with appropriate precision"""
    if milliseconds == 0:
        return "0 GPU hours"
    # Convert milliseconds to hours
    hours = milliseconds / (1000 * 3600)
    if hours < 0.01:
        # For very small values, show in minutes
        minutes = milliseconds / (1000 * 60)
        return f"{minutes:.2f} GPU minutes"
    elif hours < 1:
        # For less than an hour, show with 2 decimal places
        return f"{hours:.2f} GPU hours"
    else:
        # For hours or more, show with 1 decimal place
        return f"{hours:.1f} GPU hours"


def add_statistics_to_flowables(
    flowables,
    all_results,
    workspace_projects_input,
    website_name=None,
    time_unit="month",
    debug=False,
):
    """
    Create statistics content and add it to the list of flowables for ReportLab.

    Args:
        flowables: List of ReportLab flowables to append to
        all_results: List of dictionaries with experiment data
        workspace_projects_input: List of workspace/project strings that were requested
        website_name: Optional website name (domain) to display in the title
        debug: If True, print debug information
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

    # Find date range across all results - parse according to time unit
    all_start_dates = []
    all_end_dates = []
    for result in all_results:
        date_range = result.get("date_range")
        if date_range:
            all_start_dates.append(parse_time_key(date_range[0], time_unit))
            all_end_dates.append(parse_time_key(date_range[1], time_unit))

    earliest_date = min(all_start_dates) if all_start_dates else None
    latest_date = max(all_end_dates) if all_end_dates else None

    # Find peak time unit
    combined_time_unit_counts = defaultdict(int)
    for result in all_results:
        time_unit_counts = result.get(
            "monthly_counts", {}
        )  # Backward compatibility key
        for time_key, count in time_unit_counts.items():
            combined_time_unit_counts[time_key] += count

    peak_time_unit = (
        max(combined_time_unit_counts.items(), key=lambda x: x[1])
        if combined_time_unit_counts
        else None
    )

    # Calculate average run time
    avg_run_time_seconds = (
        total_run_time_seconds / run_time_count if run_time_count > 0 else 0
    )

    # Calculate GPU utilization statistics
    all_gpu_utilizations = []
    all_memory_utilizations = []
    for result in all_results:
        gpu_utils = result.get("avg_gpu_utilizations", [])
        memory_utils = result.get("avg_memory_utilizations", [])
        # Collect all non-None values
        all_gpu_utilizations.extend([u for u in gpu_utils if u is not None])
        all_memory_utilizations.extend([u for u in memory_utils if u is not None])

    avg_gpu_utilization = (
        sum(all_gpu_utilizations) / len(all_gpu_utilizations)
        if all_gpu_utilizations
        else None
    )
    avg_memory_utilization = (
        sum(all_memory_utilizations) / len(all_memory_utilizations)
        if all_memory_utilizations
        else None
    )

    # Calculate total unique users across all results
    all_unique_users = set()
    for result in all_results:
        unique_users = result.get("unique_users", set())
        if unique_users:
            all_unique_users.update(unique_users)
    total_unique_users = len(all_unique_users)

    # Calculate total GPU hours across all results
    total_gpu_duration_ms = sum(
        result.get("total_gpu_duration_ms", 0) for result in all_results
    )

    # Title - display on two separate centered lines
    flowables.append(
        Paragraph(escape("Usage Report - Summary Statistics"), title_style)
    )
    if website_name:
        flowables.append(Paragraph(escape(website_name), title_style))
    flowables.append(Spacer(1, 0.3 * inch))

    # Overall Statistics section
    flowables.append(Paragraph("Overall Statistics", heading_style))

    flowables.append(
        Paragraph(
            escape(f"Total Number of Experiments: {total_experiments:,}"), body_style
        )
    )

    flowables.append(
        Paragraph(escape(f"Total Number of Users: {total_unique_users:,}"), body_style)
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
        # Format date range according to time unit
        start_key = format_time_key(earliest_date, time_unit)
        end_key = format_time_key(latest_date, time_unit)
        unit_label_plural = get_unit_label_plural(time_unit)
        flowables.append(
            Paragraph(
                escape(f"Date Range: {start_key} to {end_key}"),
                body_style,
            )
        )
        # Calculate span in time units
        if time_unit == "month":
            span = (
                (latest_date.year - earliest_date.year) * 12
                + (latest_date.month - earliest_date.month)
                + 1
            )
        elif time_unit == "week":
            span = int((latest_date - earliest_date).days / 7) + 1
        elif time_unit == "day":
            span = (latest_date - earliest_date).days + 1
        elif time_unit == "hour":
            span = int((latest_date - earliest_date).total_seconds() / 3600) + 1
        else:
            span = 0
        flowables.append(
            Paragraph(
                escape(f"Time Span: {span} {unit_label_plural.lower()}"), body_style
            )
        )

    flowables.append(
        Paragraph(
            escape(f"Number of Workspaces/Projects: {len(all_results)}"), body_style
        )
    )

    if peak_time_unit:
        unit_label = get_unit_label(time_unit)
        flowables.append(
            Paragraph(
                escape(
                    f"Peak {unit_label}: {peak_time_unit[0]} ({peak_time_unit[1]:,} experiments)"
                ),
                body_style,
            )
        )

    # Add GPU utilization statistics
    if avg_gpu_utilization is not None:
        flowables.append(
            Paragraph(
                escape(f"Average GPU Utilization: {avg_gpu_utilization:.1f}%"),
                body_style,
            )
        )

    if avg_memory_utilization is not None:
        flowables.append(
            Paragraph(
                escape(
                    f"Average GPU Memory Utilization: {avg_memory_utilization:.1f}%"
                ),
                body_style,
            )
        )

    # Add total GPU hours (always show, even if 0)
    flowables.append(
        Paragraph(
            escape(f"Total GPU Hours: {format_gpu_hours(total_gpu_duration_ms)}"),
            body_style,
        )
    )

    flowables.append(Spacer(1, 0.2 * inch))

    # Breakdown section - only show if there is more than one project
    if len(all_results) > 1:
        flowables.append(Paragraph("Breakdown by Workspace/Project", heading_style))

        for i, result in enumerate(all_results):
            wp = result.get("workspace_project", "Unknown")
            exp_count = result.get("total_experiments", 0)
            run_time = result.get("total_run_time_seconds", 0)
            run_time_exp_count = result.get("run_time_count", 0)

            # Calculate GPU utilization statistics for this workspace/project
            gpu_utils = result.get("avg_gpu_utilizations", [])
            memory_utils = result.get("avg_memory_utilizations", [])
            # Collect all non-None values
            gpu_values = [u for u in gpu_utils if u is not None]
            memory_values = [u for u in memory_utils if u is not None]

            avg_gpu_util = sum(gpu_values) / len(gpu_values) if gpu_values else None
            avg_memory_util = (
                sum(memory_values) / len(memory_values) if memory_values else None
            )

            # Get GPU hours for this workspace/project
            gpu_duration_ms = result.get("total_gpu_duration_ms", 0)

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

            # Add GPU utilization statistics if available
            if avg_gpu_util is not None:
                workspace_content.append(
                    Paragraph(
                        escape(f"Average GPU Utilization: {avg_gpu_util:.1f}%"),
                        indent_body_style,
                    )
                )

            if avg_memory_util is not None:
                workspace_content.append(
                    Paragraph(
                        escape(
                            f"Average GPU Memory Utilization: {avg_memory_util:.1f}%"
                        ),
                        indent_body_style,
                    )
                )

            # Add GPU hours (always show, even if 0)
            workspace_content.append(
                Paragraph(
                    escape(f"Total GPU Hours: {format_gpu_hours(gpu_duration_ms)}"),
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
    api,
    workspace_projects,
    no_open=False,
    max_datasets_per_chart=None,
    units="month",
    debug=False,
):
    """
    Get usage report for one or more workspace/project combinations.

    Args:
        api: Comet API instance
        workspace_projects: String or list of strings in format "workspace" or "workspace/project"
        no_open: If True, don't automatically open the generated PDF file
        max_datasets_per_chart: Maximum number of datasets per chart. If None, uses MAX_DATASETS_PER_CHART constant.
        units: Time unit for grouping experiments ("month", "week", "day", or "hour"). Default is "month".
        debug: If True, print debug information

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
                    api, workspace, project, units=units, debug=debug
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
        start_dates = [parse_time_key(dr[0], units) for dr in all_date_ranges]
        end_dates = [parse_time_key(dr[1], units) for dr in all_date_ranges]
        overall_date_range = (
            format_time_key(min(start_dates), units),
            format_time_key(max(end_dates), units),
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
                time_unit=units,
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

        # Extract website name from API base_url
        try:
            base_url = api._client.base_url
            website_name = extract_website_name(base_url)
        except Exception:
            website_name = None

        # Add statistics content first
        add_statistics_to_flowables(
            flowables,
            all_results,
            workspace_projects,
            website_name=website_name,
            time_unit=units,
            debug=debug,
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
                chart_type="experiments",
                time_unit=units,
                debug=debug,
            )

        # Generate GPU utilization charts if data is available
        # Check if any results have GPU data
        has_gpu_data = any(
            r.get("avg_gpu_utilizations")
            and any(u is not None for u in r.get("avg_gpu_utilizations", []))
            for r in all_results
        )
        has_memory_data = any(
            r.get("avg_memory_utilizations")
            and any(u is not None for u in r.get("avg_memory_utilizations", []))
            for r in all_results
        )

        # Generate GPU utilization charts
        if has_gpu_data:
            # Generate GPU utilization charts for each chunk (same chunks as experiment charts)
            gpu_chart_info = []
            with tqdm(
                total=len(chart_chunks), desc="Generating GPU utilization charts"
            ) as pbar:
                for i, chunk in enumerate(chart_chunks):
                    # Generate filename for this chart
                    if len(chart_chunks) > 1:
                        workspace_projects_chunk = [
                            r.get("workspace_project", "Unknown") for r in chunk
                        ]
                        filename_parts = [
                            wp.replace("/", "_") for wp in workspace_projects_chunk
                        ]
                        chunk_filename = f"gpu_utilization_chart_{i+1}_of_{len(chart_chunks)}_{'_'.join(filename_parts[:2])}.png"
                        if len(filename_parts) > 2:
                            chunk_filename = chunk_filename.replace(
                                ".png", f"_and_{len(filename_parts)-2}_more.png"
                            )
                    else:
                        chunk_filename = None  # Use default naming

                    # Create GPU utilization chart with consistent date range
                    gpu_chart_filename = create_combined_gpu_chart_from_data(
                        chunk,
                        png_filename=chunk_filename,
                        date_range=overall_date_range,
                        utilization_type="gpu",
                        time_unit=units,
                        debug=debug,
                    )
                    if gpu_chart_filename:
                        gpu_chart_info.append((gpu_chart_filename, i))
                    pbar.update(1)

            # Add GPU utilization charts to PDF
            for chart_idx, (chart_filename, chunk_idx) in enumerate(gpu_chart_info):
                flowables.append(PageBreak())
                chunk = chart_chunks[chunk_idx]
                if len(gpu_chart_info) > 1:
                    workspace_projects_str = ", ".join(
                        [r.get("workspace_project", "Unknown") for r in chunk]
                    )
                    workspace_projects_str = f"Chart {chart_idx+1} of {len(gpu_chart_info)}: {workspace_projects_str}"
                else:
                    workspace_projects_str = ", ".join(
                        [r.get("workspace_project", "Unknown") for r in chunk]
                    )

                add_chart_to_flowables(
                    flowables,
                    chart_filename,
                    workspace_projects_str,
                    chart_num=chart_idx + 1,
                    total_charts=len(gpu_chart_info),
                    chart_type="gpu",
                    time_unit=units,
                    debug=debug,
                )

        # Generate GPU memory utilization charts
        if has_memory_data:
            # Generate memory utilization charts for each chunk
            memory_chart_info = []
            with tqdm(
                total=len(chart_chunks), desc="Generating GPU memory utilization charts"
            ) as pbar:
                for i, chunk in enumerate(chart_chunks):
                    # Generate filename for this chart
                    if len(chart_chunks) > 1:
                        workspace_projects_chunk = [
                            r.get("workspace_project", "Unknown") for r in chunk
                        ]
                        filename_parts = [
                            wp.replace("/", "_") for wp in workspace_projects_chunk
                        ]
                        chunk_filename = f"gpu_memory_utilization_chart_{i+1}_of_{len(chart_chunks)}_{'_'.join(filename_parts[:2])}.png"
                        if len(filename_parts) > 2:
                            chunk_filename = chunk_filename.replace(
                                ".png", f"_and_{len(filename_parts)-2}_more.png"
                            )
                    else:
                        chunk_filename = None  # Use default naming

                    # Create memory utilization chart with consistent date range
                    memory_chart_filename = create_combined_gpu_chart_from_data(
                        chunk,
                        png_filename=chunk_filename,
                        date_range=overall_date_range,
                        utilization_type="memory",
                        time_unit=units,
                        debug=debug,
                    )
                    if memory_chart_filename:
                        memory_chart_info.append((memory_chart_filename, i))
                    pbar.update(1)

            # Add memory utilization charts to PDF
            for chart_idx, (chart_filename, chunk_idx) in enumerate(memory_chart_info):
                flowables.append(PageBreak())
                chunk = chart_chunks[chunk_idx]
                if len(memory_chart_info) > 1:
                    workspace_projects_str = ", ".join(
                        [r.get("workspace_project", "Unknown") for r in chunk]
                    )
                    workspace_projects_str = f"Chart {chart_idx+1} of {len(memory_chart_info)}: {workspace_projects_str}"
                else:
                    workspace_projects_str = ", ".join(
                        [r.get("workspace_project", "Unknown") for r in chunk]
                    )

                add_chart_to_flowables(
                    flowables,
                    chart_filename,
                    workspace_projects_str,
                    chart_num=chart_idx + 1,
                    total_charts=len(memory_chart_info),
                    chart_type="memory",
                    time_unit=units,
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
