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

cometx admin usage-report WORKSPACE/PROJECT

"""

import warnings
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
from comet_ml import API

# Suppress matplotlib warnings about non-GUI backend
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


def open_pdf(pdf_path):
    """Open PDF file using the default system application"""
    import os
    import webbrowser

    if os.path.exists(pdf_path):
        webbrowser.open(f"file://{os.path.abspath(pdf_path)}")
        print(f"Opening PDF: {pdf_path}")
    else:
        print(f"PDF file not found: {pdf_path}")


def generate_usage_report(api, workspace_project, no_open=False):
    """
    Get usage report for a specific workspace/project.

    Args:
        api: Comet API instance
        workspace_project: String in format "workspace/project"
        no_open: If True, don't automatically open the generated PDF file

    Returns:
        dict: The usage report data
    """
    if not workspace_project:
        print("ERROR: workspace_project is required")
        return {}

    if "/" in workspace_project:
        workspace, project = workspace_project.split("/")
    else:
        workspace = workspace_project
        project = None

    api = API()
    # Get project experiments data
    print(f"Fetching experiments for workspace: {workspace}, project: {project}")
    project_data = api._client.get_project_experiments(workspace, project)

    if not project_data:
        print(f"API returned None for {workspace_project}")
        return {}

    if not isinstance(project_data, dict):
        print(
            f"API returned unexpected data type: {type(project_data)} for {workspace_project}"
        )
        return {}

    if "experiments" not in project_data:
        print(f"No 'experiments' key in API response for {workspace_project}")
        print(
            f"Available keys: {list(project_data.keys()) if project_data else 'None'}"
        )
        return {}

    experiments = project_data["experiments"]
    print(f"Found {len(experiments)} experiments for {workspace_project}")

    # Group experiments by month/year based on startTimeMillis
    monthly_counts = defaultdict(int)

    for exp in experiments:
        if "startTimeMillis" in exp and exp["startTimeMillis"]:
            # Convert milliseconds to seconds, then to datetime
            start_time = datetime.fromtimestamp(exp["startTimeMillis"] / 1000)
            # Format as YYYY-MM for grouping
            month_key = start_time.strftime("%Y-%m")
            monthly_counts[month_key] += 1

    if not monthly_counts:
        print("No experiments with valid start times found")
        return {}

    # Create a complete range of months from first to last experiment
    all_months = sorted(monthly_counts.keys())
    if not all_months:
        print("No valid months found")
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
    png_filename = f"experiment_usage_report_{workspace_project.replace('/', '_')}.png"
    plt.savefig(png_filename, dpi=300, bbox_inches="tight")

    # Save the chart as PDF
    pdf_filename = f"experiment_usage_report_{workspace_project.replace('/', '_')}.pdf"
    plt.savefig(pdf_filename, format="pdf", bbox_inches="tight")

    # Display summary statistics
    print(f"\nSummary for {workspace_project}:")
    print(f"Total experiments: {sum(counts)}")
    print(f"Date range: {complete_months[0]} to {complete_months[-1]}")
    print(f"Months with experiments: {sum(1 for c in counts if c > 0)}")
    print(f"Months with zero experiments: {sum(1 for c in counts if c == 0)}")
    print(f"Average experiments per month: {sum(counts)/len(counts):.1f}")

    # Show file save message after summary
    print(f"Bar chart saved as: {pdf_filename}")

    # Open PDF file unless --no-open flag is set
    if not no_open:
        open_pdf(pdf_filename)

    return {
        "total_experiments": sum(counts),
        "monthly_counts": dict(monthly_counts),
        "complete_monthly_counts": dict(zip(complete_months, counts)),
        "date_range": (complete_months[0], complete_months[-1]),
        "png_file": png_filename,
        "pdf_file": pdf_filename,
    }
