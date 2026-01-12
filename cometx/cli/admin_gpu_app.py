#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streamlit app for GPU report visualization.

This app provides a web-based interface for the admin GPU report functionality,
allowing users to visualize GPU metrics with different aggregations and time units.
"""

import datetime
import json
import os
import tempfile
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import streamlit as st

from cometx.cli.admin_gpu_report import extract_metric_value
from cometx.cli.admin_utils import get_distinct_colors


def load_json_data(json_file_path: str) -> Dict:
    """Load GPU report data from JSON file."""
    try:
        with open(json_file_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"Error loading JSON file: {e}")
        return {}


def parse_time_key(time_key: str, time_unit: str) -> datetime.datetime:
    """Parse a time key string into a datetime object based on time unit."""
    try:
        if time_unit == "year":
            return datetime.datetime.strptime(time_key, "%Y")
        elif time_unit == "month":
            return datetime.datetime.strptime(time_key, "%Y-%m")
        elif time_unit == "week":
            # Format: YYYY-WW
            if "-W" in time_key:
                year, week = time_key.split("-W")
                # Use ISO week format
                year = int(year)
                week = int(week)
                # Get the first day of the ISO week
                jan4 = datetime.datetime(year, 1, 4)
                jan4_weekday = jan4.weekday()  # Monday is 0
                days_since_monday = (week - 1) * 7
                first_day = (
                    jan4
                    - datetime.timedelta(days=jan4_weekday)
                    + datetime.timedelta(days=days_since_monday)
                )
                return first_day
            else:
                return datetime.datetime.strptime(time_key, "%Y-%m")
        elif time_unit == "day":
            return datetime.datetime.strptime(time_key, "%Y-%m-%d")
        else:
            return datetime.datetime.strptime(time_key, "%Y-%m")
    except Exception:
        # Fallback to month parsing
        try:
            return datetime.datetime.strptime(time_key, "%Y-%m")
        except Exception:
            return None


def format_time_key(dt_obj: datetime.datetime, time_unit: str) -> str:
    """Format a datetime object into a time key string based on time unit."""
    if dt_obj is None:
        return ""
    if time_unit == "year":
        return dt_obj.strftime("%Y")
    elif time_unit == "month":
        return dt_obj.strftime("%Y-%m")
    elif time_unit == "week":
        # ISO week format: YYYY-WW
        year, week, _ = dt_obj.isocalendar()
        return f"{year}-W{week:02d}"
    elif time_unit == "day":
        return dt_obj.strftime("%Y-%m-%d")
    else:
        return dt_obj.strftime("%Y-%m")


def get_time_key_from_timestamp(server_timestamp, time_unit: str) -> str:
    """Convert server_timestamp to a time key based on time unit."""
    try:
        if isinstance(server_timestamp, (int, float)):
            # If it's a large number, assume milliseconds, otherwise seconds
            if server_timestamp > 1e10:
                dt_obj = datetime.datetime.fromtimestamp(server_timestamp / 1000)
            else:
                dt_obj = datetime.datetime.fromtimestamp(server_timestamp)
        elif isinstance(server_timestamp, str):
            # Try parsing as ISO format
            dt_obj = datetime.datetime.fromisoformat(
                server_timestamp.replace("Z", "+00:00")
            )
        else:
            return None

        return format_time_key(dt_obj, time_unit)
    except (ValueError, TypeError, AttributeError):
        return None


def process_metrics_with_aggregation(
    all_metrics: Dict,
    experiment_map: Dict,
    metrics_to_track: List[str],
    aggregation: str,
    time_unit: str,
    start_date: str = None,
    end_date: str = None,
    aggregate_metric: str = "avg",
) -> Tuple[Dict, Dict]:
    """
    Process metric data with specified aggregation and time unit.

    Args:
        all_metrics: Dictionary keyed by experiment key with metric data
        experiment_map: Dictionary keyed by experiment key with experiment metadata
        metrics_to_track: List of metric names to process
        aggregation: "workspace", "project", or "user"
        time_unit: "year", "month", "week", or "day"
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
        aggregate_metric: Aggregation method for time series - "max" or "avg" (default: "avg")

    Returns:
        tuple: (avg_data, time_series_data)
            avg_data: Dict[metric_name][aggregation_key] = average_value
            time_series_data: Dict[metric_name][time_key][aggregation_key] = aggregated_value
    """
    # Structure: metric_name -> aggregation_key -> list of values
    aggregation_values = defaultdict(lambda: defaultdict(list))

    # Structure: metric_name -> time_key -> aggregation_key -> list of values
    time_series_values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Parse date filters
    start_dt = None
    end_dt = None
    if start_date:
        try:
            start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError:
            pass
    if end_date:
        try:
            end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")
            # Make end_date inclusive by adding one day
            end_dt = end_dt + datetime.timedelta(days=1)
        except ValueError:
            pass

    for exp_key, metric_data in all_metrics.items():
        if exp_key not in experiment_map:
            continue

        exp_info = experiment_map[exp_key]
        server_timestamp = exp_info.get("server_timestamp")

        # Apply date filter
        if start_dt or end_dt:
            exp_dt = None
            if server_timestamp:
                try:
                    if isinstance(server_timestamp, (int, float)):
                        if server_timestamp > 1e10:
                            exp_dt = datetime.datetime.fromtimestamp(
                                server_timestamp / 1000
                            )
                        else:
                            exp_dt = datetime.datetime.fromtimestamp(server_timestamp)
                    elif isinstance(server_timestamp, str):
                        exp_dt = datetime.datetime.fromisoformat(
                            server_timestamp.replace("Z", "+00:00")
                        )
                except (ValueError, TypeError, AttributeError):
                    pass

            if exp_dt:
                if start_dt and exp_dt < start_dt:
                    continue
                if end_dt and exp_dt >= end_dt:
                    continue

        # Determine aggregation key
        if aggregation == "workspace":
            agg_key = exp_info.get("workspace", "Unknown")
        elif aggregation == "project":
            workspace = exp_info.get("workspace", "Unknown")
            project = exp_info.get("project_name", "Unknown")
            agg_key = f"{workspace}/{project}"
        elif aggregation == "user":
            # User information might not be available in experiment_map
            # Try to get it from the experiment data if available
            agg_key = exp_info.get("userName", exp_info.get("user", "Unknown"))
        else:
            agg_key = "Unknown"

        # Determine time key from server_timestamp
        time_key = None
        if server_timestamp:
            time_key = get_time_key_from_timestamp(server_timestamp, time_unit)

        # Extract values for each metric
        for metric_name in metrics_to_track:
            # For aggregation averages, always use max from experiment, then average across experiments
            value = extract_metric_value(metric_data, metric_name, aggregate="max")
            if value is not None:
                aggregation_values[metric_name][agg_key].append(value)
                if time_key:
                    time_series_values[metric_name][time_key][agg_key].append(value)

    # Calculate aggregated values per aggregation key (respects aggregate_metric selection)
    avg_data = {}
    for metric_name in metrics_to_track:
        avg_data[metric_name] = {}
        for agg_key, values in aggregation_values[metric_name].items():
            if values:
                if aggregate_metric == "avg":
                    avg_data[metric_name][agg_key] = sum(values) / len(values)
                else:  # max
                    avg_data[metric_name][agg_key] = max(values)

    # Calculate aggregated value per time unit per aggregation key
    time_series_data = {}
    for metric_name in metrics_to_track:
        time_series_data[metric_name] = {}
        for time_key in sorted(time_series_values[metric_name].keys()):
            time_series_data[metric_name][time_key] = {}
            for agg_key, values in time_series_values[metric_name][time_key].items():
                if values:
                    if aggregate_metric == "avg":
                        time_series_data[metric_name][time_key][agg_key] = sum(
                            values
                        ) / len(values)
                    else:  # max
                        time_series_data[metric_name][time_key][agg_key] = max(values)

    return avg_data, time_series_data


def create_aggregation_avg_chart(
    avg_data: Dict,
    metric_name: str,
    aggregation: str,
    aggregate_metric: str = "avg",
    png_filename=None,
):
    """Create a bar chart showing aggregated metric value per aggregation key."""
    if metric_name not in avg_data or not avg_data[metric_name]:
        return None

    data = avg_data[metric_name]

    if png_filename is None:
        safe_metric = metric_name.replace(".", "_").replace("/", "_")
        png_filename = f"gpu_report_avg_{safe_metric}_by_{aggregation}.png"

    # Sort aggregation keys by average value (descending)
    sorted_keys = sorted(data.items(), key=lambda x: x[1], reverse=True)
    keys = [k for k, _ in sorted_keys]
    values = [v for _, v in sorted_keys]

    # Create bar chart
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.bar(
        range(len(keys)),
        values,
        color="steelblue",
        edgecolor="navy",
        alpha=0.7,
        width=0.8,
    )

    # Customize the chart
    aggregation_label = aggregation.capitalize()
    aggregate_label = "Maximum" if aggregate_metric == "max" else "Average"
    ax.set_title(
        f"{aggregate_label} {metric_name} by {aggregation_label}",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel(aggregation_label, fontsize=14)
    ax.set_ylabel(f"{aggregate_label} {metric_name}", fontsize=14)

    # Set x-axis labels
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha="right")

    # Add grid
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5, axis="y")

    fig.tight_layout()
    return fig


def create_time_series_chart(
    time_series_data: Dict,
    metric_name: str,
    aggregation: str,
    time_unit: str,
    aggregate_metric: str = "avg",
    png_filename=None,
):
    """Create a time series line chart showing aggregated metric value per time unit."""
    if metric_name not in time_series_data or not time_series_data[metric_name]:
        return None

    data = time_series_data[metric_name]

    if png_filename is None:
        safe_metric = metric_name.replace(".", "_").replace("/", "_")
        png_filename = f"gpu_report_max_{safe_metric}_by_{time_unit}.png"

    # Collect all unique aggregation keys and existing time keys
    all_agg_keys = set()
    existing_time_keys = sorted(data.keys())

    for time_data in data.values():
        all_agg_keys.update(time_data.keys())

    all_agg_keys = sorted(all_agg_keys)

    if not all_agg_keys or not existing_time_keys:
        return None

    # Generate complete list of time keys from earliest to latest
    if len(existing_time_keys) > 0:
        # Parse first and last time keys
        start_dt = parse_time_key(existing_time_keys[0], time_unit)
        end_dt = parse_time_key(existing_time_keys[-1], time_unit)

        if start_dt and end_dt:
            all_time_keys = []
            current_dt = start_dt

            while current_dt <= end_dt:
                time_key = format_time_key(current_dt, time_unit)
                all_time_keys.append(time_key)

                # Move to next time unit
                if time_unit == "year":
                    current_dt = current_dt.replace(year=current_dt.year + 1)
                elif time_unit == "month":
                    if current_dt.month == 12:
                        current_dt = current_dt.replace(
                            year=current_dt.year + 1, month=1
                        )
                    else:
                        current_dt = current_dt.replace(month=current_dt.month + 1)
                elif time_unit == "week":
                    current_dt = current_dt + datetime.timedelta(weeks=1)
                elif time_unit == "day":
                    current_dt = current_dt + datetime.timedelta(days=1)
                else:
                    break
        else:
            all_time_keys = existing_time_keys
    else:
        all_time_keys = existing_time_keys

    # Create line chart
    fig, ax = plt.subplots(figsize=(14, 8))

    # Generate distinct colors for each aggregation key
    colors_list = get_distinct_colors(len(all_agg_keys))

    # Plot line for each aggregation key
    for i, agg_key in enumerate(all_agg_keys):
        values = []
        for time_key in all_time_keys:
            value = data.get(time_key, {}).get(agg_key)
            values.append(value if value is not None else None)

        ax.plot(
            range(len(all_time_keys)),
            values,
            label=agg_key,
            color=colors_list[i],
            marker="o",
            linewidth=2,
            markersize=6,
        )

    # Customize the chart
    time_unit_label = time_unit.capitalize()
    aggregate_label = "Maximum" if aggregate_metric == "max" else "Average"
    ax.set_title(
        f"{aggregate_label} {metric_name} by {time_unit_label}",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel(time_unit_label, fontsize=14)
    ax.set_ylabel(f"{aggregate_label} {metric_name}", fontsize=14)

    # Set x-axis labels - show every Nth to avoid crowding
    max_labels = 20
    step = max(1, len(all_time_keys) // max_labels)
    x_ticks = range(0, len(all_time_keys), step)
    x_labels = [all_time_keys[i] for i in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")

    # Add grid
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    # Add legend outside the plot area to prevent overlap
    # Place legend to the right of the plot, or below if too many items
    if len(all_agg_keys) > 10:
        # For many items, place legend below the chart (lower to avoid x-axis label)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
            ncol=min(len(all_agg_keys), 4),
            fontsize=8,
            framealpha=0.9,
            fancybox=True,
            shadow=True,
        )
        # Adjust layout to make room for legend
        fig.tight_layout(rect=[0, 0.1, 1, 1])
    else:
        # For fewer items, place legend to the right
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            ncol=1,
            fontsize=10,
            framealpha=0.9,
            fancybox=True,
            shadow=True,
        )
        # Adjust layout to make room for legend
        fig.tight_layout(rect=[0, 0, 0.85, 1])

    return fig


def get_available_dates(data: Dict) -> Tuple[List[str], str, str]:
    """Extract available date range from the data."""
    experiment_map = data.get("experiment_map", {})
    dates = []

    for exp_info in experiment_map.values():
        server_timestamp = exp_info.get("server_timestamp")
        if server_timestamp:
            try:
                if isinstance(server_timestamp, (int, float)):
                    if server_timestamp > 1e10:
                        dt_obj = datetime.datetime.fromtimestamp(
                            server_timestamp / 1000
                        )
                    else:
                        dt_obj = datetime.datetime.fromtimestamp(server_timestamp)
                elif isinstance(server_timestamp, str):
                    dt_obj = datetime.datetime.fromisoformat(
                        server_timestamp.replace("Z", "+00:00")
                    )
                else:
                    continue
                dates.append(dt_obj)
            except (ValueError, TypeError, AttributeError):
                continue

    if not dates:
        return [], None, None

    min_date = min(dates)
    max_date = max(dates)

    # Generate list of available dates as strings
    date_strings = [
        (min_date + datetime.timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range((max_date - min_date).days + 1)
    ]

    return (
        date_strings,
        min_date.strftime("%Y-%m-%d"),
        max_date.strftime("%Y-%m-%d"),
    )


def main():
    """Main Streamlit app entry point."""
    st.set_page_config(
        page_title="Comet GPU Report",
        page_icon="🖥️",
        layout="wide",
    )

    st.title("Comet GPU Report")
    st.markdown(
        "Visualize GPU metrics with different aggregations and time units from a saved JSON report."
    )

    # Check for JSON file path from temp file (set when --app is used with data collection)
    # Store in session state so it persists across reruns
    if "json_file_path" not in st.session_state:
        temp_dir = tempfile.gettempdir()
        json_path_file = os.path.join(temp_dir, "comet_gpu_report_json_path.txt")
        auto_json_path = None
        if os.path.exists(json_path_file):
            try:
                with open(json_path_file, "r") as f:
                    auto_json_path = f.read().strip()
                # Remove the temp file after reading (only on first run)
                try:
                    os.remove(json_path_file)
                except Exception:
                    pass
            except Exception:
                pass

        # Auto-load the JSON file from temp file
        if not auto_json_path or not os.path.exists(auto_json_path):
            st.error(
                "❌ No JSON file found. Please run the gpu-report command with --app and data collection arguments."
            )
            st.stop()

        # Store in session state
        st.session_state.json_file_path = auto_json_path

    # Get JSON file path from session state
    json_file_path = st.session_state.json_file_path

    # Load the JSON file (cache it in session state to avoid reloading on every rerun)
    if "gpu_report_data" not in st.session_state:
        data = load_json_data(json_file_path)
        if not data:
            st.error(f"❌ Failed to load JSON file: {json_file_path}")
            st.stop()
        st.session_state.gpu_report_data = data
        st.success(f"✅ Loaded JSON file: {json_file_path}")
    else:
        data = st.session_state.gpu_report_data

    # Get available dates
    available_dates, min_date, max_date = get_available_dates(data)

    # Sidebar configuration
    with st.sidebar:
        st.divider()
        st.header("Configuration")

        # Aggregation selection
        st.subheader("Aggregation")
        aggregation = st.selectbox(
            "Group by",
            options=["workspace", "project", "user"],
            index=0,
            help="Select how to aggregate the data",
        )

        # Time unit selection
        st.subheader("Time Unit")
        time_unit = st.selectbox(
            "Time unit",
            options=["year", "month", "week", "day"],
            index=1,  # Default to month
            help="Select the time unit for time series charts",
        )

        # Aggregate metric selection
        st.subheader("Aggregate Metric")
        # Get default from data if available, otherwise default to "avg"
        default_aggregate = data.get("aggregate_metric", "avg")
        # Use session state to persist user selection, but initialize from data if not set
        if "aggregate_metric" not in st.session_state:
            st.session_state.aggregate_metric = default_aggregate

        aggregate_metric = st.selectbox(
            "Aggregate metric",
            options=["avg", "max"],
            index=0 if st.session_state.aggregate_metric == "avg" else 1,
            key="aggregate_metric",
            help="Select aggregation method for time series charts (avg or max)",
        )

        # Date range selection
        st.subheader("Date Range")
        if min_date and max_date:
            start_date = st.date_input(
                "Start Date",
                value=datetime.datetime.strptime(min_date, "%Y-%m-%d").date(),
                min_value=datetime.datetime.strptime(min_date, "%Y-%m-%d").date(),
                max_value=datetime.datetime.strptime(max_date, "%Y-%m-%d").date(),
                help="Filter data from this date onwards",
            )
            end_date = st.date_input(
                "End Date",
                value=datetime.datetime.strptime(max_date, "%Y-%m-%d").date(),
                min_value=datetime.datetime.strptime(min_date, "%Y-%m-%d").date(),
                max_value=datetime.datetime.strptime(max_date, "%Y-%m-%d").date(),
                help="Filter data up to this date (inclusive)",
            )

            start_date_str = start_date.strftime("%Y-%m-%d")
            end_date_str = end_date.strftime("%Y-%m-%d")
        else:
            start_date_str = None
            end_date_str = None
            st.warning("No date information available in the data.")

    # Get metrics to track
    metrics_to_track = data.get("metrics_to_track", [])
    if not metrics_to_track:
        st.error("❌ No metrics found in the data.")
        return

    # Process data with selected aggregation and time unit
    all_metrics = data.get("metrics", {})
    experiment_map = data.get("experiment_map", {})

    # Process data with selected aggregation and time unit
    # Note: Even though both max and avg are saved in JSON, we still need to recalculate
    # for different time units (year/week/day) and aggregations (project/user)
    # The pre-computed data is organized by month and workspace only
    with st.spinner("Processing data..."):
        avg_data, time_series_data = process_metrics_with_aggregation(
            all_metrics,
            experiment_map,
            metrics_to_track,
            aggregation,
            time_unit,
            start_date_str,
            end_date_str,
            aggregate_metric=aggregate_metric,
        )

    # Display statistics
    st.header("Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)

    total_experiments = len(all_metrics)
    unique_workspaces = len(
        set(exp.get("workspace", "Unknown") for exp in experiment_map.values())
    )
    unique_projects = len(
        set(
            f"{exp.get('workspace', 'Unknown')}/{exp.get('project_name', 'Unknown')}"
            for exp in experiment_map.values()
        )
    )
    metrics_count = len(metrics_to_track)

    with col1:
        st.metric("Total Experiments", f"{total_experiments:,}")

    with col2:
        st.metric("Workspaces", f"{unique_workspaces:,}")

    with col3:
        st.metric("Projects", f"{unique_projects:,}")

    with col4:
        st.metric("Metrics Tracked", f"{metrics_count}")

    # Create tabs for each metric
    metric_tabs = st.tabs([f"📊 {metric}" for metric in metrics_to_track])

    for idx, metric_name in enumerate(metrics_to_track):
        with metric_tabs[idx]:
            col1, col2 = st.columns(2)

            # Aggregated chart (respects aggregate_metric selection)
            with col1:
                aggregate_label = "Maximum" if aggregate_metric == "max" else "Average"
                st.subheader(
                    f"{aggregate_label} {metric_name} by {aggregation.capitalize()}"
                )
                if metric_name in avg_data and avg_data[metric_name]:
                    fig_avg = create_aggregation_avg_chart(
                        avg_data,
                        metric_name,
                        aggregation,
                        aggregate_metric=aggregate_metric,
                    )
                    if fig_avg:
                        st.pyplot(fig_avg)
                        plt.close(fig_avg)
                    else:
                        st.info("No data available for this chart.")
                else:
                    st.info("No data available for this chart.")

            # Time series chart
            with col2:
                aggregate_label = "Maximum" if aggregate_metric == "max" else "Average"
                st.subheader(
                    f"{aggregate_label} {metric_name} by {time_unit.capitalize()}"
                )
                if metric_name in time_series_data and time_series_data[metric_name]:
                    fig_ts = create_time_series_chart(
                        time_series_data,
                        metric_name,
                        aggregation,
                        time_unit,
                        aggregate_metric=aggregate_metric,
                    )
                    if fig_ts:
                        st.pyplot(fig_ts)
                        plt.close(fig_ts)
                    else:
                        st.info("No data available for this chart.")
                else:
                    st.info("No data available for this chart.")


# Streamlit will execute this when the file is run
if __name__ == "__main__":
    main()
