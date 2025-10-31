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
Streamlit app for generating usage reports with experiment counts and statistics.

This app provides a web-based interface for the admin usage report functionality,
allowing users to visualize experiment data, GPU utilization, and statistics
interactively.
"""

import os
import tempfile
from collections import defaultdict

import streamlit as st
from comet_ml import API

from cometx.cli.admin_usage_report import (
    create_combined_chart_from_data,
    create_combined_gpu_chart_from_data,
    extract_website_name,
    format_gpu_hours,
    format_time,
    format_time_key,
    generate_experiment_chart,
    get_unit_label,
    get_unit_label_plural,
    parse_time_key,
)


@st.cache_data(persist="disk", show_spinner=False)
def fetch_experiment_data(_api, workspace, project, units="month"):
    """
    Fetch experiment data with disk persistence caching.

    This function caches the results to disk so data doesn't need to be
    reloaded on each Streamlit rerun. The cache is invalidated when
    workspace, project, or units change. The API object is identified
    by its base_url for cache key purposes.

    Args:
        api: Comet API instance
        workspace: Workspace name
        project: Project name
        units: Time unit for grouping ("month", "week", "day", or "hour")

    Returns:
        dict: Experiment data results or empty dict if error
    """
    try:
        results = generate_experiment_chart(
            _api, workspace, project, units=units, debug=False
        )
        return results if results else {}
    except Exception as e:
        # Return empty dict on error, don't cache errors
        return {}


def display_statistics(
    all_results, workspace_projects_input, website_name=None, time_unit="month"
):
    """Display summary statistics using Streamlit components."""
    if not all_results:
        st.warning("No data available to display statistics.")
        return

    st.header("Summary Statistics")

    if website_name:
        st.subheader(website_name)

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
            all_start_dates.append(parse_time_key(date_range[0], time_unit))
            all_end_dates.append(parse_time_key(date_range[1], time_unit))

    earliest_date = min(all_start_dates) if all_start_dates else None
    latest_date = max(all_end_dates) if all_end_dates else None

    # Find peak time unit
    combined_time_unit_counts = defaultdict(int)
    for result in all_results:
        time_unit_counts = result.get("monthly_counts", {})
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

    # Calculate total unique users
    all_unique_users = set()
    for result in all_results:
        unique_users = result.get("unique_users", set())
        if unique_users:
            all_unique_users.update(unique_users)
    total_unique_users = len(all_unique_users)

    # Calculate total GPU hours
    total_gpu_duration_ms = sum(
        result.get("total_gpu_duration_ms", 0) for result in all_results
    )

    # Display key metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Experiments", f"{total_experiments:,}")

    with col2:
        st.metric("Total Users", f"{total_unique_users:,}")

    with col3:
        if avg_gpu_utilization is not None:
            st.metric("Avg GPU Utilization", f"{avg_gpu_utilization:.1f}%")
        else:
            st.metric("Avg GPU Utilization", "N/A")

    with col4:
        st.metric("Total GPU Hours", format_gpu_hours(total_gpu_duration_ms))

    # Display detailed statistics
    st.subheader("Overall Statistics")

    stats_data = {"Metric": [], "Value": []}

    stats_data["Metric"].append("Total Number of Experiments")
    stats_data["Value"].append(f"{total_experiments:,}")

    stats_data["Metric"].append("Total Number of Users")
    stats_data["Value"].append(f"{total_unique_users:,}")

    if total_run_time_seconds > 0:
        stats_data["Metric"].append("Total Run Time")
        stats_data["Value"].append(format_time(total_run_time_seconds))

        stats_data["Metric"].append("Experiments with Run Time Data")
        stats_data["Value"].append(f"{run_time_count:,}")

        if run_time_count > 0:
            stats_data["Metric"].append("Average Run Time per Experiment")
            stats_data["Value"].append(format_time(avg_run_time_seconds))

    if earliest_date and latest_date:
        start_key = format_time_key(earliest_date, time_unit)
        end_key = format_time_key(latest_date, time_unit)
        unit_label_plural = get_unit_label_plural(time_unit)

        stats_data["Metric"].append("Date Range")
        stats_data["Value"].append(f"{start_key} to {end_key}")

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

        stats_data["Metric"].append("Time Span")
        stats_data["Value"].append(f"{span} {unit_label_plural.lower()}")

    stats_data["Metric"].append("Number of Workspaces/Projects")
    stats_data["Value"].append(f"{len(all_results)}")

    if peak_time_unit:
        unit_label = get_unit_label(time_unit)
        stats_data["Metric"].append(f"Peak {unit_label}")
        stats_data["Value"].append(
            f"{peak_time_unit[0]} ({peak_time_unit[1]:,} experiments)"
        )

    if avg_gpu_utilization is not None:
        stats_data["Metric"].append("Average GPU Utilization")
        stats_data["Value"].append(f"{avg_gpu_utilization:.1f}%")

    if avg_memory_utilization is not None:
        stats_data["Metric"].append("Average GPU Memory Utilization")
        stats_data["Value"].append(f"{avg_memory_utilization:.1f}%")

    stats_data["Metric"].append("Total GPU Hours")
    stats_data["Value"].append(format_gpu_hours(total_gpu_duration_ms))

    # Display statistics as a table (hide index column)
    st.dataframe(stats_data, use_container_width=True, hide_index=True)

    # Breakdown section - only show if there is more than one project
    if len(all_results) > 1:
        st.subheader("Breakdown by Workspace/Project")

        breakdown_data = []
        for result in all_results:
            wp = result.get("workspace_project", "Unknown")
            exp_count = result.get("total_experiments", 0)
            run_time = result.get("total_run_time_seconds", 0)
            run_time_exp_count = result.get("run_time_count", 0)

            # Calculate GPU utilization for this workspace/project
            gpu_utils = result.get("avg_gpu_utilizations", [])
            memory_utils = result.get("avg_memory_utilizations", [])
            gpu_values = [u for u in gpu_utils if u is not None]
            memory_values = [u for u in memory_utils if u is not None]

            avg_gpu_util = sum(gpu_values) / len(gpu_values) if gpu_values else None
            avg_memory_util = (
                sum(memory_values) / len(memory_values) if memory_values else None
            )

            gpu_duration_ms = result.get("total_gpu_duration_ms", 0)

            row = {
                "Workspace/Project": wp,
                "Experiments": f"{exp_count:,}",
            }

            if run_time > 0:
                row["Total Run Time"] = format_time(run_time)
                if run_time_exp_count > 0:
                    avg = run_time / run_time_exp_count
                    row["Avg Run Time"] = format_time(avg)

            if avg_gpu_util is not None:
                row["Avg GPU Utilization"] = f"{avg_gpu_util:.1f}%"

            if avg_memory_util is not None:
                row["Avg GPU Memory Utilization"] = f"{avg_memory_util:.1f}%"

            row["Total GPU Hours"] = format_gpu_hours(gpu_duration_ms)

            breakdown_data.append(row)

        st.dataframe(breakdown_data, use_container_width=True)


def main():
    """Main Streamlit app entry point."""
    st.set_page_config(
        page_title="Comet Usage Report",
        page_icon="📊",
        layout="wide",
    )

    st.title("Comet Usage Report")
    st.markdown(
        "Generate usage reports with experiment counts, statistics, and GPU utilization data."
    )

    # Initialize API (uses environment variables COMET_API_KEY and COMET_URL_OVERRIDE)
    try:
        api = API()
    except Exception as e:
        st.error(f"❌ Failed to initialize API: {e}")
        st.info("💡 Make sure COMET_API_KEY is set in your environment variables.")
        st.exception(e)
        return

    # Get workspaces
    try:
        with st.spinner("Loading workspaces..."):
            workspaces = sorted(api.get_workspaces())
    except Exception as e:
        st.error(f"❌ Failed to load workspaces: {e}")
        st.exception(e)
        return

    if not workspaces:
        st.warning("⚠️ No workspaces found.")
        return

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")

        # Workspace/Project Selection
        st.subheader("Workspace/Project Selection")
        selected_workspace = st.selectbox(
            "Workspace",
            options=workspaces,
            help="Select a workspace",
        )

        # Get projects for selected workspace
        try:
            with st.spinner("Loading projects..."):
                projects = sorted(api.get_projects(selected_workspace))
        except Exception as e:
            st.error(f"❌ Failed to load projects for {selected_workspace}: {e}")
            st.stop()

        if not projects:
            st.warning(f"⚠️ No projects found in workspace '{selected_workspace}'.")
            st.stop()

        # Add empty string and "All Projects" options, then actual projects
        project_options = ["", "All Projects"] + sorted(projects)
        selected_project = st.selectbox(
            "Project",
            options=project_options,
            help="Select a project or 'All Projects' to view combined data",
            key="project_selectbox",
        )

        # Reset time unit to "month" when project changes
        if "last_selected_project" not in st.session_state:
            st.session_state.last_selected_project = selected_project

        if st.session_state.last_selected_project != selected_project:
            # Reset time unit when project changes
            st.session_state.time_unit_selectbox = "month"
            st.session_state.last_selected_project = selected_project

        # Initialize time_unit_selectbox in session state if not present
        if "time_unit_selectbox" not in st.session_state:
            st.session_state.time_unit_selectbox = "month"

        st.divider()

        # Report Options
        st.subheader("Report Options")
        time_unit_options = ["month", "week", "day", "hour"]

        # Ensure session state value is valid
        if st.session_state.time_unit_selectbox not in time_unit_options:
            st.session_state.time_unit_selectbox = "month"

        # Get the index for the current session state value
        current_index = time_unit_options.index(st.session_state.time_unit_selectbox)

        # Render selectbox - Streamlit will manage the state via the key
        _ = st.selectbox(
            "Time Unit",
            options=time_unit_options,
            index=current_index,
            help="Time unit for grouping experiments",
            key="time_unit_selectbox",
        )

        # Read the value directly from session state (this is the source of truth)
        units = st.session_state.time_unit_selectbox

    # Main content area
    if not selected_project:
        st.info(
            "👈 Select a workspace and project in the sidebar to generate the report."
        )
        return

    # Collect data for selected workspace/project(s)
    all_results = []

    if selected_project == "All Projects":
        # Collect data for all projects in the workspace
        with st.spinner(f"Collecting data for all projects in {selected_workspace}..."):
            for project in sorted(projects):
                results = fetch_experiment_data(
                    api, selected_workspace, project, units=units
                )
                if results:
                    all_results.append(results)
                # Silently skip projects with no data
    else:
        # Collect data for single project
        with st.spinner(
            f"Collecting data for {selected_workspace}/{selected_project}..."
        ):
            results = fetch_experiment_data(
                api, selected_workspace, selected_project, units=units
            )
            if not results:
                st.error(
                    f"❌ Error collecting experiment data for {selected_workspace}/{selected_project}"
                )
                return
            all_results.append(results)

    if not all_results:
        st.error("❌ No data collected. Cannot create report.")
        return

    # Extract website name
    try:
        base_url = api._client.base_url
        website_name = extract_website_name(base_url)
    except Exception:
        website_name = None

    # Calculate overall date range for charts (for consistent scaling across all projects)
    all_date_ranges = [r.get("date_range") for r in all_results if r.get("date_range")]
    if all_date_ranges:
        start_dates = [parse_time_key(dr[0], units) for dr in all_date_ranges]
        end_dates = [parse_time_key(dr[1], units) for dr in all_date_ranges]
        date_range = (
            format_time_key(min(start_dates), units),
            format_time_key(max(end_dates), units),
        )
    else:
        date_range = None

    # Check if experiment count data exists
    has_experiment_data = any(
        r.get("counts") and any(count > 0 for count in r.get("counts", []))
        for r in all_results
    )

    # Check if GPU data exists
    has_gpu_data = any(
        r.get("avg_gpu_utilizations")
        and any(u is not None for u in r.get("avg_gpu_utilizations", []))
        for r in all_results
    )

    # Check if memory data exists
    has_memory_data = any(
        r.get("avg_memory_utilizations")
        and any(u is not None for u in r.get("avg_memory_utilizations", []))
        for r in all_results
    )

    # Create tabs for statistics and charts
    tab_labels = ["📊 Statistics"]
    if has_experiment_data:
        tab_labels.append("📈 Experiment Count")
    if has_gpu_data:
        tab_labels.append("🖥️ GPU Utilization")
    if has_memory_data:
        tab_labels.append("💾 GPU Memory")

    tabs = st.tabs(tab_labels)

    # Statistics tab
    with tabs[0]:
        # Prepare workspace/project list for display
        if selected_project == "All Projects":
            workspace_projects_display = [
                f"{selected_workspace}/{p}" for p in sorted(projects)
            ]
        else:
            workspace_projects_display = [f"{selected_workspace}/{selected_project}"]

        display_statistics(
            all_results,
            workspace_projects_display,
            website_name=website_name,
            time_unit=units,
        )

    # Experiment Count Chart tab
    tab_index = 1
    if has_experiment_data:
        with tabs[tab_index]:
            with st.spinner("Generating experiment count chart..."):
                try:
                    with tempfile.NamedTemporaryFile(
                        suffix=".png", delete=False
                    ) as tmp_file:
                        tmp_filename = tmp_file.name

                    chart_filename = create_combined_chart_from_data(
                        all_results,
                        png_filename=tmp_filename,
                        date_range=date_range,
                        time_unit=units,
                        debug=False,
                    )

                    if chart_filename and os.path.exists(chart_filename):
                        st.image(chart_filename)
                        # Clean up temp file
                        try:
                            os.unlink(chart_filename)
                        except:
                            pass
                except Exception as e:
                    st.error(f"❌ Error generating chart: {e}")
                    st.exception(e)
        tab_index += 1

    # GPU Utilization Chart tab
    if has_gpu_data:
        with tabs[tab_index]:
            with st.spinner("Generating GPU utilization chart..."):
                try:
                    with tempfile.NamedTemporaryFile(
                        suffix=".png", delete=False
                    ) as tmp_file:
                        tmp_filename = tmp_file.name

                    gpu_chart_filename = create_combined_gpu_chart_from_data(
                        all_results,
                        png_filename=tmp_filename,
                        date_range=date_range,
                        utilization_type="gpu",
                        time_unit=units,
                        debug=False,
                    )

                    if gpu_chart_filename and os.path.exists(gpu_chart_filename):
                        st.image(gpu_chart_filename)
                        try:
                            os.unlink(gpu_chart_filename)
                        except:
                            pass
                except Exception as e:
                    st.error(f"❌ Error generating GPU chart: {e}")
                    st.exception(e)
        tab_index += 1

    # GPU Memory Utilization Chart tab
    if has_memory_data:
        with tabs[tab_index]:
            with st.spinner("Generating GPU memory utilization chart..."):
                try:
                    with tempfile.NamedTemporaryFile(
                        suffix=".png", delete=False
                    ) as tmp_file:
                        tmp_filename = tmp_file.name

                    memory_chart_filename = create_combined_gpu_chart_from_data(
                        all_results,
                        png_filename=tmp_filename,
                        date_range=date_range,
                        utilization_type="memory",
                        time_unit=units,
                        debug=False,
                    )

                    if memory_chart_filename and os.path.exists(memory_chart_filename):
                        st.image(memory_chart_filename)
                        try:
                            os.unlink(memory_chart_filename)
                        except:
                            pass
                except Exception as e:
                    st.error(f"❌ Error generating memory chart: {e}")
                    st.exception(e)

    # Success message
    if selected_project == "All Projects":
        st.success(
            f"✅ Report generated successfully for all projects in {selected_workspace}!"
        )
    else:
        st.success(
            f"✅ Report generated successfully for {selected_workspace}/{selected_project}!"
        )


# Streamlit will execute this when the file is run
main()
