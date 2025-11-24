import json
import logging
import os
import warnings
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
from comet_ml.config import get_config, get_optimizer_address


def fetch_dashboard_data(
    optimizer_id, api_key, optimizer_url, status_filter=None, page=1, page_size=100
):
    """Fetch data from the Flask API endpoint with filtering and pagination."""
    try:
        # Build query parameters
        params = [("id", optimizer_id), ("page", page), ("page_size", page_size)]

        # Add filter parameters - requests needs tuples for multiple values with same key
        if status_filter:
            for status in status_filter:
                params.append(("status", status))

        response = requests.get(
            f"{optimizer_url}/dashboard/data",
            params=params,
            headers={"X-API-KEY": api_key},
            timeout=10,
        )
        if response.ok:
            return response.json(), None
        else:
            return None, f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return None, f"Error fetching data: {str(e)}"


def format_duration(ms):
    """Format duration in milliseconds to human-readable format."""
    if not ms or ms == 0:
        return None
    seconds = ms // 1000
    minutes = seconds // 60
    hours = minutes // 60
    days = hours // 24

    if days > 0:
        return f"{days}d {hours % 24}h {minutes % 60}m"
    elif hours > 0:
        return f"{hours}h {minutes % 60}m {seconds % 60}s"
    elif minutes > 0:
        return f"{minutes}m {seconds % 60}s"
    else:
        return f"{seconds}s"


def get_status_emoji(status):
    """Get emoji for status badge."""
    status_lower = (status or "").lower()
    if status_lower == "completed":
        return "✅"
    elif status_lower == "aborted":
        return "❌"
    elif status_lower in ["running", "assigned"]:
        return "🔄"
    else:
        return "⚪"


def fetch_all_dashboard_data(optimizer_id, api_key, optimizer_url, status_filter=None):
    """
    Fetch all dashboard data for an optimizer by paginating through all pages.

    Args:
        optimizer_id: The optimizer instance ID
        api_key: API key for authentication
        optimizer_url: Base URL for the optimizer service
        status_filter: Optional list of statuses to filter by

    Returns:
        tuple: (all_data_dict, error_message)
        - all_data_dict: Dictionary containing all dashboard data and jobs
        - error_message: None if successful, error string otherwise
    """
    all_jobs = []
    page = 1
    page_size = 100
    first_page_data = None

    while True:
        data, error = fetch_dashboard_data(
            optimizer_id,
            api_key,
            optimizer_url,
            status_filter=status_filter,
            page=page,
            page_size=page_size,
        )

        if error:
            return None, error

        if not data or data.get("code") != 200:
            return None, f"Failed to load data: {data}"

        # Store first page data for metadata
        if page == 1:
            first_page_data = data.copy()
            first_page_data.pop("jobs", None)  # Remove jobs, we'll add all jobs later

        # Collect jobs from this page
        jobs = data.get("jobs", [])
        if jobs:
            all_jobs.extend(jobs)

        # Check if there are more pages
        pagination = data.get("pagination", {})
        total_pages = pagination.get("total_pages", 1)

        if page >= total_pages:
            break

        page += 1

    # Combine all data
    if first_page_data:
        first_page_data["jobs"] = all_jobs
        first_page_data["pagination"] = {
            "total_jobs": len(all_jobs),
            "total_pages": 1,  # All jobs are now in one "page"
            "current_page": 1,
            "page_size": len(all_jobs),
        }
        return first_page_data, None

    return None, "No data retrieved"


def generate_json_report(
    optimizer_id, api_key=None, optimizer_url=None, output_file=None
):
    """
    Generate a JSON report of all dashboard/data items for an optimizer.

    Args:
        optimizer_id: The optimizer instance ID
        api_key: Optional API key (uses config if not provided)
        optimizer_url: Optional optimizer URL (uses config if not provided)
        output_file: Optional output file path (defaults to optimizer-report-{optimizer_id}.json)

    Returns:
        str: Path to the generated JSON file, or None on error
    """
    # Get configuration if not provided
    if api_key is None or optimizer_url is None:
        config = get_config()
        if api_key is None:
            try:
                api_key = config["comet.api_key"]
            except (KeyError, TypeError):
                api_key = None
        if optimizer_url is None:
            optimizer_url = get_optimizer_address(config)

    if not api_key:
        print("ERROR: API key is required. Set COMET_API_KEY or use --api-key flag.")
        return None

    if not optimizer_url:
        print("ERROR: Optimizer URL is required.")
        return None

    print(f"Fetching all dashboard data for optimizer {optimizer_id}...")

    # Fetch all data
    all_data, error = fetch_all_dashboard_data(optimizer_id, api_key, optimizer_url)

    if error:
        print(f"ERROR: {error}")
        return None

    # Determine output filename
    if output_file is None:
        output_file = f"optimizer-report-{optimizer_id}.json"

    # Write JSON file
    try:
        with open(output_file, "w") as f:
            json.dump(all_data, f, indent=2)
        print(f"JSON report saved to: {output_file}")
        return output_file
    except Exception as e:
        print(f"ERROR writing JSON file: {e}")
        return None


def run_streamlit_app():
    """Run the Streamlit app for optimizer dashboard."""

    # Suppress the harmless ScriptRunContext warning from Streamlit's logger
    # Only set up filters when actually running in Streamlit context
    class ScriptRunContextFilter(logging.Filter):
        def filter(self, record):
            return (
                "missing ScriptRunContext" not in record.getMessage()
                and "Session state does not function" not in record.getMessage()
            )

    # Apply filter to Streamlit loggers
    streamlit_logger = logging.getLogger(
        "streamlit.runtime.scriptrunner_utils.script_run_context"
    )
    streamlit_logger.addFilter(ScriptRunContextFilter())
    # Also filter general streamlit warnings
    logging.getLogger("streamlit").addFilter(ScriptRunContextFilter())

    # Also suppress Python warnings
    warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
    warnings.filterwarnings("ignore", message=".*Session state does not function.*")

    # Page configuration - must be first Streamlit call
    st.set_page_config(
        page_title="Comet Optimizer Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for better styling
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 600;
            color: #667eea;
            margin-bottom: 1rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # Get configuration from comet_ml.config (after Streamlit is initialized)
    try:
        config = get_config()
        api_key = config["comet.api_key"]
        optimizer_url = get_optimizer_address(config)
    except Exception as e:
        st.error(f"Failed to get configuration: {e}")
        return

    # Sidebar for optimizer ID input
    with st.sidebar:
        st.header("📊 Dashboard Settings")
        st.markdown("---")

        # Get optimizer ID from query params, environment, or user input
        default_id = st.query_params.get("id", "") or os.environ.get(
            "COMET_OPTIMIZER_ID", ""
        )
        optimizer_id = st.text_input(
            "Optimizer ID",
            value=default_id,
            help="Enter the optimizer instance ID",
        )

        st.markdown("---")
        st.markdown("### Configuration")
        st.text(f"Server: {optimizer_url}")

        # Filters
        st.markdown("---")
        st.markdown("### Filters")

        # Status filter - we'll get available statuses from a first fetch
        # Use on_change callback to reset page when filter changes
        # Streamlit automatically reruns when widget values change, so we just update state
        def on_status_change():
            st.session_state.current_page = 1

        status_filter = st.multiselect(
            "Filter by Status",
            options=["completed", "aborted", "running", "assigned"],
            default=[],
            help="Select statuses to display (empty = all)",
            key="status_filter",
            on_change=on_status_change,
        )

        # Manual refresh button
        st.markdown("---")
        if st.button("🔄 Refresh", width="stretch"):
            st.rerun()

        # Pagination settings
        st.markdown("---")
        st.markdown("### Pagination")
        page_size = st.selectbox(
            "Items per page", [1, 2, 3, 25, 50, 100, 200, 500], index=5, key="page_size"
        )

    if not optimizer_id:
        st.info("👈 Please enter an Optimizer ID in the sidebar to view the dashboard.")
        return

    # Initialize session state for pagination
    if "current_page" not in st.session_state:
        st.session_state.current_page = 1

    # Track page_size changes and reset to page 1 when it changes
    if "last_page_size" not in st.session_state:
        st.session_state.last_page_size = page_size
    elif st.session_state.last_page_size != page_size:
        # Page size changed - reset to page 1
        st.session_state.current_page = 1
        st.session_state.last_page_size = page_size

    # Main content
    st.markdown(
        '<div class="main-header">Optimizer Dashboard</div>', unsafe_allow_html=True
    )

    # Fetch data with filters and pagination
    # Always use session state directly - this ensures we get the latest value
    data, error = fetch_dashboard_data(
        optimizer_id,
        api_key,
        optimizer_url,
        status_filter=status_filter if status_filter else None,
        page=st.session_state.current_page,
        page_size=page_size,
    )

    if error:
        st.error(error)
        return

    if not data or data.get("code") != 200:
        st.error(f"Failed to load data: {data}")
        return

    # Display stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Jobs",
            f"{data['max_jobs']:,}",
            help="Maximum number of jobs (maxCombo or config_space_size)",
        )

    with col2:
        st.metric(
            "Assigned",
            f"{data['assigned_count']:,}",
            delta=f"{data['assigned_count'] - data.get('completed_count', 0):,} in progress",
            delta_color="normal",
        )

    with col3:
        st.metric(
            "Completed", f"{data['completed_count']:,}", help="Number of completed jobs"
        )

    with col4:
        st.metric(
            "Failed",
            f"{data['failed_count']:,}",
            delta_color="inverse" if data["failed_count"] > 0 else "normal",
        )

    # Progress bar
    if data["max_jobs"] > 0:
        progress = (data["assigned_count"] / data["max_jobs"]) * 100
        st.progress(progress / 100)
        st.caption(
            f"Progress: {data['assigned_count']:,} of {data['max_jobs']:,} jobs assigned ({progress:.1f}%)"
        )

    st.markdown("---")

    # Jobs table
    if data.get("jobs") and len(data["jobs"]) > 0:
        st.subheader(f"Assignments ({len(data['jobs']):,})")

        # Convert to DataFrame
        df = pd.DataFrame(data["jobs"])

        # Format timestamps
        if "startTime" in df.columns:
            df["Time Created"] = pd.to_datetime(
                df["startTime"], unit="ms", errors="coerce"
            )
        if "endTime" in df.columns:
            df["Time Finished"] = pd.to_datetime(
                df["endTime"], unit="ms", errors="coerce"
            )

        # Format duration
        if "duration" in df.columns:
            df["Duration"] = df["duration"].apply(format_duration)

        # Format metadata - convert dict to readable string
        if "metadata" in df.columns:

            def format_metadata(meta):
                if meta is None or (isinstance(meta, float) and pd.isna(meta)):
                    return "-"
                if isinstance(meta, dict):
                    return (
                        json.dumps(meta, indent=0).replace("\n", " ").replace("  ", " ")
                    )
                return str(meta)

            df["Metadata"] = df["metadata"].apply(format_metadata)

        # Get pagination info
        pagination = data.get("pagination", {})
        total_jobs = pagination.get("total_jobs", len(data.get("jobs", [])))
        total_pages = pagination.get("total_pages", 1)

        # Get current page from session state and ensure it's within valid range
        current_page = st.session_state.get("current_page", 1)
        if total_pages > 0 and current_page > total_pages:
            st.session_state.current_page = total_pages
            current_page = total_pages
        if current_page < 1:
            st.session_state.current_page = 1
            current_page = 1

        # Pagination controls
        if total_pages > 1:
            st.markdown("### Navigation")
            col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1.5, 1, 1, 1])
            with col1:
                if st.button(
                    "⏮️ First",
                    disabled=(current_page == 1),
                    width="stretch",
                    key="btn_first",
                ):
                    st.session_state.current_page = 1
                    st.rerun()
            with col2:
                if st.button(
                    "◀️ Prev",
                    disabled=(current_page == 1),
                    width="stretch",
                    key="btn_prev",
                ):
                    st.session_state.current_page = max(1, current_page - 1)
                    st.rerun()
            with col3:
                # Page number input for direct navigation
                # Use on_change to avoid conflicts with buttons
                # Store total_pages in session state to avoid closure issues
                if "total_pages_for_input" not in st.session_state:
                    st.session_state.total_pages_for_input = total_pages
                else:
                    st.session_state.total_pages_for_input = total_pages

                def on_page_change():
                    new_page = st.session_state.page_input
                    max_pages = st.session_state.total_pages_for_input
                    if 1 <= new_page <= max_pages:
                        st.session_state.current_page = new_page

                st.number_input(
                    "Go to page",
                    min_value=1,
                    max_value=total_pages,
                    value=current_page,
                    key="page_input",
                    on_change=on_page_change,
                    help=f"Page {current_page} of {total_pages} ({total_jobs:,} total jobs)",
                )
            with col4:
                if st.button(
                    "Next ▶️",
                    disabled=(current_page >= total_pages),
                    width="stretch",
                    key="btn_next",
                ):
                    st.session_state.current_page = min(total_pages, current_page + 1)
                    st.rerun()
            with col5:
                if st.button(
                    "Last ⏭️",
                    disabled=(current_page >= total_pages),
                    width="stretch",
                    key="btn_last",
                ):
                    st.session_state.current_page = total_pages
                    st.rerun()
            with col6:
                st.markdown(
                    f"<div style='text-align: center; padding-top: 0.5rem;'><strong>Page {current_page} of {total_pages}</strong><br/>{total_jobs:,} total jobs</div>",
                    unsafe_allow_html=True,
                )
        elif total_jobs > 0:
            st.caption(f"Showing all {total_jobs:,} jobs (single page)")

        # Select and rename columns for display
        display_columns = {
            "count": "#",
            "status": "Status",
            "Time Created": "Time Created",
            "Time Finished": "Time Finished",
            "Duration": "Duration",
            "score": "Metric (Score)",
            "retries": "Retries",
            "trial": "Trial",
            "Metadata": "Metadata",
        }

        # Filter available columns
        available_cols = {k: v for k, v in display_columns.items() if k in df.columns}
        df_display = df[list(available_cols.keys())].copy()
        df_display = df_display.rename(columns=available_cols)

        # Add status indicator with emoji
        if "Status" in df_display.columns:
            df_display["Status"] = df_display["Status"].apply(
                lambda x: f"{get_status_emoji(x)} {x.upper()}" if x else "⚪ UNKNOWN"
            )

        # Format numbers
        if "Metric (Score)" in df_display.columns:
            df_display["Metric (Score)"] = df_display["Metric (Score)"].apply(
                lambda x: (
                    f"{x:,.2f}" if pd.notna(x) and isinstance(x, (int, float)) else "-"
                )
            )

        # Display table with pagination
        st.dataframe(
            df_display,
            width="stretch",
            height=600,
            hide_index=True,
            column_config={
                "Time Created": st.column_config.DatetimeColumn(
                    "Time Created", format="YYYY-MM-DD HH:mm:ss"
                ),
                "Time Finished": st.column_config.DatetimeColumn(
                    "Time Finished", format="YYYY-MM-DD HH:mm:ss"
                ),
                "Metric (Score)": st.column_config.NumberColumn(
                    "Metric (Score)", format="%.2f"
                ),
            },
        )

        # Summary statistics (use filtered dataframe)
        with st.expander("📈 Summary Statistics"):
            col1, col2, col3 = st.columns(3)

            with col1:
                if "score" in df.columns and df["score"].notna().any():
                    st.metric("Average Score", f"{df['score'].mean():.4f}")
                    st.metric("Best Score", f"{df['score'].max():.4f}")
                    st.metric("Worst Score", f"{df['score'].min():.4f}")

            with col2:
                if "duration" in df.columns and df["duration"].notna().any():
                    avg_duration_ms = df["duration"].mean()
                    st.metric("Average Duration", format_duration(avg_duration_ms))
                    max_duration_ms = df["duration"].max()
                    st.metric("Max Duration", format_duration(max_duration_ms))
                    min_duration_ms = df["duration"].min()
                    st.metric("Min Duration", format_duration(min_duration_ms))

            with col3:
                if "retries" in df.columns:
                    st.metric("Total Retries", f"{df['retries'].sum():,}")
                    st.metric("Jobs with Retries", f"{(df['retries'] > 0).sum():,}")

    else:
        st.info("No assignments found for this optimizer.")

    # Status and refresh info
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        status = data.get("status", "unknown")
        status_emoji = {
            "running": "🟢",
            "completed": "✅",
            "paused": "⏸️",
            "aborted": "🔴",
        }.get(status.lower(), "⚪")
        st.caption(f"{status_emoji} Status: {status.upper()}")
        st.caption(f"📝 Name: {data.get('name', 'N/A')}")

    with col2:
        st.caption(f"🕒 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    """Main entry point for Streamlit app."""
    run_streamlit_app()


# Streamlit will execute this when the file is run
# Only run if this is being executed by Streamlit
if __name__ == "__main__":
    main()
