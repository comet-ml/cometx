# -*- coding: utf-8 -*-
# ****************************************
#                              __
#   _________  ____ ___  ___  / /__  __
#  / ___/ __ \/ __ `__ \/ _ \/ __/ |/_/
# / /__/ /_/ / / / / / /  __/ /__>  <
# \___/\____/_/ /_/ /_/\___/\__/_/|_|
#
#
#  Copyright (c) 2026 Cometx Development
#      Team. All rights reserved.
# ****************************************
"""
The Streamlit app behind `cometx panel-studio`.

Run it with `cometx panel-studio`, which also starts the server the panel is
shown from.
"""
import base64
import os

import streamlit as st
import streamlit.components.v1 as components
from comet_ml.config import get_config

from cometx.api import API
from cometx.panel_studio import (
    clientlib_url,
    dashboard_url,
    default_selection,
    display_path,
    list_directory,
    new_page_name,
    panel_config,
    project_url,
    server_root,
    sort_entries,
    upload_to_dashboard,
    upload_to_library,
    write_panel_page,
)

# Comet's own starting code for a new Python panel: pyCode in the frontend's
# experiment-management-shared/constants/customPanelTemplates/defaultPyTemplate.js
NEW_PANEL_CODE = """\
# Comet Python Panels BETA, full documentation available at:
# https://www.comet.com/docs/v2/guides/comet-ui/experiment-management/visualizations/python-panel/

from comet_ml import API, ui
import plotly.express as px

# Initialize Comet API
api = API()

# Get available metrics and select one
available_metrics = api.get_panel_metrics_names()
selected_metric = ui.dropdown("Select a metric:", available_metrics)

# Fetch experiment data
experiment_keys = api.get_panel_experiment_keys()
if experiment_keys and selected_metric:
    # Fetch the selected metric data for all experiments
    metrics_df = api.get_metrics_df(experiment_keys, [selected_metric])

    # Let's make sure experiments are the right colors
    colors = api.get_panel_experiment_colors()
    if colors:
        colors = {key: value["primary"] for key,value in colors.items()}
    fig = px.line(
        metrics_df,
        x="step",
        y=selected_metric,
        color="experiment_key",
        color_discrete_map=colors,
    )

    # Update layout
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        xaxis_title=None,
        yaxis_title=None,
    )

    # Display the Plotly figure
    ui.display(fig)
else:
    ui.display("No data to plot. Make sure your metric data is logged by step.")
"""

NEW_DASHBOARD = "New dashboard"

PAGES_DIR = os.environ["COMETX_PANEL_STUDIO_PAGES"]
PANEL_PORT = os.environ["COMETX_PANEL_STUDIO_PANEL_PORT"]

# The panel fills the editor's right-hand column, at this height.
PANEL_HEIGHT = 720

# Comet's logo mark, from the frontend's public/images/comet-only-logo.png.
LOGO_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets", "comet-logo.png"
)

st.set_page_config(page_title="Comet Panel Studio", page_icon=LOGO_PATH, layout="wide")

# Less empty space above the page, and a code editor's look for the code box:
# as tall as the window allows, monospace, with long lines scrolling instead of
# wrapping.
st.markdown(
    """
<style>
[data-testid="stMainBlockContainer"] {
  padding-top: 1.5rem;
}
.st-key-code textarea {
  /* Fill the window down to the Run button, whatever the window's size. */
  height: calc(100vh - 410px) !important;
  min-height: 240px;
  font-family: "Source Code Pro", ui-monospace, Menlo, Consolas, monospace;
  font-size: 13px;
  white-space: pre;
  overflow-wrap: normal;
  overflow-x: auto;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_api(url_override, api_key, cache=True):
    """An API for this server and key. Uploads pass cache=False: they read a
    dashboard or the Panel library and write it back, so they need current
    data, and some comet_ml versions cache those reads."""
    # comet_ml reads the server from the environment when an API is created.
    os.environ["COMET_URL_OVERRIDE"] = url_override
    return API(api_key=api_key, cache=cache)


@st.cache_data(show_spinner=False, ttl=60)
def list_workspaces(url_override, api_key):
    return sorted(get_api(url_override, api_key).get_workspaces())


@st.cache_data(show_spinner=False, ttl=60)
def default_workspace(url_override, api_key):
    return get_api(url_override, api_key).get_default_workspace()


@st.cache_data(show_spinner=False, ttl=60)
def list_projects(url_override, api_key, workspace):
    results = get_api(url_override, api_key)._client.get_project_jsons(workspace)
    return [
        (p["projectName"], p["projectName"], p.get("lastUpdated") or 0)
        for p in (results or {}).get("projects", [])
    ]


@st.cache_data(show_spinner=False, ttl=60)
def list_experiments(url_override, api_key, workspace, project):
    experiments = get_api(url_override, api_key).get_experiments(workspace, project)
    # The later of start and end: when a run last changed, even if still running.
    return [
        (
            e.id,
            e.name or e.id[:9],
            max(e.start_server_timestamp or 0, e.end_server_timestamp or 0),
        )
        for e in experiments
    ]


def read_file(path):
    try:
        with open(path) as fp:
            return fp.read()
    except OSError:
        return None


def init_state():
    if "code" in st.session_state:
        return
    path = display_path(os.environ.get("COMETX_PANEL_STUDIO_FILE", ""), os.getcwd())
    st.session_state.file_path = path
    st.session_state.code = (read_file(path) if path else None) or NEW_PANEL_CODE
    st.session_state.page_name = new_page_name()
    st.session_state.run_count = 0
    st.session_state.panel_name = (
        os.path.splitext(os.path.basename(path))[0] if path else "My panel"
    )


def load_file():
    code = read_file(st.session_state.file_path)
    if code is None:
        st.session_state.file_message = ("error", "Could not read that file.")
    else:
        st.session_state.code = code
        st.toast("Loaded %s" % os.path.basename(st.session_state.file_path))


def file_chosen():
    """A path was typed into the File box: open it if it exists. A new path
    stays empty until Save creates it."""
    path = st.session_state.file_path
    if not path:
        return
    st.session_state.panel_name = os.path.splitext(os.path.basename(path))[0]
    if os.path.exists(path):
        load_file()


def open_browser():
    """Start the file browser in the current file's folder, or the folder the
    studio was started in."""
    path = st.session_state.file_path
    folder = os.path.dirname(os.path.abspath(path)) if path else os.getcwd()
    st.session_state.browse_dir = folder if os.path.isdir(folder) else os.getcwd()


def browse_to(folder):
    st.session_state.browse_dir = os.path.normpath(folder)


def browse_open(path):
    st.session_state.opened_file = path


@st.dialog("Open a file", width="large")
def browse_dialog():
    st.text_input("Folder", key="browse_dir")
    folder = st.session_state.browse_dir
    try:
        folders, files = list_directory(folder)
    except OSError as error:
        st.error("Can't open %s: %s" % (folder, error))
        folders, files = [], []

    with st.container(height=360):
        parent = os.path.dirname(os.path.abspath(folder))
        if parent != os.path.abspath(folder):
            st.button("📁 ..", type="tertiary", on_click=browse_to, args=(parent,))
        for name in folders:
            st.button(
                "📁 " + name,
                key="dir:" + name,
                type="tertiary",
                on_click=browse_to,
                args=(os.path.join(folder, name),),
            )
        for name in files:
            st.button(
                "📄 " + name,
                key="file:" + name,
                type="tertiary",
                on_click=browse_open,
                args=(os.path.join(folder, name),),
            )
        if not folders and not files:
            st.caption("No folders or .py files here.")

    if st.session_state.get("opened_file"):
        st.rerun()


def apply_opened_file():
    """Open the file chosen in the browser. Done before the File box is drawn,
    since a widget's value can't change after it is."""
    path = st.session_state.pop("opened_file", None)
    if path:
        st.session_state.file_path = display_path(path, os.getcwd())
        st.session_state.panel_name = os.path.splitext(os.path.basename(path))[0]
        load_file()


def save_file():
    try:
        with open(st.session_state.file_path, "w") as fp:
            fp.write(st.session_state.code)
        st.toast("Saved %s" % os.path.basename(st.session_state.file_path))
    except OSError as error:
        st.session_state.file_message = ("error", "Could not save: %s" % error)


def sort_toggle(label, key):
    return st.sidebar.radio(
        label,
        ["updated", "name"],
        format_func={"updated": "Recently updated", "name": "Name"}.get,
        horizontal=True,
        key=key,
    )


def pick_index(options, wanted):
    return options.index(wanted) if wanted in options else 0


def sidebar():
    """The panel's context: what Comet would hand it on a dashboard."""
    st.sidebar.header("Comet")
    server = st.sidebar.text_input(
        "Server",
        value=server_root(get_config("comet.url_override") or "https://www.comet.com"),
        help="Your Comet server's address, e.g. https://www.comet.com",
    )
    url_override = clientlib_url(server)
    api_key = st.sidebar.text_input(
        "API key", value=get_config("comet.api_key") or "", type="password"
    )
    if not api_key:
        st.sidebar.info("Enter your API key to continue.")
        return None

    try:
        workspaces = list_workspaces(url_override, api_key)
    except Exception as error:
        st.sidebar.error("Could not connect: %s" % error)
        return None

    workspace = st.sidebar.selectbox(
        "Workspace",
        workspaces,
        index=pick_index(
            workspaces,
            os.environ.get("COMETX_PANEL_STUDIO_WORKSPACE")
            or get_config("comet.workspace")
            or default_workspace(url_override, api_key),
        ),
    )
    projects = list_projects(url_override, api_key, workspace)
    if not projects:
        st.sidebar.info("This workspace has no projects.")
        return None
    projects_by = sort_toggle("Sort projects by", "projects_by")
    project_names = [name for name, _, _ in sort_entries(projects, projects_by)]
    project = st.sidebar.selectbox(
        "Project",
        project_names,
        index=pick_index(project_names, os.environ.get("COMETX_PANEL_STUDIO_PROJECT")),
        key="project:%s" % workspace,
    )
    st.sidebar.markdown(
        "[Open in Comet ↗](%s)" % project_url(server, workspace, project)
    )

    st.sidebar.header("Panel")
    experiments_by = sort_toggle("Sort experiments by", "experiments_by")
    experiments = sort_entries(
        list_experiments(url_override, api_key, workspace, project), experiments_by
    )
    names = {key: name for key, name, _ in experiments}
    experiment_keys = st.sidebar.multiselect(
        "Experiments",
        [key for key, _, _ in experiments],
        default=default_selection([key for key, _, _ in experiments]),
        format_func=lambda key: names[key],
        key="experiments:%s/%s" % (workspace, project),
        help=(
            "The experiments the panel sees. Like a Comet dashboard, which "
            "shows a panel the first page of its experiments table, the first "
            "25 are selected to start."
        ),
    )

    return {
        "url_override": url_override,
        "api_key": api_key,
        "workspace": workspace,
        "project": project,
        "experiment_keys": experiment_keys,
    }


@st.dialog("Upload panel")
def upload_dialog(context):
    api = get_api(context["url_override"], context["api_key"], cache=False)
    workspace, project = context["workspace"], context["project"]
    code = st.session_state.code

    name = st.text_input("Panel name", key="panel_name")
    where = st.radio(
        "Upload to",
        ["dashboard", "library"],
        format_func=lambda choice: {
            "dashboard": "A dashboard in %s" % project,
            "library": "The Panel library of %s" % workspace,
        }[choice],
    )
    if where == "dashboard":
        dashboards = sorted(
            v.name
            for v in api.get_views(workspace, project)
            if v.name and v.name != "Unsaved Changes"
        )
        choice = st.selectbox("Dashboard", [NEW_DASHBOARD] + dashboards)
        if choice == NEW_DASHBOARD:
            dashboard = st.text_input("New dashboard name", value="Panel Studio")
        else:
            dashboard = choice
        st.caption(
            "A panel with the same name on that dashboard is replaced; "
            "otherwise the panel is added to its Panel Studio section."
        )
    else:
        st.caption(
            "Library panels can be added to dashboards from Add panel > "
            "Panel library in Comet."
        )

    if not st.button("Upload", type="primary", disabled=not name):
        return
    try:
        if where == "dashboard":
            view, created = upload_to_dashboard(
                api, workspace, project, dashboard, name, code
            )
            url = dashboard_url(api.server_url, workspace, project, view.template_id)
            st.success(
                "%s [%s](%s)." % ("Created" if created else "Updated", dashboard, url)
            )
        else:
            template_id, created = upload_to_library(api, workspace, name, code)
            st.success(
                "%s %r in the Panel library (%s)."
                % ("Added" if created else "Updated", name, template_id)
            )
    except Exception as error:
        st.error("Upload failed: %s" % error)


def show_iframe(url, height):
    # st.iframe replaced components.iframe in newer Streamlit versions.
    if hasattr(st, "iframe"):
        st.iframe(url, height=height)
    else:
        components.iframe(url, height=height, scrolling=True)


def panel_url(page_name, run_count):
    # The panel server runs beside the editor, on another port.
    host = st.context.headers.get("Host", "localhost").split(":")[0]
    return "http://%s:%s/%s?embed=true&run=%d" % (
        host,
        PANEL_PORT,
        page_name,
        run_count,
    )


def button_row(count):
    """Places for buttons side by side, each as wide as its label."""
    try:
        row = st.container(horizontal=True)
    except TypeError:  # Streamlit before horizontal containers
        return st.columns(count)
    return [row] * count


def title_with_logo():
    with open(LOGO_PATH, "rb") as fp:
        logo = base64.b64encode(fp.read()).decode()
    st.markdown(
        """
<div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.75rem;">
  <img src="data:image/png;base64,%s" alt="Comet" style="height: 2.6rem;">
  <h2 style="margin: 0; padding: 0;">Comet Panel Studio</h2>
</div>
"""
        % logo,
        unsafe_allow_html=True,
    )


def main():
    init_state()
    apply_opened_file()
    context = sidebar()

    title_with_logo()
    editor, panel = st.columns(2, gap="medium")

    with editor:
        st.text_input(
            "File",
            key="file_path",
            placeholder="Type a path, or Browse",
            on_change=file_chosen,
        )
        browse_col, load_col, save_col = button_row(3)
        if browse_col.button("Browse...", on_click=open_browser):
            browse_dialog()
        load_col.button(
            "Load",
            on_click=load_file,
            disabled=not st.session_state.file_path,
        )
        save_col.button(
            "Save",
            on_click=save_file,
            disabled=not st.session_state.file_path,
        )
        message = st.session_state.pop("file_message", None)
        if message:
            getattr(st, message[0])(message[1])

        st.text_area("Code", key="code", height=380, label_visibility="collapsed")

        run_col, upload_col = button_row(2)
        run = run_col.button("Run", type="primary", disabled=context is None)
        if upload_col.button("Upload...", disabled=context is None):
            upload_dialog(context)

    if run and context is not None:
        cfg = panel_config(
            context["url_override"],
            context["api_key"],
            context["workspace"],
            context["project"],
            context["experiment_keys"],
            # Comet always gives a dashboard's Python panel no options.
            {},
            None,
            None,
            st.session_state.page_name,
        )
        write_panel_page(
            st.session_state.code, PAGES_DIR, st.session_state.page_name, cfg
        )
        st.session_state.run_count += 1

    with panel:
        if st.session_state.run_count and context is not None:
            show_iframe(
                panel_url(st.session_state.page_name, st.session_state.run_count),
                PANEL_HEIGHT,
            )


main()
