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
The parts of `cometx panel-studio` that don't need Streamlit: the panel
context Comet hands a Python panel, writing a panel as a page the way Comet's
panel server does, and putting a panel's code on a dashboard.
"""

import copy
import os
from typing import Any, Dict, List, Optional, Tuple

from .panel_engine import copy_user_code
from .views import Section, build_view, new_id, python_panel

# The frontend's chart colors (frontend constants/colorConstants.ts), used for
# experiments that have no color of their own.
CHART_COLORS = [
    "#fb9341",
    "#5899da",
    "#19a979",
    "#ed4a7b",
    "#f4b400",
    "#945fcf",
    "#12a4b4",
    "#525df4",
    "#bf399e",
    "#6c8893",
    "#ef6868",
    "#2e6497",
]

# A dashboard's experiments table shows this many per page, and a panel sees
# that page (frontend DEFAULT_EXPERIMENTS_PER_PAGE).
DASHBOARD_PAGE_SIZE = 25

# Where panels added from the studio go on an existing dashboard.
STUDIO_SECTION = "Panel Studio"

PANEL_ROW_HEIGHT = 2


def lighten_color(color: str, percent: float = 0.8) -> str:
    """The frontend's lightenColor: mix a "#rrggbb" color toward white."""
    value = int(color[1:], 16)
    target = 0 if percent < 0 else 255
    percent = abs(percent)
    channels = [(value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF]
    mixed = [round((target - c) * percent) + c for c in channels]
    return "#%02x%02x%02x" % tuple(mixed)


def experiment_colors(experiment_keys: List[str]) -> Dict[str, Dict[str, str]]:
    """The color map Comet passes to a panel: the frontend's round-robin rule,
    which picks a chart color by the last hex digit of the experiment key."""
    colors = {}
    for key in experiment_keys:
        primary = CHART_COLORS[int(key[-1], 16) % len(CHART_COLORS)]
        colors[key] = {
            "experimentKey": key,
            "primary": primary,
            "light": lighten_color(primary),
        }
    return colors


def panel_config(
    url_override: str,
    api_key: str,
    workspace: str,
    project: str,
    experiment_keys: List[str],
    options: Dict[str, Any],
    width: Optional[int],
    height: Optional[int],
    instance_id: str,
) -> Dict[str, Any]:
    """The config Comet's panel server injects into a panel's page."""
    return {
        "comet.url_override": url_override,
        "comet.api_key": api_key,
        "comet.workspace": workspace,
        "comet.project_name": project,
        "COMET_EXPERIMENT_KEYS": experiment_keys,
        "COMET_PANEL_OPTIONS": options,
        "COMET_PANEL_WIDTH": width,
        "COMET_PANEL_HEIGHT": height,
        "COMET_PANEL_EXPERIMENT_COLORS": experiment_colors(experiment_keys),
        "COMET_PANEL_METRIC_COLORS": None,
        "COMET_PANEL_INSTANCE_ID": instance_id,
    }


def write_panel_page(code: str, pages_dir: str, page_name: str, cfg: Dict[str, Any]):
    """Write the panel as pages_dir/PAGE_NAME.py, as Comet's panel server does.

    The page name must start with "page_": comet_ml finds the page's config by
    that prefix.
    """
    if not page_name.startswith("page_"):
        raise ValueError("a panel page name must start with 'page_'")
    work_dir = os.path.dirname(os.path.abspath(pages_dir))
    path = os.path.join(pages_dir, page_name + ".py")
    copy_user_code(
        code,
        path,
        cfg,
        cache_dir=os.path.join(work_dir, "include-cache"),
        lock_path=os.path.join(work_dir, "pip.lock"),
    )
    return path


def server_root(url: str) -> str:
    """A Comet server's address without a trailing /clientlib or slashes."""
    root = url.strip().rstrip("/")
    while root.endswith("/clientlib"):
        root = root[: -len("/clientlib")].rstrip("/")
    return root


def clientlib_url(url: str) -> str:
    """The client URL comet_ml wants for a server: ROOT/clientlib/, with
    exactly one slash on each side, however the address was typed."""
    return server_root(url) + "/clientlib/"


def project_url(url: str, workspace: str, project: str) -> str:
    return "%s/%s/%s" % (server_root(url), workspace, project)


def dashboard_url(server_url: str, workspace: str, project: str, view_id: str) -> str:
    return "%s/%s/%s/view/%s/panels" % (
        server_url.rstrip("/"),
        workspace,
        project,
        view_id,
    )


def _find_python_panel(v3: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    for section in v3.get("sections") or []:
        for panel in section.get("panels") or []:
            if panel.get("chartType") == "python" and panel.get("chartName") == name:
                return panel
    return None


def put_panel_in_layout(
    v3: Dict[str, Any], name: str, code: str, columns: int = 1
) -> Dict[str, Any]:
    """Return a copy of a dashboard's v3 layout with this Python panel in it.

    A Python panel with the same name has its code replaced, wherever it is.
    Otherwise the panel is added at the bottom of a "Panel Studio" section,
    which is created if needed. Nothing else on the dashboard changes.

    Args:
        v3: the dashboard's ``view.v3``.
        name: the panel's title, which identifies it on later uploads.
        code: the panel's Python source.
        columns: how many such panels fit across the dashboard (1, 2, 3 or 6).
    """
    v3 = copy.deepcopy(v3)
    existing = _find_python_panel(v3, name)
    replacement = python_panel(code=code, name=name)
    if existing is not None:
        for field in ("code", "codeVersion", "templateId", "revisionId"):
            existing[field] = replacement[field]
        return v3

    sections = v3.setdefault("sections", [])
    section = next((s for s in sections if s.get("title") == STUDIO_SECTION), None)
    if section is None:
        section = Section(STUDIO_SECTION).to_dict()
        sections.append(section)
    section.setdefault("panels", []).append(replacement)
    layout = section.setdefault("layout", [])
    bottom = max((item["y"] + item["h"] for item in layout), default=0)
    layout.append(
        {
            "i": replacement["chartId"],
            "x": 0,
            "y": bottom,
            "w": 6 // columns,
            "h": PANEL_ROW_HEIGHT,
            # As Section.layout() writes them:
            "minW": 2,
            "maxH": 3,
            "moved": False,
            "static": False,
        }
    )
    return v3


def upload_to_dashboard(
    api, workspace: str, project: str, dashboard: str, name: str, code: str
):
    """Put the panel on a project dashboard, creating the dashboard if needed.

    Returns (view, created) where created is True for a new dashboard.
    """
    existing = [v for v in api.get_views(workspace, project) if v.name == dashboard]
    if not existing:
        view = build_view(
            dashboard,
            sections=[
                Section(
                    STUDIO_SECTION,
                    [python_panel(code=code, name=name)],
                    columns=1,
                    height=PANEL_ROW_HEIGHT,
                )
            ],
        )
        return api.create_view(workspace, project, view), True

    view = existing[0]
    if not isinstance(view.v3, dict):
        raise ValueError(
            "Dashboard %r is saved in an older format this tool can't edit. "
            "Open it in Comet once to update it, or choose another dashboard."
            % dashboard
        )
    view.v3 = put_panel_in_layout(view.v3, name, code)
    return api.update_view(workspace, project, view), False


def upload_to_library(api, workspace: str, name: str, code: str):
    """Save the panel to the workspace's Panel library, replacing one with the
    same name. Returns (template_id, created)."""
    existing = [p for p in api.get_panels(workspace) if p.get("templateName") == name]
    template_id = existing[0]["templateId"] if existing else None
    result = api.upload_panel_code(workspace, name, code, template_id=template_id)
    return result["templateId"], template_id is None


def sort_entries(
    entries: List[Tuple[str, str, int]], by: str
) -> List[Tuple[str, str, int]]:
    """Order (id, name, updated) entries, such as projects or experiments.

    by: "name" (A to Z, ignoring case) or "updated" (most recently updated
    first; updated is a time in ms).
    """
    if by == "name":
        return sorted(entries, key=lambda e: (e[1].lower(), e[0]))
    if by == "updated":
        return sorted(entries, key=lambda e: (-(e[2] or 0), e[1].lower()))
    raise ValueError("sort by 'name' or 'updated', not %r" % (by,))


def default_selection(experiment_keys: List[str]) -> List[str]:
    """The experiments a dashboard's panel sees when it opens: the first page
    of its experiments table."""
    return experiment_keys[:DASHBOARD_PAGE_SIZE]


def list_directory(path: str) -> Tuple[List[str], List[str]]:
    """The folders and .py files in a folder, for the studio's file browser.

    Returns (folders, files), each sorted by name; hidden entries are left out.
    Raises OSError when the folder can't be read.
    """
    folders, files = [], []
    for entry in os.scandir(path):
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            folders.append(entry.name)
        elif entry.name.endswith(".py"):
            files.append(entry.name)
    return sorted(folders, key=str.lower), sorted(files, key=str.lower)


def display_path(path: str, root: str) -> str:
    """A path relative to root when it is inside root, else unchanged."""
    if not path:
        return path
    relative = os.path.relpath(os.path.abspath(path), os.path.abspath(root))
    return path if relative.startswith("..") else relative


def new_page_name() -> str:
    return "page_" + new_id(10)
