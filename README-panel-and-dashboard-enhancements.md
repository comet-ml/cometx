# Panel and Dashboard Enhancements

This document covers the new and updated methods in `cometx.api.API` for managing
code panels and dashboards programmatically.

## Uploading a Panel

### `upload_panel_code(workspace, panel_name, code, template_id=None)`

Upload Python code as a panel in a workspace. If `template_id` is provided, the
existing panel is **overwritten in place** rather than creating a new one.

```python
from cometx import API

api = API()

# Create a new panel
result = api.upload_panel_code("my-workspace", "My Panel", "print('hello world')")
panel_id = result["templateId"]

# Update an existing panel's code in place
result = api.upload_panel_code("my-workspace", "My Panel", "print('updated')", template_id=panel_id)
```

Returns a dict with the panel's `templateId`.

---

### `upload_panel_zip(workspace, filename, template_id=None)`

Upload a panel zip file. If `template_id` is provided, the existing panel is
**overwritten in place** rather than creating a new one. Dashboards that reference
the panel will reflect the updated code after a browser refresh.

```python
from cometx import API
from cometx.panel_utils import create_panel_zip

api = API()

# Create a new panel
result = api.upload_panel_zip("my-workspace", "panel.zip")

# Update an existing panel in place
result = api.upload_panel_zip("my-workspace", "panel.zip", template_id="abc123")
```

> **Note:** After updating a panel in place, you need to **refresh the browser page**
> to see the updated panel code in the Comet UI.

---

## Creating a Dashboard

### `create_dashboard(workspace, project_name, template_name, template_id=None, panels=None)`

Create a new dashboard in a project.

```python
from cometx import API

api = API()

# Create an empty dashboard
dashboard = api.create_dashboard(
    workspace="my-workspace",
    project_name="my-project",
    template_name="My Dashboard",
)

# Create a dashboard with panels
dashboard = api.create_dashboard(
    workspace="my-workspace",
    project_name="my-project",
    template_name="My Dashboard",
    panels=["panel-template-id-1", "panel-template-id-2"],
)

# Create a dashboard cloned from an existing one (copies panel associations)
dashboard = api.create_dashboard(
    workspace="my-workspace",
    project_name="my-project",
    template_name="My Dashboard Copy",
    template_id="existing-dashboard-template-id",
)
```

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `workspace` | str | The workspace name the project belongs to |
| `project_name` | str | The project name to create the dashboard in |
| `template_name` | str | Name for the new dashboard |
| `template_id` | str (optional) | Source dashboard template ID to clone panel associations from |
| `panels` | list (optional) | List of panel `templateId` values to include in the new dashboard |

Returns a dict representing the created `DashboardTemplate`, including its new `template_id`.

---

## Updating a Dashboard

### `update_dashboard(workspace, project_name, template_id, template_name=None, panels=None)`

Update an existing dashboard in a project. Optionally rename it, replace its panel
associations, or both.

```python
from cometx import API

api = API()

# Rename a dashboard
api.update_dashboard(
    workspace="my-workspace",
    project_name="my-project",
    template_id="existing-dashboard-id",
    template_name="Renamed Dashboard",
)

# Replace the panels on a dashboard
api.update_dashboard(
    workspace="my-workspace",
    project_name="my-project",
    template_id="existing-dashboard-id",
    panels=["panel-template-id-1", "panel-template-id-2"],
)

# Rename and update panels at once
api.update_dashboard(
    workspace="my-workspace",
    project_name="my-project",
    template_id="existing-dashboard-id",
    template_name="Renamed Dashboard",
    panels=["panel-template-id-1", "panel-template-id-2"],
)
```

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `workspace` | str | The workspace name the project belongs to |
| `project_name` | str | The project name the dashboard belongs to |
| `template_id` | str | The ID of the dashboard to update |
| `template_name` | str (optional) | New name for the dashboard |
| `panels` | list (optional) | List of panel `templateId` values to set on the dashboard |

Returns a dict representing the updated `DashboardTemplate`.

---

## Typical Workflow

```python
from cometx import API

api = API()

# 1. Upload a panel
result = api.upload_panel_code("my-workspace", "Hello World", "print('hello world')")
panel_id = result["templateId"]

# 2. Create a dashboard with the panel
dashboard = api.create_dashboard(
    workspace="my-workspace",
    project_name="my-project",
    template_name="My Dashboard",
    panels=[panel_id],
)
dashboard_id = dashboard["template_id"]

# 3. Later, update the panel code in place
api.upload_panel_code("my-workspace", "Hello World", "print('updated')", template_id=panel_id)
# Refresh your browser to see the updated panel in the Comet UI

# 4. Add more panels to the dashboard
api.update_dashboard(
    workspace="my-workspace",
    project_name="my-project",
    template_id=dashboard_id,
    panels=[panel_id, "another-panel-id"],
)
```
