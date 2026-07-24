# Panel and Dashboard Enhancements

This document covers the new and updated methods in `cometx.api.API` for managing
code panels and dashboards programmatically, plus the `cometx.views` module for
building dashboard layouts.

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

## Building Dashboard Layouts: `cometx.views`

`create_dashboard` and `update_dashboard` manage the dashboard *record* — its name,
its ID, and which code panels are attached to it. They do not describe what the
dashboard looks like. The layout itself — sections, charts, axes, filters, table
columns — lives in the dashboard's serialized state, and `cometx.views` builds that
state.

The two halves fit together like this:

| To do this | Use |
|---|---|
| Attach uploaded code panels to a dashboard | `API.create_dashboard` / `API.update_dashboard` |
| Lay out built-in charts, sections, filters | `cometx.views` + `API.create_view` |

`cometx.views` makes no network calls. It returns a `comet_ml` `View` object that you
pass to `API.create_view(workspace, project_name, view)` for a project dashboard, or
to `APIExperiment.create_view(view)` for a single experiment's own page.

### Quick start

```python
from cometx import API
from cometx.views import Section, build_view, line_panel, scalar_panel

view = build_view(
    "Training Overview",
    sections=[
        Section(
            "At a glance",
            [
                scalar_panel("accuracy", aggregation="max"),
                scalar_panel("loss", aggregation="min"),
            ],
            columns=2,
        ),
        Section(
            "Curves",
            [
                line_panel(["loss", "val_loss"], name="Loss", smoothing=0.6),
                line_panel(["accuracy", "val_accuracy"], name="Accuracy"),
            ],
            columns=2,
        ),
    ],
)

API().create_view("my-workspace", "my-project", view)
```

The new dashboard appears in the view selector at the top of the project page.

---

### Panel factories

Each factory returns a panel dict with a freshly generated `chartId`. Build a new
view object per target rather than reusing one across projects or experiments.

| Factory | Produces |
|---|---|
| `line_panel(y, x="step", smoothing=0, transform_y=None, locked=False, ...)` | Line chart; `y` may be a list of metrics to overlay |
| `bar_panel(metric, aggregation="last", plot_type="BAR", group_by_aggregation=None, ...)` | Bar, box, or violin plot, one bar per experiment |
| `scatter_panel(x, y, z=None, metrics=None, params=None, ...)` | 2D or 3D scatter, one point per experiment |
| `parallel_panel(params, metrics=(), target="", ...)` | Parallel-coordinates hyperparameter sweep |
| `scalar_panel(metric, aggregation="last", precision=3, ...)` | Single big-number tile |
| `image_panel(images=(), step=None, ...)` | Image grid across experiments |
| `data_panel(file_name, axis=0, join_type="outer", ...)` | Table from a logged tabular asset |
| `custom_panel(instance_id, instance_name="", ...)` | An instance of a JavaScript panel |
| `python_panel(template_id, revision_id="", ...)` | An instance of a Python panel |

Axis and legend helpers: `legend_key(value, label=None, source="log_other")` builds one
entry of a chart's legend, so a chart can be colored by experiment name, tags, or a
hyperparameter.

> **Note:** Built-in panels can be written from nothing. `custom_panel` and
> `python_panel` reference a panel instance that already exists, so their ids cannot
> be invented — read them off a panel already on a dashboard via `get_views()`, or
> from the panel's URL in the UI.

---

### `Section(title, panels=(), columns=3, expanded=True, height=1)`

A titled, collapsible group of panels. The dashboard grid is 6 units wide, so
`columns` must divide 6 evenly (1, 2, 3, or 6); `Section` computes every panel's
grid position for you. `height` is the row height in grid units (max 3).

```python
from cometx.views import Section, image_panel, data_panel

section = Section("Media & tables", columns=2, height=2, expanded=False)
section.add(image_panel(["prediction-grid.png"]), data_panel("per_class_metrics.csv"))
```

`add()` returns the section, so calls can be chained.

---

### `build_view(name, sections, filters=None, table=None, config=None, auto_generate=False, personal=False)`

Assemble sections into a project dashboard `View`.

| Parameter | Type | Description |
|---|---|---|
| `name` | str | The dashboard's display name |
| `sections` | list | `Section` objects, top to bottom |
| `filters` | str (optional) | Output of `query_state()` — a saved filter |
| `table` | str (optional) | Output of `table_state()` — experiment table columns and sorting |
| `config` | dict (optional) | Output of `global_config()` — dashboard-wide x-axis, smoothing, sampling |
| `auto_generate` | bool | `True` lets Comet keep appending auto-generated panels for newly logged metrics below your sections |
| `personal` | bool | Mark the dashboard as visible only to you |

Filters are built from `rule()`, `all_of()`, and `any_of()`:

```python
from cometx.views import all_of, any_of, global_config, query_state, rule, table_state

view = build_view(
    "Successful runs",
    sections=[...],
    # accuracy > 0.9 OR loss < 0.1
    filters=query_state(
        any_of(
            all_of(rule("accuracy", "greater", 0.9)),
            all_of(rule("loss", "less", 0.1)),
        )
    ),
    table=table_state(
        columns=["Name", "experimentTags", "learning_rate", "accuracy", "duration"],
        sort_by="accuracy",
        page_size=50,
    ),
    config=global_config(x_axis="step", smoothing=0.3, sample_size=500),
)
```

`global_config` applies to every panel that is not `locked=True`; a locked panel keeps
its own axis and smoothing settings instead.

---

### `build_experiment_view(name, sections, x_axis="step", smoothing=0, ...)`

Assemble the *same* sections into a view for one experiment's own page, then pass it to
`APIExperiment.create_view(view)`.

```python
from comet_ml import API as CometAPI
from cometx.views import Section, build_experiment_view, line_panel

experiment = CometAPI().get_experiments("my-workspace", "my-project")[0]
view = build_experiment_view(
    "Run layout",
    sections=[Section("Curves", [line_panel(["loss", "val_loss"])], columns=2)],
    smoothing=0.4,
)
experiment.create_view(view)
```

Project dashboards and experiment views use different storage envelopes — project state
lives in `view.v3`, experiment state under `view.v2["panels"]` — which is why there are
two builders. Sections and panels are identical between them. Do not set
`experiment_key` yourself; `create_view` stamps it on for you.

> **Note:** Passing a `build_view()` result to `APIExperiment.create_view` is accepted
> by the backend but comes back with no panels.

---

### `describe_view(view, indent="")`

Render a human-readable outline of a view's sections, panels, and filters — useful for
inspecting what `get_views()` returned before copying or rewriting it.

```python
from cometx.views import describe_view

for view in API().get_views("my-workspace", "my-project"):
    print(describe_view(view))
```

---

### Gotchas

- **`create_view` always inserts.** Calling it twice with the same name yields two
  dashboards, and there is no `delete_view`. Updating in place requires the existing
  `template_id`, which `create_view` cannot carry — it calls `as_portable()`, which
  deliberately clears it. Use the lower-level upsert instead:

  ```python
  view.template_id = existing_view.template_id
  api._client.upsert_view(api.get_project(workspace, project)["projectId"], view)
  ```

- **Panels that reference metrics the project never logged render empty**, not as
  errors. When applying one layout across many projects, read each project's metric
  names first and build only the panels that will have data.

- **`as_portable()` is what makes a view movable** between projects, workspaces, or
  accounts. It strips `template_id`, `project_id`, `experiment_key`,
  `pinned_experiments`, and the `experimentKey` embedded in `v2`/`v3`. It does *not*
  rewrite experiment keys inside `chart_state`, so a panel pinned to a specific run
  needs manual fixing.

- **`"Unsaved Changes"` views** are the UI's per-user scratch state for a project. They
  show up in `get_views()` results; skip them when copying.

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
