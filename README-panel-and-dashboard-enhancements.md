# Panel and Dashboard Enhancements

This document covers the methods in `cometx.api.API` for managing code panels and
dashboards programmatically, plus the `cometx.views` module for building dashboard
layouts. They work with any recent `comet_ml`.

| To do this | Use |
|---|---|
| Lay out built-in charts, code panels, sections, filters | `cometx.views` + `API.create_view` / `API.update_view` |
| Lay out an experiment's own page | `cometx.views.build_experiment_view` + `API.create_experiment_view` / `API.update_experiment_view` |
| Put Python code on a dashboard | `cometx.views.python_panel(code=...)` in a view |
| Manage the workspace's Panel library | `API.upload_panel_code`, `upload_panel_zip`, `get_panels`, `get_panel`, `get_panel_code`, `download_panel_zip` |
| Develop panels locally | `cometx panel-studio`, `panel-preview`, `panel-upload`, `panel-download` ([PANEL-STUDIO.md](PANEL-STUDIO.md)) |

For runnable end-to-end scripts, see [`./examples/`](./examples/) — summarized in
[Runnable Examples](#runnable-examples) below.

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

> **Note:** these are **Panel library** panels. Placing one on a dashboard creates a
> panel instance, which only the Comet UI can do (Add panel > Panel library), so an
> API key can upload a library panel but not lay it out. To put its code on a
> dashboard, use a `python_panel` with the code:
> `python_panel(code=api.get_panel_code(panel_id), name="My Panel")`.

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

### Reading the Panel library

* `get_panels(workspace)`: every library panel's metadata (`templateId`,
  `templateName`, ...).
* `get_panel(panel_id)` / `get_panel_code(panel_id)`: a panel's full data, or just
  its code.
* `download_panel_zip(panel_id, filename=None)`: save a panel as a zip file.

---

## Creating a Dashboard

> **Note:** `create_dashboard` and `update_dashboard` manage a dashboard's *record*:
> its name, its ID, and which library panels are associated with it. Associating a
> panel does **not** put it on the dashboard; a dashboard created with `panels=[...]`
> opens empty. To lay out a dashboard, including Python panels, build it with
> `cometx.views` and save it with `API.create_view` (see below).

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

A dashboard's layout (sections, charts, code panels, axes, filters, table
columns) lives in its serialized state, and `cometx.views` builds that state.

`cometx.views` makes no network calls. It returns a `comet_ml` `View` object that you
pass to `API.create_view(workspace, project_name, view)` for a project dashboard, or
`build_experiment_view()` returns one for `API.create_experiment_view(experiment,
view)`, a layout for experiments' own pages.

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
| `bar_panel(metric, aggregation="last", plot_type="BAR", group_by=None, ...)` | Bar, stacked bar, box, or violin plot; one bar per experiment, or one per value of the `group_by` hyperparameter |
| `scatter_panel(x, y, z=None, metrics=None, params=None, ...)` | 2D or 3D scatter, one point per experiment |
| `parallel_panel(params, metrics=(), target="", ...)` | Parallel-coordinates hyperparameter sweep |
| `scalar_panel(metric, aggregation="last", precision=3, ...)` | Single big-number tile |
| `image_panel(images=(), step=None, ...)` | Image grid across experiments |
| `data_panel(file_name, axis=0, join_type="outer", ...)` | Table from a logged tabular asset |
| `python_panel(code=..., name=...)` | A Python panel whose code is stored in the view |
| `javascript_panel(code, html=..., css="", resources=(), name=...)` | A JavaScript panel whose code is stored in the view (its class must be named `MyPanel`) |
| `custom_panel(instance_id, instance_name="", ...)` | A Panel library panel already placed on a dashboard in the UI |
| `python_panel(template_id, revision_id="", ...)` | A Python panel added in the UI |

Axis and legend helpers: `legend_key(value, label=None, source="log_other")` builds one
entry of a chart's legend, so a chart can be colored by experiment name, tags, or a
hyperparameter.

Option values are checked, because the Comet UI ignores a value it doesn't know
without any error; the factories map these names to the UI's values and raise
`ValueError` on anything else:

| Option | Values |
|---|---|
| `transform_y` | `None`, `"log"` (log-scale axis), `"ln"` / `"log10"` (plot the log of each value), `"moving average"` |
| `outliers` | `"show"`, `"hide"` |
| `aggregation` | `"last"`, `"first"`, `"min"`, `"max"`, `"average"`, `"median"`, `"sum"`, `"count"`, `"mode"`, `"rms"`, `"stddev"` |
| bar `plot_type` | `"BAR"`, `"STACKED_BAR"`, `"BOX"`, `"VIOLIN"` |
| bar `group_by_aggregation` | `"mean"`, `"median"`, `"min"`, `"max"` (with `group_by`) |

> **Note:** Built-in panels, `python_panel(code=...)` and `javascript_panel` can be
> written from nothing. `custom_panel` and `python_panel(template_id, ...)` reference
> a panel that already exists, so their ids cannot be invented — read them off a
> panel already on a dashboard via `get_views()`, or from the panel's URL in the UI.
> Panels uploaded with `upload_panel_code` are Panel library panels: an API key can
> upload them but not place them on a dashboard, so to show one's code on a
> dashboard, use `python_panel(code=api.get_panel_code(template_id))`. To write and
> test a Python panel locally first, see `cometx panel-studio`.

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

`global_config` applies to every panel that is not `locked=True`: line panels take
their x-axis, `transform_y`, smoothing, sample size and outliers from it, and scatter
and bar panels their sample size. A locked panel keeps its own. An experiment's page
applies its own settings the same way, so a log-scale line panel in either kind of
view needs `locked=True`.

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

- **`create_view` only adds views.** A project can't have two views with the same
  name: `create_view` raises a `ValueError` for a name already in use (and there is
  no `delete_view`). To replace a view in place, give the new one the existing
  view's `template_id` and call `update_view`:

  ```python
  existing = next(v for v in api.get_views(workspace, project) if v.name == "My Dashboard")
  view = build_view("My Dashboard", sections=[...])
  view.template_id = existing.template_id
  api.update_view(workspace, project, view)
  ```

  For experiment views, `update_experiment_view(experiment, view)` does the same.

- **Experiment view names are unique across all your projects**, while
  `APIExperiment.get_views()` lists only the experiment's project. A name used in
  another project is refused; `create_experiment_view` raises a `ValueError` that
  says so. One experiment view serves every experiment in its project.

- **Use the matching builder.** A `build_experiment_view()` result saved as a project
  dashboard, or a `build_view()` result saved as an experiment view, would save but
  show no panels, so `create_view`, `update_view` and the experiment-view methods
  raise a `ValueError` instead.

- **Reading right after writing.** `comet_ml`'s `API` caches reads, so reading a
  dashboard or a library panel you changed earlier in the same process can return
  the old version. Use `API(cache=False)` for read-modify-write code, or
  `api.use_cache(False)` on an existing API.

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
from cometx.views import Section, build_view, line_panel, python_panel

api = API(cache=False)  # this script reads views after writing them
workspace, project = "my-workspace", "my-project"

panel_code = """
from comet_ml import API, ui
api = API()
ui.display("%d experiments" % len(api.get_panel_experiment_keys()))
"""

# 1. Create a dashboard with a built-in chart and a Python panel
view = build_view(
    "My Dashboard",
    sections=[
        Section("Curves", [line_panel(["loss", "val_loss"])]),
        Section("Custom", [python_panel(code=panel_code, name="Experiment count")]),
    ],
)
created = api.create_view(workspace, project, view)

# 2. Later, change the panel's code: rebuild and replace the view in place
view = build_view(
    "My Dashboard",
    sections=[
        Section("Curves", [line_panel(["loss", "val_loss"])]),
        Section("Custom", [python_panel(code=panel_code + "\n# v2\n", name="Experiment count")]),
    ],
)
view.template_id = created.template_id
api.update_view(workspace, project, view)
# Refresh your browser to see the updated dashboard in the Comet UI

# 3. Share the panel's code through the workspace's Panel library too
api.upload_panel_code(workspace, "Experiment count", panel_code)
```

To write a Python panel in your own editor and see it run as you save, then upload it
to a dashboard, see [PANEL-STUDIO.md](PANEL-STUDIO.md).

---

## Runnable Examples

The [`./examples/`](./examples/) directory holds eight scripts that exercise everything
above against a live workspace. See [`examples/README.md`](./examples/README.md) for the
full walkthrough. Scripts 04, 05 and 06 are dry-run by default; pass `--apply` to write.

| Script | Demonstrates |
| --- | --- |
| [`00_seed_demo_project.py`](./examples/00_seed_demo_project.py) | Logs six small runs — metrics, params, an image and a CSV table — so every panel type in the later scripts has real data to draw. |
| [`01_inspect_views.py`](./examples/01_inspect_views.py) | Reading views in both scopes (`API.get_views` and `APIExperiment.get_views`), how `view_source` distinguishes dashboard from chart templates, and what the `include_workspace_views` flags widen to. `--json` dumps every field with the serialized blobs unpacked. |
| [`02_build_project_dashboard.py`](./examples/02_build_project_dashboard.py) | The widest `cometx.views` demo: scalar tiles, multi-metric line charts, log-scale axes, a legend keyed on a hyperparameter, a locked panel, bar/box plots, 2D and 3D scatter, parallel coordinates, an image grid, a table panel, a saved OR-of-AND filter, a configured experiment table, and dashboard-wide smoothing/sampling. |
| [`03_build_experiment_view.py`](./examples/03_build_experiment_view.py) | The same kind of sections for experiments' own pages via `build_experiment_view()`; one experiment view serves every run in the project, and rerunning updates it in place. |
| [`04_migrate_views.py`](./examples/04_migrate_views.py) | Copying views between projects, workspaces or accounts (`--dst-api-key`), with `--match` filtering and `"Unsaved Changes"` views skipped by default. Safe to rerun; creates the destination project if needed. |
| [`05_standardize_workspace.py`](./examples/05_standardize_workspace.py) | Pushing one house dashboard across a whole workspace, building each project's panels from the metrics that project actually logs. `--update` replaces it in place with `update_view` instead of adding a duplicate. |
| [`06_harvest_panels.py`](./examples/06_harvest_panels.py) | Collects every code panel on existing dashboards (JavaScript, Python, and Panel library panels placed in the UI) and rebuilds them into a "Panel Gallery" view. |
| [`07_code_panels.py`](./examples/07_code_panels.py) | Python and JavaScript panels written in the script, plus a panel uploaded to the Panel library whose code is copied onto the dashboard. |
