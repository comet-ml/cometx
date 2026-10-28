# -*- coding: utf-8 -*-
"""
07 - Python and JavaScript panels written from the SDK.

Built-in panels draw what Comet already knows how to draw. Code panels run your
own code: a Python panel runs on a Comet server and shows what it passes to
``ui.display()``; a JavaScript panel runs in the browser and hands a Plotly
figure to ``this.plot()``. Both keep their code in the view itself, so neither
needs anything created in the UI first.

This builds one dashboard with three code panels:

  * a Python panel whose code is written here,
  * a Python panel whose code comes from a panel uploaded to the Panel library
    with ``API.upload_panel_code`` (a library panel can't be placed on a
    dashboard with an API key, so its code is copied in with ``get_panel_code``),
  * a JavaScript panel.

Rerunning updates the dashboard in place instead of adding another.

    python 07_code_panels.py --workspace WS
    python 07_code_panels.py --workspace WS --no-javascript

JavaScript panels need a Comet version with JavaScript panel support; on one
without it, the panel is left out of the dashboard. Pass ``--no-javascript``
there.
"""

import argparse

import comet_ml

from cometx import API
from cometx.views import (
    Section,
    build_view,
    describe_view,
    javascript_panel,
    python_panel,
)

# A Python panel: the experiments selected on the dashboard come from
# get_panel_experiment_keys(), each line takes its experiment's dashboard color,
# and ui.display() shows a Plotly figure.
PYTHON_CODE = """
from comet_ml import API, ui
import plotly.graph_objects as go

api = API()
experiment_keys = api.get_panel_experiment_keys()
colors = api.get_panel_experiment_colors() or {}
results = api.get_metrics_for_chart(experiment_keys, ["val_accuracy"])

fig = go.Figure()
for key, result in results.items():
    for metric in result["metrics"]:
        fig.add_trace(
            go.Scatter(
                x=metric["steps"],
                y=metric["values"],
                mode="lines",
                name=result.get("experimentName") or key[:9],
                line=dict(color=colors.get(key, {}).get("primary")),
            )
        )
fig.update_layout(margin=dict(l=40, r=10, t=10, b=40), yaxis_title="val_accuracy")
ui.display(fig)
"""

# Uploaded to the Panel library, then copied into the dashboard.
LIBRARY_PANEL_NAME = "Best run (SDK example)"
LIBRARY_CODE = """
from comet_ml import API, ui

api = API()
experiment_keys = api.get_panel_experiment_keys()
results = api.get_metrics_for_chart(experiment_keys, ["accuracy"])

best = max(
    (
        (max(metric["values"]), result.get("experimentName") or key[:9])
        for key, result in results.items()
        for metric in result["metrics"]
        if metric["values"]
    ),
    default=None,
)
if best is None:
    ui.display("No accuracy logged yet.")
else:
    ui.display("Best accuracy: %.3f (%s)" % best)
"""

# A JavaScript panel. The class must be named MyPanel.
JAVASCRIPT_CODE = """
class MyPanel extends Comet.Panel {
  setup() {
    this.options = { metric: "loss" };
  }

  async draw(experimentKeys, projectId) {
    const results = await this.api.experimentMetricsForChart(
      experimentKeys, [this.options.metric]
    );
    const data = Object.entries(results).flatMap(([key, result]) =>
      result.metrics.map(metric => ({
        type: "scatter",
        mode: "lines",
        name: result.experimentName || key.slice(0, 9),
        x: metric.steps,
        y: metric.values,
        line: { color: this.getColor(key) },
      }))
    );
    this.plot({ data, layout: { yaxis: { title: { text: this.options.metric } } } });
  }
}
"""


def upload_library_panel(api, workspace):
    """Upload the library panel, overwriting it if an earlier run uploaded it."""
    existing = [
        panel
        for panel in api.get_panels(workspace)
        if panel.get("templateName") == LIBRARY_PANEL_NAME
    ]
    template_id = existing[0]["templateId"] if existing else None
    result = api.upload_panel_code(
        workspace, LIBRARY_PANEL_NAME, LIBRARY_CODE, template_id=template_id
    )
    return result["templateId"]


def build(name, library_code, with_javascript):
    panels = [
        python_panel(code=PYTHON_CODE, name="Python: validation accuracy"),
        python_panel(code=library_code, name="Python: %s" % LIBRARY_PANEL_NAME),
    ]
    if with_javascript:
        panels.append(javascript_panel(JAVASCRIPT_CODE, name="JavaScript: loss"))
    return build_view(
        name, sections=[Section("Code panels", panels, columns=2, height=2)]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--workspace")
    parser.add_argument("--project", default="view-api-demo")
    parser.add_argument("--name", default="SDK Code Panels")
    parser.add_argument(
        "--no-javascript",
        action="store_true",
        help="leave out the JavaScript panel",
    )
    args = parser.parse_args()

    comet_ml.login()
    api = API()
    workspace = args.workspace or api.get_default_workspace()

    template_id = upload_library_panel(api, workspace)
    print("Library panel %r: %s" % (LIBRARY_PANEL_NAME, template_id))
    library_code = api.get_panel_code(template_id)

    view = build(args.name, library_code, not args.no_javascript)
    print(describe_view(view))

    existing = [
        v for v in api.get_views(workspace, args.project) if v.name == args.name
    ]
    if existing:
        view.template_id = existing[0].template_id
        saved = api.update_view(workspace, args.project, view)
    else:
        saved = api.create_view(workspace, args.project, view)
    if not saved:
        raise SystemExit("The backend rejected the view.")

    print(
        "\n%s %r\nOpen: %s/%s/%s/view/%s/panels"
        % (
            "Updated" if existing else "Created",
            saved.name,
            api.server_url.rstrip("/"),
            workspace,
            args.project,
            saved.template_id,
        )
    )


if __name__ == "__main__":
    main()
