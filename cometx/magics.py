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

from IPython.core.magic import register_cell_magic, register_line_magic
from IPython.display import display


def remove_quotes(text):
    if text[0] == text[-1] == "'":
        return text[1:-1]
    elif text[0] == text[-1] == '"':
        return text[1:-1]
    return text


@register_cell_magic
@register_line_magic
def cometx(line, cell=None):
    # workspace/project
    # workspace/project/experiment
    # workspace/project "Panel Name"
    code = cell

    if " " in line.strip():
        line, panel_name = line.split(" ", 1)
    else:
        panel_name = None

    path = line.split("/")
    if len(path) == 2:
        workspace = path[0]
        project_name = path[1]
        experiment_key = ""
    elif len(path) == 3:
        workspace = path[0]
        project_name = path[1]
        experiment_key = path[2]
    else:
        raise Exception(
            "Need to provide WORKSPACE/PROJECT or WORKSPACE/PROJECT/EXPERIMENT to %%cometx"
        )

    if cell is None:
        if panel_name is None:
            display("Loading Python Panels...")
            cell = """
from cometx import API
from comet_ml import ui
import datetime
api = API()
templates = api.get_panels("{workspace}")
selected = ui.dropdown(
            "Python Panels:", templates,
            format_func=lambda item: "%s - %s" % (item["templateName"], datetime.datetime.fromtimestamp(item["revisionId"] / 1000)))
if selected:
    ui.display_markdown("<pre>" + selected["code"] + "</pre>")
    ui.display_markdown("To edit and run here: **%%cometx {line} %r**" % selected["templateName"])
""".format(
                workspace=workspace, line=line
            )
        else:
            panel_name = remove_quotes(panel_name)
            from cometx import API

            api = API()
            contents = api.get_panel_code(workspace, panel_name)
            contents = (f"%%cometx {line}\n\n") + contents
            get_ipython().set_next_input(contents, replace=True)
            return

    if "def main(" not in cell:
        cell = "\n".join(["    %s\n" % line for line in cell.splitlines()])
        # We replace API with this to allow use of local workspace and project:
        cell = """
def main(st):
    import cometx
    import comet_ml
    from comet_ml._ui import UI

    class API(cometx.API):
        def get_panel_project_name(self):
            return "{project_name}"
        def get_panel_workspace(self):
            return "{workspace}"
        def get_panel_experiments(self):
            if "{experiment_key}":
                return [self.get_experiment("{workspace}", "{project_name}", "{experiment_key}")]
            else:
                return self.get_experiments("{workspace}", "{project_name}")
        def get_panel_experiment_keys(self):
            return [e.id for e in self.get_panel_experiments()]
        def get_panel_metrics_names(self):
            return sorted(
                [
                    name
                    for name in self._get_metrics_name(
                        "{workspace}",
                        "{project_name}",
                    )
                    if not name.startswith("sys.")
                ]
            )
    # Replace _st with ipywidgets
    class UI(UI):
        _st = st # argument passed into main()
        session_state = st.session_state
    comet_ml.ui = UI()
    comet_ml.API = API
    del comet_ml, API, UI, cometx
    {cell}
    import comet_ml
    from cometx.panel_utils import create_panel_zip
    ui.display("<hr>")
    cols = ui.columns(2)
    panel_name = cols[0].input("Name of Panel:", "Custom Panel")
    if cols[1].button("Deploy to Comet"):
        if panel_name:
            api = comet_ml.API()
            zip_filename = create_panel_zip(
                panel_name,
                '''{cell_str}''',
            )
            api.upload_panel_zip("{workspace}", zip_filename)
    ui.display("<hr>")
""".format(
            cell=cell,
            workspace=workspace,
            project_name=project_name,
            experiment_key=experiment_key,
            cell_str=code.replace("'", "\\'"),
        )
    code = """
## User code:
{cell}
## End user code
from cometx._ui import Streamlit
st = Streamlit()
st._run(main)
""".format(
        cell=cell
    )
    get_ipython().run_cell(code)
    # print(code)
