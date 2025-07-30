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

import inspect
import io
import traceback

import ipywidgets as widgets
import markdown
from IPython.display import HTML, clear_output, display

STYLE = """
<style>
button.lm-Widget.jupyter-widgets.jupyter-button.widget-button {
    width: fit-content;
}
.widget-dropdown > select {
    background-color: rgb(240, 242, 246);
}
.widget-dropdown, .jupyter-widget-dropdown {
    max-width: 600px;
    width: 90%;
}
table {
    width: -webkit-fill-available;
}
.widget-image, .jupyter-widget-image {
    max-width: 600px;
    height: auto;
}
</style>
"""


class SessionState(dict):
    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value


class Streamlit:
    def __init__(self, parent=None):
        self.layout = widgets.Layout(width="auto")
        self._response = {}
        self._output = widgets.Output()
        self.session_state = SessionState()
        if parent is None:
            self._make_panel()
        else:
            self._parent = parent

    def _observe(self, widget, callback, names=None):
        widget.observe(callback, names=names)

    def _on_click(self, widget, callback):
        widget.on_click(callback)

    def _make_panel(self):
        self._top_level = widgets.VBox()
        self._parent = self._top_level

    def _clone(self, parent):
        app = Streamlit(parent)
        app._output = self._output
        app._response = self._response
        app.session_state = self.session_state
        app._function = self._function
        app._top_level = self._top_level
        return app

    def _make_key(self, widget_type, key):
        if key is None:
            calling_frame = inspect.getouterframes(inspect.currentframe())[3]
            key = calling_frame.lineno
            desc = "lineno"
        else:
            desc = "userkey"
        return "%s-%s-%s" % (widget_type, desc, key)

    def _append(self, widget):
        self._parent.children = self._parent.children + tuple([widget])

    def _clear(self):
        self._parent.children = tuple()

    # Widgets, replicates streamlit widgets

    def pyplot(self, figure):
        img_buf = io.BytesIO()
        figure.savefig(img_buf, format="png")
        img_buf.seek(0)
        image_data = img_buf.read()
        image = widgets.Image(
            value=image_data,
            format="png",
        )
        self._append(image)

    def image(self, data):
        image = widgets.Image(
            value=data,
        )
        self._append(image)

    def write(self, data, unsafe_allow_html=False):
        if data.__class__.__name__ == "DataFrame":
            try:
                from ipydatagrid import DataGrid

                data = DataGrid(data)
            except ImportError:
                data = widgets.HTML(data)
        else:
            data = widgets.HTML(str(data))
        self._append(data)

    def selectbox(
        self,
        label,
        options,
        index=0,
        format_func=str,
        key=None,
        help=None,
        on_change=None,
        args=None,
        kwargs=None,
    ):
        key = self._make_key("selectbox", key)
        options = list(options)
        if len(options) == 0:
            label_options = [""]
            options = [None]

        label_options = [format_func(option) for option in options]
        index = self._response.get(key, index)
        if index < len(label_options):
            value = label_options[index]
        else:
            value = label_options[0]
            index = 0
        widget = widgets.Dropdown(
            style={
                "description_width": "initial",
                "width": "initial",
            },
            options=label_options,
            value=value,
        )
        self._observe(
            widget,
            lambda results: self._rerun(key, results["owner"].index, on_change, args),
            names="value",
        )
        self._append(widgets.HTML(label))
        self._append(widget)
        self._response[key] = index
        return options[index]

    def multiselect(
        self,
        label,
        options,
        default=None,
        format_func=None,
        key=None,
        on_change=None,
        args=None,
        kwargs=None,
    ):
        key = self._make_key("multiselect", key)
        if default is None:
            default = []
        options = list(options)
        if len(options) == 0:
            options = [None]
        value = self._response.get(key, default)
        label_options = [format_func(option) for option in options]
        value = [x for x in value if x in label_options]
        widget = widgets.SelectMultiple(
            style={
                "description_width": "initial",
                "width": "initial",
            },
            options=label_options,
            value=value,
        )
        self._observe(
            widget,
            lambda results: self._rerun(key, results["new"], on_change, args),
            names="value",
        )
        self._append(widgets.HTML(label))
        self._append(widget)
        self._response[key] = value
        return value

    def _check_results(self, results, label_options):
        valid = []
        for item in results["new"]:
            if item in label_options:
                valid.append(item)
        return valid

    def plotly_chart(self, figure, use_container_width=False):
        # FIXME
        pass

    def button(self, label, key=None, help=None, on_click=None, args=None, kwargs=None):
        key = self._make_key("button", key)
        widget = widgets.Button(description=label)
        self._on_click(widget, lambda value: self._rerun(key, value, on_click, args))
        self._append(widget)
        retval = self._response.get(key, False)
        self._response[key] = False
        return retval

    def markdown(self, text, unsafe_allow_html=False):
        md = (
            """<div class="jp-RenderedHTMLCommon">"""
            + markdown.markdown(text)
            + """</div>"""
        )
        self._append(widgets.HTML(md))

    def text_input(
        self, label, value, key=None, on_change=None, args=None, kwargs=None
    ):
        key = self._make_key("input", key)
        retval = self._response.get(key, value)
        widget = widgets.Text(
            value=retval,
            description=label,
            continuous_update=False,
            layout=self.layout,
        )
        widget.style = {"description_width": "auto"}
        self._observe(
            widget,
            lambda results: self._rerun(key, results["owner"].value, on_change, args),
            names="value",
        )
        self._append(widget)
        self._response[key] = retval
        return retval

    def checkbox(self, label, value, key=None, on_change=None, args=None, kwargs=None):
        # FIXME
        pass

    def columns(self, config):
        row = widgets.HBox()
        if isinstance(config, int):
            items = config
            widths = [(1 / items) * 100] * items
        else:
            items = len(config)
            widths = [((i / sum(config)) * 100) for i in config]
        row.children = tuple(
            [
                widgets.VBox(layout=widgets.Layout(width=f"{widths[i]}%"))
                for i in range(items)
            ]
        )
        self._append(row)
        return [self._clone(child) for child in row.children]

    def _run(self, function):
        self._function = function
        self._execute()

    def _rerun(self, key, value, callback, args):
        if key is not None:
            self._response[key] = value
        if callback:
            if args is None:
                callback()
            else:
                callback(*args)
        self._execute()

    def _execute(self):
        self._make_panel()
        with self._output:
            clear_output(wait=True)
            try:
                self._function(self)
            except Exception:
                traceback_str = traceback.format_exc()
                self._clear()
                self.markdown(
                    f'<pre style="background-color:#fdd;">{traceback_str}</pre>'
                )
            # This will only actually display the first time
            display(self._top_level, HTML(STYLE))
        clear_output(wait=True)
        display(self._output)
