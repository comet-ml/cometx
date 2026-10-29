# Comet Panel Studio

`cometx panel-studio` is an editor for writing and testing Comet **Python panels** on
your own machine. You write the panel's code, run it against your real experiments,
and see it exactly as a Comet view would show it. When it works, upload it to a
view or to your workspace's Panel library.

Prefer your own editor, such as VS Code or vim? The studio can follow a file you
edit elsewhere, and three more commands cover that way of working without the
studio: `cometx panel-preview`, `cometx panel-upload` and `cometx panel-download`.
See [Using your own editor](#using-your-own-editor).

* [Starting the studio](#starting-the-studio)
* [The studio at a glance](#the-studio-at-a-glance)
* [Choosing what the panel sees](#choosing-what-the-panel-sees)
* [Editing and files](#editing-and-files)
* [Running a panel](#running-a-panel)
* [Uploading a panel](#uploading-a-panel)
* [Using your own editor](#using-your-own-editor)
* [Writing panels](#writing-panels)
* [How it matches comet.com](#how-it-matches-cometcom)
* [Command-line options](#command-line-options)
* [Troubleshooting](#troubleshooting)
* [Security notes](#security-notes)

## Starting the studio

Install or upgrade cometx, which brings Streamlit (1.41 or later) and Plotly with
it:

```bash
pip install -U cometx
```

Then, from the folder that holds your panel files:

```bash
cometx panel-studio
```

and open the address it prints, normally **http://localhost:8501**. To open a file
and pick a project right away:

```bash
cometx panel-studio my_panel.py --workspace my-workspace --project my-project
```

Your API key and Comet server come from your usual Comet configuration
(`COMET_API_KEY`, `COMET_URL_OVERRIDE`, or `~/.comet.config`). You can also pass
them with `--api-key` and `--url-override`, or type them in the sidebar.

Stop the studio with **Ctrl-C** in the terminal where it runs. Run it in its own
terminal: it keeps running until you stop it.

## The studio at a glance

* **Sidebar, Comet:** your server, API key, workspace and project, plus an
  **Open in Comet** link to that project.
* **Sidebar, Panel:** the experiments the panel sees.
* **Left column:** the **File** box with **Browse...**, **Load** and **Save**, the
  code editor, and the **Run** and **Upload...** buttons.
* **Right column:** the running panel.

## Choosing what the panel sees

**Server.** Type your Comet address, such as `https://www.comet.com` or your own
installation's address. The studio adds the `/clientlib/` part itself, so
`https://comet.example.com`, `https://comet.example.com/clientlib` and
`https://comet.example.com/clientlib/` all work.

**API key.** Filled in from your Comet configuration when it has one. It stays in
the studio's session and is not saved anywhere by the sidebar.

**Workspace and project.** The workspace starts as the one given with
`--workspace`, your configured workspace, or your account's default. Projects can
be sorted by **Recently updated** (the default, so the newest project is chosen
when you don't pass `--project`) or by **Name**.

**Experiments.** These are the experiments the panel sees through
`api.get_panel_experiment_keys()`. As on a Comet view, which hands a panel the
first page of its experiments table, the **first 25** are selected to start, in the
current sort order. Every experiment in the project is listed, so you can add or
remove any of them. Sort by **Recently updated** (the later of an experiment's start
and end times) or by **Name**; changing the sort keeps your selection.

## Editing and files

The editor starts with the same code Comet's own Python panel editor starts with, a
line chart of a metric you pick, so you can press **Run** straight away to check your
key and project.

**File** holds the path of the file you're working on:

* **Browse...** opens a file browser on your machine, starting in the current file's
  folder (or the folder you started the studio in). Click **📁 ..** to go up, a
  **📁** folder to go in, or type a folder into the **Folder** box to jump there.
  Clicking a **📄** `.py` file opens it.
* Or type a path into **File** and press Enter. An existing file opens; a new path
  stays empty until you save.
* **Save** writes the editor's code to that file, creating it if needed.
* **Load** reads the file again, for example after editing it elsewhere.
* **Follow file** reloads the file and runs the panel each time it's saved; see
  [Following a file](#following-a-file).

Keeping panels in files means they can live in version control next to your
training code.

## Running a panel

Press **Run**. The panel appears on the right, running your current code against
the experiments selected in the sidebar. Press **Run** again after each change.
Widgets in the panel, such as a dropdown, work as they do on a view.

Errors in your code show in the panel as a Streamlit error with its traceback, the
way they would in Comet.

## Uploading a panel

Press **Upload...** and choose where it goes:

* **A view in the project.** Pick an existing view or **New view**
  and give it a name.
  * If that view already has a Python panel with the same **Panel name**, its
    code is replaced in place; nothing else changes.
  * Otherwise the panel is added at the bottom of a **Panel Studio** section, which
    is created if needed. The rest of the view is left as it was.
  * The panel's code is stored in the view itself. When it's done, the dialog
    links to the view.
* **The Panel library of the workspace.** Saves the panel as a library panel,
  replacing a library panel with the same name. Library panels are placed on
  views from **Add panel > Panel library** in Comet; they can't be placed on a
  view from the API.

The **Panel name** starts as the file's name.

## Using your own editor

### Following a file

Open the file in the studio (with `FILE`, **Browse...**, or the **File** box) and
turn on **Follow file**. Each time you save the file in your own editor, the studio
loads the new code and runs the panel again, within about a second. While
following, the code in the studio is read-only and **Save** is off, so the studio
never overwrites what you wrote; turn **Follow file** off to edit in the studio
again. Experiments, the project and **Upload...** work as usual.

### Previewing without the studio: `cometx panel-preview`

For the lightest loop, `panel-preview` shows just the panel, in your browser, and
runs it again each time you save the file:

```bash
cometx panel-preview loss_curves.py --project my-project
```

It prints the panel's address (and opens it) and keeps running until Ctrl-C. Keep
that browser tab, or VS Code's Simple Browser, beside your editor.

| Option | Meaning |
|---|---|
| `FILE` | The panel file. |
| `--project` | The project to run the panel in. Required. |
| `--workspace` | Defaults to your configured workspace, then your account's default. |
| `--experiments` | Which experiments the panel sees: a number (the first N, like a view; default 25), `all`, or names, keys or key prefixes separated by commas. |
| `--sort` | `updated` (default) or `name`: the order used before taking the first N. |
| `--port` | The panel's port. Default 8502; the next free port if taken. |
| `--no-browser` | Don't open the panel in a browser. |
| `--api-key`, `--url-override` | As for `panel-studio`. |

### Uploading from the command line: `cometx panel-upload`

```bash
# To a view (created if needed); the panel is named after the file
cometx panel-upload loss_curves.py --project my-project --view "Training"

# With another name
cometx panel-upload loss_curves.py --project my-project --view "Training" \
    --name "Loss curves"

# To the workspace's Panel library
cometx panel-upload loss_curves.py --library
```

It follows the same rules as the studio's **Upload...**: on a view, a Python
panel with the same name is replaced, otherwise the panel goes in the **Panel
Studio** section; in the library, a panel with the same name is replaced. It prints
the view's address. `--workspace`, `--api-key` and `--url-override` work as
above. Because it's a plain command, it also fits in scripts and CI, for example to
publish a repository's panels to a team view.

### Downloading a panel: `cometx panel-download`

To edit a panel that's already on a view or in the library:

```bash
cometx panel-download --project my-project --view "Training" -o loss_curves.py
cometx panel-download --project my-project --view "Training" \
    --name "Loss curves" -o loss_curves.py
cometx panel-download --library --name "Loss curves" -o loss_curves.py
```

On a view, `--name` can be left out when it has a single Python panel; with
several, the command lists their names. Without `-o` the code is printed. An
existing `-o` file is only overwritten with `--force`. Python panels added through
Comet's own editor keep their code where only the Comet UI can read it; copy their
code from the panel's editor in Comet instead.

### A typical round trip

```bash
cometx panel-download --project my-project --view "Training" \
    --name "Loss curves" -o loss_curves.py
cometx panel-preview loss_curves.py --project my-project    # edit, save, look
cometx panel-upload loss_curves.py --project my-project --view "Training" \
    --name "Loss curves"
```

**VS Code:** run `panel-preview` in the integrated terminal and open its address
with **Simple Browser: Show** from the command palette. **vim:** `:!cometx
panel-upload % --project my-project --view "Training"` uploads the file you're
editing.

## Writing panels

A Python panel is a Python script that draws with `comet_ml.ui`, a thin layer over
Streamlit. The essentials:

```python
from comet_ml import API, ui

api = API()

experiment_keys = api.get_panel_experiment_keys()   # the selected experiments
colors = api.get_panel_experiment_colors() or {}    # {key: {"primary": "#..."}}
metric = ui.dropdown("Metric", api.get_panel_metrics_names())

results = api.get_metrics_for_chart(experiment_keys, [metric])
# ... build a Plotly or matplotlib figure from results ...
ui.display(fig)
```

Other panel helpers include `api.get_panel_workspace()`,
`api.get_panel_project_name()`, `api.get_panel_project_id()` and
`api.get_panel_experiments()`. Use `colors[key]["primary"]` for each experiment so
the panel's lines match the view's colors.

Inside a panel, `print()` writes to the panel, as it does on Comet.

**Magic commands.** As on Comet, a panel can start with:

* `%pip install PACKAGE ...` to install packages the panel needs. Locally they are
  installed into the Python environment running the studio.
* `%include URL` to insert code downloaded from a URL.

See Comet's
[Python panel documentation](https://www.comet.com/docs/v2/guides/comet-ui/experiment-management/visualizations/python-panel/)
for the full API.

## How it matches comet.com

The studio runs panels the same way Comet's Python panel server does, using that
server's own code for building a panel's page. On Comet, each panel becomes a
Streamlit page whose first lines hand `comet_ml` the panel's context; the studio
writes the same pages and serves them from a second Streamlit server, which the
editor shows on the right. So the same panel code behaves the same in both places:

| | Comet view | Panel studio |
|---|---|---|
| Experiments | the first page of the experiments table (25 by default) | the first 25, or the ones you choose |
| Experiment colors | the view's colors | the same color rule as the view |
| Panel options | none (`get_panel_options()` returns `{}`) | none |
| `print()`, `%pip`, `%include` | supported | supported |

Two differences to keep in mind:

* **Installed packages.** Locally the panel uses your Python environment; on Comet
  it uses Comet's panel server. A panel that imports something you happen to have
  installed can still fail on Comet. Put a `%pip install` line at the top for
  anything beyond the usual (Plotly, pandas, NumPy, matplotlib) so it works in both.
* **Startup time.** On Comet, the first panel in a session waits for the panel
  server to start, which can take most of a minute. Locally, panels start at once.

## Command-line options

```
cometx panel-studio [FILE] [--workspace WORKSPACE] [--project PROJECT]
                    [--port PORT] [--panel-port PANEL_PORT]
                    [--api-key API_KEY] [--url-override URL_OVERRIDE]
```

| Option | Meaning |
|---|---|
| `FILE` | The panel file to open. It's created when you save. |
| `--workspace` | The workspace to start with. |
| `--project` | The project to start with. |
| `--port` | The editor's port. Default 8501. |
| `--panel-port` | The port panels are served on. Default 8502. |
| `--api-key` | A Comet API key, instead of your configured one. |
| `--url-override` | A Comet server, instead of your configured one. |

If a port is already in use, for example by another studio, the next free port is
used and the studio says which.

## Troubleshooting

**The startup output shows a "RequestsDependencyWarning".** That comes from your
Python environment, where `requests` is older than the `urllib3` or `chardet`
installed beside it. `pip install -U requests` fixes it.

**"Port 8501 is in use; the editor uses port 8503."** Another program, often an
earlier studio, holds the port. Open the address printed below that line, or stop
the other studio.

**"Could not connect"** in the sidebar. Check the server address and API key. For
your own Comet installation, use its address (e.g. `https://comet.example.com`).

**The panel area stays blank after Run.** The panel is served from the same host as
the editor, on the panel port (8502 by default). If you reach the studio from
another machine, both ports must be reachable from your browser.

**A panel works in the studio but not on Comet.** Usually a package that is
installed locally but not on Comet's panel server; add `%pip install` for it.

## Security notes

* Your API key is used to reach Comet from your machine, as with any `comet_ml`
  script.
* As on Comet's panel server, each panel page holds the API key in plain text. The
  studio writes those pages to a private temporary folder (readable only by you)
  named `cometx-panel-studio-*` in your system's temp directory, and deletes it
  when the studio stops. If the studio is killed rather than stopped with Ctrl-C,
  the folder can be left behind; delete it if you share the machine.
* Panel code runs on your machine with your permissions, so only run panel code you
  trust.
