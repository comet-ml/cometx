# cometx

```
   _________  ____ ___  ___  / /__  __
  / ___/ __ \/ __ `__ \/ _ \/ __/ |/_/
 / /__/ /_/ / / / / / /  __/ /__>  <
 \___/\____/_/ /_/ /_/\___/\__/_/|_|
```

Open source extensions for the [Comet](https://www.comet.com/site/?utm_source=cometx&utm_medium=referral&utm_campaign=cometx_2022&utm_content=github) SDK.

These extensions are created and supported by the community and are
not an official project of Comet ML. We welcome contributions!

## Installation

```
pip install cometx --upgrade
```

To use these command-line functions, you can set your Comet API key and URL override using command-line flags:

```
cometx --api-key="YOUR-COMET-API-KEY" COMMAND
```

If you are a Comet on-prem user, and your installation does not use smart-keys, you'll also need to set the URL override:

```
cometx --api-key="YOUR-COMET-API-KEY" --url-override="https://your-companys-comet.com/clientlib/" COMMAND
```

## Usage

`cometx` is composed of a series of commands that are useful
independently, and can be used together to create sophisticated tools
for ML management.

### Commands

* [cometx admin](#cometx-admin)
* [cometx config](#cometx-config)
* [cometx copy](#cometx-copy)
* [cometx delete-assets](#cometx-delete-assets)
* [cometx download](#cometx-download)
* [cometx list](#cometx-list)
* [cometx log](#cometx-log)
* [cometx reproduce](#cometx-reproduce)
* [cometx smoke-test](#cometx-smoke-test)
* [cometx update](#cometx-update)

For all commands, use the `--help` flag to get additional information.

## Global Options

These flags are availble before a command:

* `--api-key API_KEY` - Set the COMET_API_KEY
* `--url-override URL_OVERRIDE` - Set the COMET_URL_OVERRIDE

This command shows the cometx version:

* `--version` - Display comet_ml version

This command can be used globally, or for individual commands:

* `-h, --help` - Show help message

## cometx list

This command is used to:

* get a list of all workspaces that you are a member of
* get a list of all projects in a workspace
* get a list of all experiments (by name or key) in a project

cometx list examples:

```
cometx list WORKSPACE/PROJECT/EXPERIMENT-KEY-OR-NAME
cometx list WORKSPACE/PROJECT
cometx list WORKSPACE
cometx list
```

### Flags

* `-u, --use-name` - Use experiment names for experiment folders and listings
* `--query QUERY` - Only list experiments that match this Comet query string
* `--debug` - Provide debug info

For more information, `cometx list --help`

## cometx download

This command is used to:

* download all workspaces, projects, and experiments of workspaces that you are a member of
* download all projects, and experiments of a given workspace
* download all experiments of a given workspace/project
* download artifacts and models from the registry
* download panels

> **Note**: For detailed information on copying experiments from one Comet installation to another, see [MIGRATIONS.md](https://github.com/comet-ml/cometx/blob/main/MIGRATIONS.md).

cometx download examples:

```
cometx download WORKSPACE/PROJECT/EXPERIMENT-KEY-OR-NAME [RESOURCE ...] [FLAGS ...]
cometx download WORKSPACE/PROJECT [RESOURCE ...] [FLAGS ...]
cometx download WORKSPACE [RESOURCE ...] [FLAGS ...]
cometx download [RESOURCE ...] [FLAGS ...]
```

Where [RESOURCE ...] is zero or more of the following names:

* run - alias for: code, git, output, graph, and requirements
* system
* others
* parameters
* metadata
* metrics
* assets
* html
* project - alias for: project_notes, project_metadata

If no RESOURCE is given it will download all of them.

### Flags

* `--from from` - Source of data to copy. Should be: comet, wandb, or neptune
* `-i IGNORE [IGNORE ...], --ignore IGNORE [IGNORE ...]` - Resource(s) (or 'experiments') to ignore
* `-j PARALLEL, --parallel PARALLEL` - The number of threads to use for parallel downloading; default (None) is based on CPUs
* `-o OUTPUT, --output OUTPUT` - Output directory for downloads
* `-u, --use-name` - Use experiment names for experiment folders and listings
* `-l, --list` - List the items at this level (workspace, project, experiment, artifacts, or model-registry) rather than download
* `--flat` - Download the files without subfolders
* `-f, --ask` - Queries the user; if flag not included system will answer `yes` for all queries
* `--filename FILENAME` - Only get resources ending with this
* `--query QUERY` - Only download experiments that match this Comet query string
* `--asset-type ASSET_TYPE` - Only get assets with this type
* `--sync SYNC` - What level to sync at: all, experiment, project, or workspace
* `--debug` - Provide debug info

To download artifacts:

```
cometx download WORKSPACE/artifacts/NAME [FLAGS ...]
cometx download WORKSPACE/artifacts/NAME/VERSION-OR-ALIAS [FLAGS ...]
```

To download models from the model registry:

```
cometx download WORKSPACE/model-registry/NAME [FLAGS ...]
cometx download WORKSPACE/model-registry/NAME/VERSION-OR-STAGE [FLAGS ...]
```

To download panels:

```
cometx download WORKSPACE/panels/NAME-OR-ID [FLAGS ...]
cometx download WORKSPACE/panels [FLAGS ...]
```

For more information, `cometx download --help`

## cometx copy

This command is used to:

* copy downloaded data to a new experiment
* create a symlink from one project to existing experiments
* copy panels

> **Note**: For detailed information on copying experiments from one Comet installation to another, see [MIGRATIONS.md](https://github.com/comet-ml/cometx/blob/main/MIGRATIONS.md).

cometx copy examples:

```
cometx copy SOURCE DESTINATION
cometx copy --symlink SOURCE DESTINATION
```

where SOURCE is:

* if not `--symlink`, "WORKSPACE/PROJECT/EXPERIMENT", "WORKSPACE/PROJECT", or "WORKSPACE" folder
* if `--symlink`, then it is a Comet path to workspace or workspace/project
* "WORKSPACE/panels" or "WORKSPACE/panels/PANEL-ZIP-FILENAME" to copy panels

where DESTINATION is:

* WORKSPACE
* WORKSPACE/PROJECT

Not all combinations are possible:

| Destination → <br/>Source ↓ | WORKSPACE            | WORKSPACE/PROJECT      |
|--------------------|----------------------|------------------------|
| `WORKSPACE/*/*`      | Copies all projects  | N/A                    |
| `WORKSPACE/PROJ/*`   | N/A                  | Copies all experiments |
| `WORKSPACE/PROJ/EXP` | N/A                  | Copies experiment      |

### Asset Types

* 3d-image
* 3d-points - deprecated
* audio
* confusion-matrix - may contain assets
* curve
* dataframe
* dataframe-profile
* datagrid
* embeddings - may reference image asset
* histogram2d - not used
* histogram3d - internal only, single histogram, partial logging
* histogram_combined_3d
* image
* llm_data
* model-element
* notebook
* source_code
* tensorflow-model-graph-text - not used
* text-sample
* video

### Flags

* `-i IGNORE [IGNORE ...], --ignore IGNORE [IGNORE ...]` - Resource(s) (or 'experiments') to ignore
* `-j PARALLEL, --parallel PARALLEL` - The number of threads to use for parallel uploading; default (None) is based on CPUs
* `--debug` - If given, allow debugging
* `--quiet` - If given, don't display update info
* `--symlink` - Instead of copying, create a link to an experiment in a project
* `--sync` - Check to see if experiment name has been created first; if so, skip

For more information, `cometx copy --help`

## cometx log

This command is used to log a resource (metrics, parameters, asset,
etc) file to a specific experiment or experiments.

cometx log examples:

```
cometx log WORKSPACE/PROJECT/EXPERIMENT-KEY FILENAME ... --type=TYPE
cometx log WORKSPACE PANEL-ZIP-FILENAME ... --type=panel
cometx log WORKSPACE PANEL.py ... --type=panel
cometx log WORKSPACE PANEL-URL ... --type=panel
cometx log WORKSPACE/PROJECT --type=other --set "key:value"
cometx log WORKSPACE --type=other --set "key:value"
```

Where TYPE is one of the following names:

* all
* asset
* audio
* code
* image
* metrics
* notebook
* panel
* tensorflow-file
* text-sample
* video
* other
* tensorboard-folder-assets

### Flags

* `--type TYPE` - The type of item to log
* `--set SET` - The key:value to log
* `--query QUERY` - A Comet Query string, see https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/API/#apiquery
* `--debug` - If given, allow debugging
* `--use-base-name` - If given, using the basename for logging assets

For more information, `cometx log --help`

## cometx delete-assets

To delete experiments assets:

```
cometx delete-assets WORKSPACE/PROJECT --type=image
cometx delete-assets WORKSPACE/PROJECT/EXPERIMENT --type=all
```

Type can be valid asset type, including:

* all
* asset
* audio
* code
* image
* notebook
* text-sample
* video

### Flags

* `--type TYPE` - The type of item to log
* `--debug` - If given, allow debugging
* `--query QUERY` - Only delete experiments that match this Comet query string

For more information, `cometx delete-assets --help`

## cometx config

To enable auto-logging of your notebooks in Jupyter environments:

```python
cometx config --auto-log-notebook yes
```

To turn auto-logging of notebooks off, use:

```python
cometx config --auto-log-notebook no
```

If you keep the generated experiment URLs in the notebook, but later edit the notebook, the notebooks will be updated in all of the experiments created in the notebook.

### Flags

* `--debug` - If given, allow debugging
* `--auto-log-notebook AUTO_LOG_NOTEBOOK` - Takes a 1/yes/true, or 0/no/false

For more information, `cometx config --help`

## cometx reproduce

```
cometx reproduce [-h] [--run] [--executable EXECUTABLE] COMET_PATH OUTPUT_DIR
```

This command is used to reproduce experiments by copying files to a specified output directory.

### Flags

* `--run` - Run the reproducible script
* `--executable EXECUTABLE` - Run the reproducible script with specified executable

For more information, `cometx reproduce --help`

## cometx update

```
cometx update [-h] [--debug] COMET_SOURCE COMET_DESTINATION
```

To update existing experiments.

cometx update SOURCE DESTINATION

where SOURCE is a folder:

* "WORKSPACE/PROJECT/EXPERIMENT"
* "WORKSPACE/PROJECT"
* "WORKSPACE"

where DESTINATION is a Comet:

* WORKSPACE
* WORKSPACE/PROJECT

### Flags

* `--debug` - If given, allow debugging

For more information, `cometx update --help`

## cometx admin

```
cometx admin [-h] [--host HOST] [--debug] ACTION [YEAR-MONTH]
```

To perform admin functions

cometx admin chargeback-report

### Flags

* `--host HOST` - Override the HOST URL
* `--debug` - If given, allow debugging

For more information, `cometx admin --help`

## cometx smoke-test

```
cometx smoke-test [-h] [--exclude [EXCLUDE ...]] [--debug DEBUG] COMET_PATH [include ...]
```

Perform a smoke test on a Comet installation. Logs results to WORKSPACE/smoke-tests or WORKSPACE/PROJECT.

Examples:

Run all tests:
```
cometx smoke-test WORKSPACE    # project defaults to smoke-tests
cometx smoke-test WORKSPACE/PROJECT
```

Run everything except mpm tests:
```
cometx smoke-test WORKSPACE/PROJECT --exclude mpm
```

Run just optimizer tests:
```
cometx smoke-test WORKSPACE/PROJECT optimizer
```

Run just metric tests:
```
cometx smoke-test WORKSPACE/PROJECT metric
```

Items to include or exclude:

* optimizer
* mpm
* panel
* opik
* experiment
  * metric
  * image
  * asset
  * dataset-info
  * confusion-matrix
  * embedding

### Flags

* `--exclude [EXCLUDE ...]` - Items to exclude; any of: asset, confusion-matrix, dataset-info, embedding, experiment, image, metric, mpm, opik, optimizer, panel
* `--debug DEBUG` - Show debugging information

For more information, `cometx smoke-test --help`

## Copy/Download Use Cases

In this section we'll explore some common scenarios.

1. Copy a specific project from one Comet installation to another
2. Copy all projects in workspace to a new workspace
3. Copy specific experiments in a project to new experiments

### 1. Copy a specific project from one comet installation to another

A useful idiom is to set your Comet environment variables on the line
of a command. In this manner, you can set the `COMET_URL_OVERRIDE`
and `COMET_API_KEY` for different installations.

Of course, you don't have to set the environment variables if you are
copying experiments on the same Comet installation.

Here is how you one could download the experiments in
WORKSPACE/PROJECT from http://comet.a.com:

```shell
cometx --api-key=A-KEY download WORKSPACE/PROJECT
```

The `cometx download` command downloads all of the Comet experiment
data into local files. Note that WORKSPACE/PROJECT refers to a
workspace and project on http://comet.a.com.

One could then copy the downloaded experiment data with a similar command:

```shell
cometx --api-key=B-KEY copy WORKSPACE/PROJECT NEW-WORKSPACE/NEW-PROJECT
```

Note that WORKSPACE/PROJECT now refers to a directory, and
NEW-WORKSPACE/NEW-PROJECT refers to a workspace and project on
http://comet.b.com.

### 2. Copy all projects in workspace to a new workspace

Similarly, one can copy all of the projects by first downloading them:

```shell
cometx --api-key=A-KEY download WORKSPACE
```

and then copying them:

```shell
cometx --api-key=B-KEY copy WORKSPACE NEW-WORKSPACE
```

### 3. Copy specific experiments in a project to new experiments

Similarly, one can copy a single experiment first downloading it:

```shell
cometx --api-key=A-KEY download WORKSPACE/PROJECT/EXPERIMENT-NAME-OR-ID
```

and then copying it:

```shell
cometx --api-key=B-KEY copy WORKSPACE/PROJECT/EXPERIMENT-NAME-OR-ID NEW-WORKSPACE/NEW-PROJECT
```

## Running Tests

WARNING: Running the tests will create experiments, models, assets, etc.
in your default workspace if not set otherwise.

To run the tests, you can either export all of these items in the
environment:

```shell
$ export COMET_USER="<USERNAME>"
$ export COMET_WORKSPACE="<WORKSPACE>"
$ export COMET_API_KEY="<API-KEY>"
$ pytest tests
```

Or, define `workspace` and `api_key` in your ~/.comet.config file:

```shell
$ export COMET_USER="<USERNAME>"
$ pytest tests
```
