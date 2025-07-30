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
pip install cometx
```

To use these command-line functions, you'll need to have your Comet
API key set in one of the following two ways.

1. [Environment variables](https://www.comet.com/docs/v2/guides/tracking-ml-training/configuring-comet/?utm_source=cometx&utm_medium=referral&utm_campaign=cometx_2022&utm_content=github#configure-comet-through-environment-variables)
2. [Comet config file](https://www.comet.com/docs/v2/guides/tracking-ml-training/configuring-comet/?utm_source=cometx&utm_medium=referral&utm_campaign=cometx_2022&utm_content=github#configure-comet-using-the-comet-config-file)

Either this way:

```
export COMET_API_KEY="YOUR-COMET-API-KEY"
```

or this way:

```
[comet]
api_key = YOUR-COMET-API-KEY
```

If you are an Comet on-prem user, you'll also need to set the
`COMET_URL_OVERRIDE` environment variable, or add it to your
`~/.comet.config` file as below:

Either this way:

```
export COMET_API_KEY="YOUR-COMET-API-KEY"
export COMET_URL_OVERRIDE="https://your-companys-comet.com/clientlib/"
```

or this way:

```
[comet]
api_key = YOUR-COMET-API-KEY
url_override = https://your-companys-comet.com/clientlib/
```

## Usage

`cometx` is composed of a series of subcommands that are useful
independently, and can be used together to create sophisticated tools
for ML management.

This section will examine some common uses, followed by a more
detailed exploration of each subcommand.

### Use Cases

In this section we'll explore some common scenarios.

1. Copy a specific project from one Comet installation to another
2. Copy all projects in workspace to a new workspace
4. Copy specific experiments in a project to new experiments

## 1. Copy a specific project from one comet installation to another

A useful idiom is to set your Comet environment variables on the line
of a subcommand. In this manner, you can set the `COMET_URL_OVERRIDE`
and `COMET_API_KEY` for different installations.

Of course, you don't have to set the environment variables if you are
copying experiments on the same Comet installation.

Here is how you one could download the experiments in
WORKSPACE/PROJECT from http://comet.a.com:

```shell
COMET_URL_OVERRIDE=http://comet.a.com/clientlib \
COMET_API_KEY=A-KEY \
cometx download WORKSPACE/PROJECT
```

The `cometx download` subcommand downloads all of the Comet experiment
data into local files. Note that WORKSPACE/PROJECT refers to a
workspace and project on http://comet.a.com.

One could then copy the downloaded experiment data with a similar command:

```shell
COMET_URL_OVERRIDE=http://comet.b.com/clientlib \
COMET_API_KEY=B-KEY \
cometx copy WORKSPACE/PROJECT NEW-WORKSPACE/NEW-PROJECT
```

Note that WORKSPACE/PROJECT now refers to a directory, and
NEW-WORKSPACE/NEW-PROJECT refers to a workspace and project on
http://comet.b.com.

## 2. Copy all projects in workspace to a new workspace

Similarly, one can copy all of the projects by first downloading them:

```shell
COMET_URL_OVERRIDE=http://comet.a.com/clientlib \
COMET_API_KEY=A-KEY \
cometx download WORKSPACE
```

and then copying them:

```shell
COMET_URL_OVERRIDE=http://comet.b.com/clientlib \
COMET_API_KEY=B-KEY \
cometx copy WORKSPACE NEW-WORKSPACE
```

## 3. Copy specific experiments in a project to new experiments

Similarly, one can copy a single experiment first downloading it:

```shell
COMET_URL_OVERRIDE=http://comet.a.com/clientlib \
COMET_API_KEY=A-KEY \
cometx download WORKSPACE/PROJECT/EXPERIMENT-NAME-OR-ID
```

and then copying it:

```shell
COMET_URL_OVERRIDE=http://comet.b.com/clientlib \
COMET_API_KEY=B-KEY \
cometx copy WORKSPACE/PROJECT/EXPERIMENT-NAME-OR-ID NEW-WORKSPACE/NEW-PROJECT
```

### Subcommands

* [cometx list](#cometx-list)
* [cometx download](#cometx-download)
* [cometx copy](#cometx-copy)
* [cometx log](#cometx-log)
* [cometx reproduce](#cometx-reproduce)
* [cometx delete-assets](#cometx-delete-assets)
* [cometx config](#cometx-config)


For all subcommands, use the `--help` flag to get additional information.

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

* `--query` - use to limit the experiments to download (see [experiment.api.query()](https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/API/#apiquery) for more information)
*  `-u`, `--use-name` - use experiment names for experiment folders and listings

For more information, `cometx list --help`

## cometx copy

This command is used to:

* copy downloaded data to a new experiment
* create a symlink from one project to existing experiments

cometx copy examples:

```
cometx SOURCE DESTINATION
cometx --symlink SOURCE DESTINATION
```

where SOURCE is:

* if not `--symlink`, "WORKSPACE/PROJECT/EXPERIMENT", "WORKSPACE/PROJECT/*", or "WORKSPACE/*/*" folder (use quotes)
* if `--symlink`, then it is a Comet path to workspace or workspace/project

where DESTINATION is:

* WORKSPACE
* WORKSPACE/PROJECT

Not all combinations are possible:


| Destination → <br/>Source ↓ | WORKSPACE            | WORKSPACE/PROJECT      |
|--------------------|----------------------|------------------------|
| `WORKSPACE/*/*`      | Copies all projects  | N/A                    |
| `WORKSPACE/PROJ/*`   | N/A                  | Copies all experiments |
| `WORKSPACE/PROJ/EXP` | N/A                  | Copies experiment      |


For more information, `cometx copy --help`

## cometx download

This command is used to:

* download all workspaces, projects, and experiments of workspaces that you are a member of
* download all projects, and experiments of a given workspace
* download all experiments of a given workspace/project

cometx dowload examples:

```
cometx download WORKSPACE/PROJECT/EXPERIMENT-KEY-OR-NAME [RESOURCE ...] [FLAGS ...]
cometx download WORKSPACE/PROJECT [RESOURCE ...] [FLAGS ...]
cometx download WORKSPACE [RESOURCE ...] [FLAGS ...]
cometx download [RESOURCE ...] [FLAGS ...]
```

Where [RESOURCE ...] is zero or more of the following names:

* assets
* html
* metadata
* metrics
* others
* parameters
* project - alias for: project_notes, project_metadata
* run - alias for: code, git, output, graph, and requirements
* system

If no RESOURCE is given it will download all of them.

## Flags

* `--query` - use to limit the experiments to download (see [experiment.api.query()](https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/API/#apiquery) for more information)
* `--list` - use to list available workspaces, projects, experiments,
    artifacts, or models (same as `cometx list`)
* `--output` - download resources to folder other than current one
* `--flat` - don't use the normal hiearchy for downloaded items
* `--use-name` - use experiment names for folders and listings
* `--ignore` - don't download the following resources (use one or more
    RESOURCE names from above)
* `--asset-type` - asset type to match, or leave off to match all
* `--filename` - filename to match, or leave off to match all
* `--overwrite` - overwrite any existing files
* `--force` - don't ask to download, just do it
* `--help` - this message

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

For more information, `cometx download --help`


## cometx log

This command is used to log a resource (metrics, parameters, asset,
etc) file to a specific experiment or experiments.

cometx log example:

```
cometx log WORKSPACE/PROJECT/EXPERIMENT-KEY-OR-NAME [--type TYPE] FILENAME.EXT
```

where TYPE is the asset type of the filename. TYPE is not needed if the
filename extension (FILENAME.EXT) is known.

Known extensions:

* png
* jpg
* gif
* txt
* webm
* mp4
* ogg
* ipynb
* wav
* mp3

Known types:

* asset
* image
* text-sample
* video
* ipynb
* audio

Example to set an other key:value:

```
cometx log WORKSPACE/PROJECT --type other --set "key:value"
cometx log WORKSPACE/PROJECT/EXPERIMENT-KEY-OR-NAME --type other --set "key:value"
```
The first version will set the other key:value in all experiments in a project, and the second will set the other key:value in the experiment.


Example to log all items:

```
cometx log WORKSPACE/PROJECT PATH-TO-DOWNLOAD --type all
cometx log WORKSPACE/PROJECT/EXPERIMENT-KEY-OR-NAME PATH-TO-DOWNLOAD --type all
```
The first version will create an experiment, and the second will log everything to an existing experiment.

### Flags

* `--query` - use to limit the experiments to download (see [experiment.api.query()](https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/API/#apiquery) for more information)
* `--type` - the type of item to log
* `--set` - the "key:value" to log

For more information, `cometx log --help`

## cometx delete-assets

To delete experiments assets:

```
cometx delete-assets WORKSPACE/PROJECT --type=image
cometx delete-assets WORKSPACE/PROJECT/EXPERIMENT --type=all
```
Type can be valid asset tupe, including:

* all
* asset
* audio
* code
* image
* notebook
* text-sample
* video

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

## cometx reproduce

```
cometx reproduce [-h] [--run] [--executable EXECUTABLE] COMET_PATH OUTPUT_DIR
```

For more information, `cometx reproduce --help`

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
