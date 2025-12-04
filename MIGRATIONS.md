# Migrating Projects/Experiments

As described in the [README](https://github.com/comet-ml/cometx/blob/main/README.md), you can instruct `cometx` to
download data from and copy data to:

* experiment to experiment
* project to project
* workspace to workspace
* Comet installation to Comet installation
* WandB installation to Comet installation
* Neptune installation to Comet installation

This is useful when you want to migrate projects or experiments
between different Comet instances or to Comet from different
vendors. This document describes how to do this.

By default, `cometx` connects to the Comet Cloud SaaS instance, unless you have
configured it to connect to a different Comet instance. If you want to migrate
projects or experiments between different Comet instances, you need to
configure `cometx` to connect to the source and destination Comet instances.

You can either change the configuration in the `cometx` configuration file
(`~/.comet.config`) or use the `--url-override` and `--api-key` parameters to specify the source and destination Comet instances.
But you must ensure to either update the configuration file or change the
parameters between the source and destination Comet instances before
copying data to the destination, so that `cometx` connects to the correct
Comet instance. This is because only one Comet instance can be configured at a
time.

## Migrating

Migrating your data is a two step process. First you must download the data
from the source, and then copy it to the destination Comet instance.

> **Note**: if your installation does not use smart keys, you'll need to add the `--url-override=http://comet.X.com/clientlib` for the associated `--api-key=X-KEY`.

### Downloading Data

The first step in a migration is to use `cometx download`.
For example, to download from an existing Comet installation:

```shell
cometx --api-key A-KEY download <WORKSPACE>/<PROJECT>
```
See below for migrating from another vendor.

The `cometx download` subcommand downloads all of the Comet experiment
data into local files. Note that `<WORKSPACE>/<PROJECT>` refers to a
workspace and project on `http://comet.a.com`. This command will
create a folder in the filesystem with the same name:
`<WORKSPACE>/<PROJECT>`.

##### Downloading a Single Experiment

If you want to download a single experiment, you can specify the
experiment ID or experiment name in addition to the project name:

```shell
cometx --api-key A-KEY download <WORKSPACE>/<PROJECT>/<EXPERIMENT_ID_OR_NAME>
```

##### Downloading an Entire Workspace

You can also omit the project name to download all of the projects in
a workspace:

```shell
cometx --api-key A-KEY download <WORKSPACE>
```

#### Filtering Resources

You can also filter the resources that are downloaded by specifying them as
arguments to the `download` subcommand:

```shell
cometx --api-key A-KEY download <WORKSPACE>/<PROJECT> [RESOURCE ...]
```

Where `[RESOURCE ...]` is zero or more of the following names:

* `assets`
* `html`
* `metadata`
* `metrics`
* `others`
* `parameters`
* `project` - alias for: `project_notes`, `project_metadata`
* `run` - alias for: `code`, `git`, `output`, `graph`, and `requirements`
* system

If no `RESOURCE` is given it will download all of them.

#### Downloading from other Vendors

You can also download data from other vendors using the `--from`
flag. Currently, `cometx` supports:

* `--from wandb`
* `--from neptune`

**For WandB**: Note that you need to be logged into wandb before downloading your
data.

For example:

```shell
cometx download --from wandb stacey/yolo-drive/1dwb18ia
```

This will download the WandB run: https://wandb.ai/stacey/yolo-drive/runs/1dwb18ia

**For Neptune**: You must set the `NEPTUNE_API_TOKEN` environment variable before downloading.

For example:

```shell
export NEPTUNE_API_TOKEN="your-neptune-api-token"
cometx download --from neptune WORKSPACE/PROJECT
cometx download --from neptune WORKSPACE
```

The `--from neptune` option works like the other download frameworks and supports the same flags and resource filtering options.

After download, the following `copy` commands will be relevant.

#### Additional Download Flags

These flags may be useful:

* `--sync SYNC` - if additional data has been logged at the source (wandb, neptune, etc.) since last download. This is the level to sync at: all, experiment, project, or workspace

### Copying Data

As noted above, the `download` subcommand will create a directory with
the same name as the project in the current working directory. You can
then use the `copy` subcommand to upload the data to the destination
Comet instance.

```shell
cometx --api-key B-KEY copy <WORKSPACE>/<PROJECT> <NEW-WORKSPACE>/<NEW-PROJECT>
```

Note that you will need to add the associated`--url-override` values
for each installation that doesn't use smart keys.

Also note that `<WORKSPACE>/<PROJECT>` now refers to a directory, and
`<NEW-WORKSPACE>/<NEW-PROJECT>` refers to a workspace and project on
`http://comet.b.com`. The old and new workspaces and projects can be
the same. No experiment data will ever be overwritten, but rather new
experiments are always created.

##### Copying a Single Experiment

You can similarly copy a single experiment:

```shell
cometx --api-key B-KEY copy <WORKSPACE>/<PROJECT>/<EXPERIMENT_ID_OR_NAME> <NEW-WORKSPACE>/<NEW-PROJECT>
```

Note the absence of the experiment ID in the destination path.

##### Copy an Entire Workspace

As well as uploading an entire workspace:

```shell
cometx --api-key B-KEY copy <WORKSPACE> <NEW-WORKSPACE>
```

## Command Line Reference

### Download Command

The `cometx download` command downloads experiment data, artifacts, models, and panels from Comet or other vendors.

#### Basic Usage

```shell
cometx download [RESOURCE ...] [FLAGS ...]
cometx download WORKSPACE [RESOURCE ...] [FLAGS ...]
cometx download WORKSPACE/PROJECT [RESOURCE ...] [FLAGS ...]
cometx download WORKSPACE/PROJECT/EXPERIMENT-KEY [RESOURCE ...] [FLAGS ...]
```

#### Downloading Different Resource Types

**Experiments and Experiment Resources:**
```shell
cometx download WORKSPACE/PROJECT [RESOURCE ...]
cometx download WORKSPACE/PROJECT/EXPERIMENT-KEY [RESOURCE ...]
```

**Artifacts:**
```shell
cometx download WORKSPACE/artifacts/NAME
cometx download WORKSPACE/artifacts/NAME/VERSION-OR-ALIAS
```

**Models from Registry:**
```shell
cometx download WORKSPACE/model-registry/NAME
cometx download WORKSPACE/model-registry/NAME/VERSION-OR-STAGE
```

**Panels:**
```shell
cometx download WORKSPACE/panels/NAME-OR-ID
cometx download WORKSPACE/panels
```

#### Available Resources

For experiments, you can specify zero or more of these resource types:

* `run` - alias for: code, git, output, graph, and requirements
* `system`
* `others`
* `parameters`
* `metadata`
* `metrics`
* `assets`
* `html`
* `project` - alias for: project_notes, project_metadata

#### Download Options

* `--from from` - Source of data to download. Options: comet, wandb, or neptune. When using `--from neptune`, you must set the `NEPTUNE_API_TOKEN` environment variable.
* `-i IGNORE, --ignore IGNORE` - Resource(s) (or 'experiments') to ignore
* `-j PARALLEL, --parallel PARALLEL` - Number of threads to use for parallel downloading (default based on CPUs)
* `-o OUTPUT, --output OUTPUT` - Output directory for downloads
* `-u, --use-name` - Use experiment names for experiment folders and listings
* `-l, --list` - List items at this level rather than download
* `--flat` - Download files without subfolders
* `-f, --ask` - Query the user before proceeding (defaults to 'yes' if not included)
* `--filename FILENAME` - Only get resources ending with this filename
* `--query QUERY` - Only download experiments that match this Comet query string
* `--asset-type ASSET_TYPE` - Only get assets with this type
* `--sync SYNC` - What level to sync at: all, experiment, project, or workspace
* `--debug` - Provide debug info

#### Asset Types

The following asset types are supported:

* 3d-image
* 3d-points (deprecated)
* audio
* confusion-matrix (may contain assets)
* curve
* dataframe
* dataframe-profile
* datagrid
* embeddings (may reference image asset)
* histogram2d (not used)
* histogram3d (internal only, single histogram, partial logging)
* histogram_combined_3d
* image
* llm_data
* model-element
* notebook
* source_code
* tensorflow-model-graph-text (not used)
* text-sample
* video

### Copy Command

The `cometx copy` command copies experiment data to new experiments.

#### Basic Usage

```shell
cometx copy [--symlink] SOURCE DESTINATION
```

#### Source and Destination Combinations

| Destination:       | WORKSPACE            | WORKSPACE/PROJECT      |
|--------------------|----------------------|------------------------|
| WORKSPACE          | Copies all projects  | N/A                    |
| WORKSPACE/PROJ     | N/A                  | Copies all experiments |
| WORKSPACE/PROJ/EXP | N/A                  | Copies experiment      |

#### Source Types

* **Local folders** (when not using `--symlink`): "WORKSPACE/PROJECT/EXPERIMENT", "WORKSPACE/PROJECT", or "WORKSPACE" folder
* **Comet paths** (when using `--symlink`): workspace or workspace/project
* **Panels**: "WORKSPACE/panels" or "WORKSPACE/panels/PANEL-ZIP-FILENAME"

#### Destination Types

* WORKSPACE
* WORKSPACE/PROJECT

#### Copy Options

* `-i IGNORE, --ignore IGNORE` - Resource(s) (or 'experiments') to ignore
* `--debug` - If given, allow debugging
* `--quiet` - If given, don't display update info
* `--symlink` - Instead of copying, create a link to an experiment in a project
* `--sync` - Check to see if experiment name has been created first; if so, skip
