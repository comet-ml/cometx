# Migrating Projects/Experiments

As described in the [README](README.md), you can instruct `cometx` to
download data from and copy data to:

* experiment to experiment
* project to project
* workspace to workspace
* Comet installation to Comet installation
* WandB installation to Comet installation

This is useful when you want to migrate projects or experiments
between different Comet instances or to Comet from different
vendors. This document describes how to do this.

By default, `cometx` connects to the Comet Cloud SaaS instance, unless you have
configured it to connect to a different Comet instance. If you want to migrate
projects or experiments between different Comet instances, you need to
configure `cometx` to connect to the source and destination Comet instances.

You can either change the configuration in the `cometx` configuration file
(`~/.comet.config`) or use the `COMET_API_KEY` and `COMET_URL_OVERRIDE`
environment variables to specify the source and destination Comet instances.
But you must ensure to either update the configuration file or change the
environment variables between the source and destination Comet instances before
compying data to the destination, so that `cometx` connects to the correct
Comet instance. This is because only one Comet instance can be configured at a
time.

## Migrating

Migrating your data is a two step process. First you must download the data
from the source, and then copy it to the destination Comet instance.

### Downloading Data

The first step in a migration is to use `cometx download`.
For example, to download from an existing Comet installation:

```shell
COMET_URL_OVERRIDE=http://comet.a.com/clientlib \
COMET_API_KEY=A-KEY \
cometx download <WORKSPACE>/<PROJECT>
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
COMET_URL_OVERRIDE=http://comet.a.com/clientlib \
COMET_API_KEY=A-KEY \
cometx download <WORKSPACE>/<PROJECT>/<EXPERIMENT_ID_OR_NAME>
```

##### Downloading an Entire Workspace

You can also omit the project name to download all of the projects in
a workspace:

```shell
COMET_URL_OVERRIDE=http://comet.a.com/clientlib \
COMET_API_KEY=A-KEY \
cometx download <WORKSPACE>
```

#### Filtering Resources

You can also filter the resources that are downloaded by specifying them as
arguments to the `download` subcommand:

```shell
COMET_URL_OVERRIDE=http://comet.a.com/clientlib \
COMET_API_KEY=A-KEY \
cometx download <WORKSPACE>/<PROJECT> [RESOURCE ...]
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

**Note that you need to be logged into wandb before downloading your
data.**

For example:

```shell
cometx download --from wandb stacey/yolo-drive/1dwb18ia
```

This will download the WandB run: https://wandb.ai/stacey/yolo-drive/runs/1dwb18ia

After download, the following `copy` commands will be relevant.

#### Additional Download Flags

These flags may be useful:

* `--sync SYNC` - if additional data has been logged at wandb since last download. This is the level to sync at: all, experiment, project, or workspace

### Copying Data

As noted above, the `download` subcommand will create a directory with
the same name as the project in the current working directory. You can
then use the `copy` subcommand to upload the data to the destination
Comet instance.

```shell
COMET_URL_OVERRIDE=http://comet.b.com/clientlib \
COMET_API_KEY=B-KEY \
cometx copy <WORKSPACE>/<PROJECT> <NEW-WORKSPACE>/<NEW-PROJECT>
```

Notice that we are using a different `COMET_URL_OVERRIDE` value than
before. This allows us to copy the downloaded data to a different
Comet installation.

Also note that `<WORKSPACE>/<PROJECT>` now refers to a directory, and
`<NEW-WORKSPACE>/<NEW-PROJECT>` refers to a workspace and project on
`http://comet.b.com`. The old and new workspaces and projects can be
the same. No experiment data will ever be overwritten, but rather new
experiments are always created.

##### Copying a Single Experiment

You can similarly copy a single experiment:

```shell
COMET_URL_OVERRIDE=http://comet.b.com/clientlib \
COMET_API_KEY=B-KEY \
cometx copy <WORKSPACE>/<PROJECT>/<EXPERIMENT_ID_OR_NAME> <NEW-WORKSPACE>/<NEW-PROJECT>
```

Note the absence of the experiment ID in the destination path.

##### Copy an Entire Workspace

As well as uploading an entire workspace:

```shell
COMET_URL_OVERRIDE=http://comet.b.com/clientlib \
COMET_API_KEY=B-KEY \
cometx copy <WORKSPACE> <NEW-WORKSPACE>
```
