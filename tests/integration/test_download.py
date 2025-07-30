# -*- coding: utf-8 -*-
# ******************************************
#                              __
#   _________  ____ ___  ___  / /__  __
#  / ___/ __ \/ __ `__ \/ _ \/ __/ |/_/
# / /__/ /_/ / / / / / /  __/ /__>  <
# \___/\____/_/ /_/ /_/\___/\__/_/|_|
#
#
# Copyright (C) 2022 Cometx Development Team
# All rights reserved.
# ******************************************

import os
import random
import subprocess
import sys
import tempfile

import comet_ml
from comet_ml.config import get_config
from comet_ml.utils import proper_registry_model_name
from mock import patch

from cometx.cli.download import main
from cometx.framework.comet import DownloadManager

from ..testlib import environ, until

THIS_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(THIS_DIR, "../model")
DIFF_CONTENTS = """diff --git a/repo-name/file.py b/repo-name/file.py
index ed4da9e..00e5b87 100644
--- a/repo-name/file.py
+++ b/repo-name/file.py
@@ -502,3 +502,27 @@

     def function(self):
         x = 40
+
+    def function(self):
+        x = 42
"""


def make_project_name():
    return "test-project-" + str(random.randint(1, 10000))


def command(command_list):
    output = subprocess.check_output(command_list)
    output = output.decode(sys.stdout.encoding)
    output = output.strip()
    return output.split("\n")


class TestDownload:
    @classmethod
    def setup_class(cls):
        cls.USER = os.environ.get("COMET_USER")
        if cls.USER is None:
            import pytest

            pytest.skip(
                "COMET_USER environment variable not set - skipping integration tests"
            )
        cls.WORKSPACE = get_config("comet.workspace") or cls.USER
        cls.API_KEY = get_config("comet.api_key")

        # create an experiment, and log everything
        cls.MODEL_NAME = "model-%s" % random.randint(1000000, 9000000)
        cls.ARTIFACT_NAME = "artifact-%s" % random.randint(1000000, 9000000)
        cls.PROPER_MODEL_NAME = proper_registry_model_name(cls.MODEL_NAME)

        # No cache:
        cls.api = comet_ml.API(api_key=cls.API_KEY, cache=False)
        cls.PROJECT_NAME = make_project_name()

        exp = comet_ml.Experiment(
            api_key=cls.API_KEY,
            workspace=cls.WORKSPACE,
            project_name=cls.PROJECT_NAME,
            log_git_metadata=False,
            log_git_patch=False,
            log_code=True,
        )
        cls.EXPERIMENT_ID = exp.id
        cls.EXPERIMENT_NAME = exp.name
        exp.log_model(cls.MODEL_NAME, MODEL_PATH)

        # Metrics
        for i in range(15):
            exp.log_metric("loss", random.random() * i, step=i)

        # Parameters
        exp.log_parameters({"learning_rate": 0.1, "hidden_layer_size": 150})
        # Assets
        exp.log_asset_folder(MODEL_PATH)
        # Artifacts
        artifact = comet_ml.Artifact(cls.ARTIFACT_NAME, "dataset")
        artifact.add(os.path.join(MODEL_PATH, "keras_module.txt"))
        exp.log_artifact(artifact)

        exp.end()

        assert until(
            lambda: cls.api.get(cls.WORKSPACE, cls.PROJECT_NAME, cls.EXPERIMENT_ID)
            is not None
        )
        cls.api_exp = cls.api.get(cls.WORKSPACE, cls.PROJECT_NAME, cls.EXPERIMENT_ID)

        assert until(
            lambda: cls.api.get_project(cls.WORKSPACE, cls.PROJECT_NAME) is not None
        )

        cls.PROJECT_ID = cls.api.get_project(cls.WORKSPACE, cls.PROJECT_NAME)[
            "projectId"
        ]

        # After everything has uploaded:
        exp = cls.api.get_experiment_by_key(exp.id)
        assert until(lambda: cls.MODEL_NAME in exp.get_model_names())
        exp.register_model(cls.MODEL_NAME, tags=["Production"])

    @classmethod
    def teardown_class(cls):
        cls.api.delete_project(cls.WORKSPACE, cls.PROJECT_NAME, delete_experiments=True)
        cls.api.delete_registry_model(cls.WORKSPACE, cls.MODEL_NAME)
        # TODO: remove artifact

    def setup_method(self):
        self.DIR = ""

    def download(self, comet_path, capsys, **kwargs):
        dm = DownloadManager(api_key=self.API_KEY)
        dm.download(comet_path, **kwargs)
        captured = capsys.readouterr()
        return captured.out.strip().split("\n")

    def test_download_workspaces_list(self, capsys):
        result = self.USER
        output = self.download(None, capsys, list_items=True)
        assert len(output) >= 0
        assert result in output

    def test_download_project_list(self, capsys):
        comet_path = "%s" % (self.WORKSPACE)
        result = "%s/%s" % (self.WORKSPACE, self.PROJECT_NAME)
        output = self.download(comet_path, capsys, list_items=True)
        assert len(output) >= 1  # projects
        assert result in output

    def test_download_experiment_list(self, capsys):
        comet_path = "%s/%s" % (self.WORKSPACE, self.PROJECT_NAME)
        result = "%s/%s/%s" % (self.WORKSPACE, self.PROJECT_NAME, self.EXPERIMENT_ID)
        output = self.download(comet_path, capsys, list_items=True)
        assert len(output) == 1
        assert result in output

    def test_download_experiment_list_2(self, capsys):
        comet_path = "%s/%s/%s" % (
            self.WORKSPACE,
            self.PROJECT_NAME,
            self.EXPERIMENT_ID,
        )
        result = "%s/%s/%s" % (self.WORKSPACE, self.PROJECT_NAME, self.EXPERIMENT_ID)
        output = self.download(comet_path, capsys, list_items=True)
        assert len(output) == 1
        assert result in output

    def test_download_artifact_list(self, capsys):
        comet_path = "%s/artifacts/%s" % (
            self.WORKSPACE,
            self.ARTIFACT_NAME,
        )
        results = [
            self.make_path("{ws}/artifacts/{aname}/1.0.0 (Latest)"),
        ]
        output = self.download(comet_path, capsys, list_items=True)
        for result in results:
            assert result in output

    def test_download_model_list(self, capsys):
        comet_path = "%s/model-registry/%s" % (
            self.WORKSPACE,
            self.MODEL_NAME,
        )
        results = [
            self.make_path("{ws}/model-registry/{mname}/1.0.0 (Production)"),
        ]
        output = self.download(comet_path, capsys, list_items=True)
        for result in results:
            assert result in output

    def make_path(self, comet_path):
        return comet_path.format(
            dir=self.DIR,
            ws=self.USER,
            proj=self.PROJECT_NAME,
            exp=self.EXPERIMENT_ID,
            ename=self.EXPERIMENT_NAME,
            mname=self.MODEL_NAME,
            aname=self.ARTIFACT_NAME,
        )

    @patch("comet_ml.API.get_project_notes")
    def test_download_project(self, mock_get_project_notes, capsys):
        mock_get_project_notes.return_value = "Project Notes Text"

        comet_path = "%s/%s" % (self.WORKSPACE, self.PROJECT_NAME)
        self.DIR = tempfile.mkdtemp()
        results = [
            self.make_path("{dir}/{ws}/{proj}/project_metadata.json"),
            self.make_path("{dir}/{ws}/{proj}/{exp}/metrics.jsonl"),
            self.make_path("{dir}/{ws}/{proj}/{exp}/metadata.json"),
            self.make_path("{dir}/{ws}/{proj}/{exp}/parameters.json"),
            self.make_path("{dir}/{ws}/{proj}/{exp}/others.jsonl"),
            self.make_path("{dir}/{ws}/{proj}/{exp}/system_details.json"),
            self.make_path("{dir}/{ws}/{proj}/{exp}/run/script.py"),
            self.make_path("{dir}/{ws}/{proj}/{exp}/assets/assets_metadata.jsonl"),
            self.make_path(
                "{dir}/{ws}/{proj}/{exp}/assets/model-element/model/keras_module.txt"
            ),
            self.make_path(
                "{dir}/{ws}/{proj}/{exp}/assets/model-element/model/model.h5"
            ),
            self.make_path("{dir}/{ws}/{proj}/{exp}/assets/asset/model.h5"),
            self.make_path("{dir}/{ws}/{proj}/{exp}/assets/asset/keras_module.txt"),
            # Via mock:
            self.make_path("{dir}/{ws}/{proj}/project_notes.md"),
        ]
        output = self.download(comet_path, capsys, output=self.DIR)
        assert len(output) > 0
        actual_results = command(["find", self.DIR])
        for result in results:
            assert result in actual_results

    def test_download_project_use_name(self, capsys):
        comet_path = "%s/%s" % (self.WORKSPACE, self.PROJECT_NAME)
        self.DIR = tempfile.mkdtemp()
        results = [
            self.make_path("{dir}/{ws}/{proj}/project_metadata.json"),
            self.make_path("{dir}/{ws}/{proj}/{ename}/metrics.jsonl"),
            self.make_path("{dir}/{ws}/{proj}/{ename}/metadata.json"),
            self.make_path("{dir}/{ws}/{proj}/{ename}/parameters.json"),
            self.make_path("{dir}/{ws}/{proj}/{ename}/others.jsonl"),
            self.make_path("{dir}/{ws}/{proj}/{ename}/system_details.json"),
            self.make_path("{dir}/{ws}/{proj}/{ename}/run/script.py"),
            self.make_path("{dir}/{ws}/{proj}/{ename}/assets/assets_metadata.jsonl"),
            self.make_path(
                "{dir}/{ws}/{proj}/{ename}/assets/model-element/model/keras_module.txt"
            ),
            self.make_path(
                "{dir}/{ws}/{proj}/{ename}/assets/model-element/model/model.h5"
            ),
            self.make_path("{dir}/{ws}/{proj}/{ename}/assets/asset/model.h5"),
            self.make_path("{dir}/{ws}/{proj}/{ename}/assets/asset/keras_module.txt"),
        ]
        output = self.download(comet_path, capsys, output=self.DIR, use_name=True)
        assert len(output) > 0
        actual_results = command(["find", self.DIR])
        for result in results:
            assert result in actual_results

    @patch("comet_ml.APIExperiment.get_model_graph")
    @patch("comet_ml.APIExperiment.get_output")
    @patch("comet_ml.APIExperiment.get_html")
    def test_download_experiment_id(self, mock_html, mock_output, mock_graph, capsys):
        mock_html.return_value = "<b>This is HTML</b>"
        mock_output.return_value = "This is output"
        mock_graph.return_value = "This the the model graph"

        comet_path = "%s/%s/%s" % (
            self.WORKSPACE,
            self.PROJECT_NAME,
            self.EXPERIMENT_ID,
        )
        self.DIR = tempfile.mkdtemp()
        results = [
            self.make_path("{dir}/{ws}/{proj}/{exp}/metrics.jsonl"),
            self.make_path("{dir}/{ws}/{proj}/{exp}/metadata.json"),
            self.make_path("{dir}/{ws}/{proj}/{exp}/parameters.json"),
            self.make_path("{dir}/{ws}/{proj}/{exp}/others.jsonl"),
            self.make_path("{dir}/{ws}/{proj}/{exp}/system_details.json"),
            self.make_path("{dir}/{ws}/{proj}/{exp}/run/script.py"),
            self.make_path("{dir}/{ws}/{proj}/{exp}/assets/assets_metadata.jsonl"),
            self.make_path(
                "{dir}/{ws}/{proj}/{exp}/assets/model-element/model/keras_module.txt"
            ),
            self.make_path(
                "{dir}/{ws}/{proj}/{exp}/assets/model-element/model/model.h5"
            ),
            self.make_path("{dir}/{ws}/{proj}/{exp}/assets/asset/model.h5"),
            self.make_path("{dir}/{ws}/{proj}/{exp}/assets/asset/keras_module.txt"),
            # Via mock:
            self.make_path("{dir}/{ws}/{proj}/{exp}/assets/html/experiment.html"),
            self.make_path("{dir}/{ws}/{proj}/{exp}/run/output.txt"),
            self.make_path("{dir}/{ws}/{proj}/{exp}/run/graph_definition.txt"),
        ]
        not_results = [
            self.make_path("{dir}/{ws}/{proj}/project_metadata.json"),
        ]
        output = self.download(comet_path, capsys, output=self.DIR)
        assert len(output) > 0
        actual_results = command(["find", self.DIR])
        for result in results:
            assert result in actual_results
        for result in not_results:
            assert result not in actual_results

    def test_download_experiment_id_alone(self, capsys):
        comet_path = "%s" % (self.EXPERIMENT_ID,)
        self.DIR = tempfile.mkdtemp()
        results = [
            self.make_path("{dir}/{ws}/{proj}/{exp}/metrics.jsonl"),
            self.make_path("{dir}/{ws}/{proj}/{exp}/metadata.json"),
            self.make_path("{dir}/{ws}/{proj}/{exp}/parameters.json"),
            self.make_path("{dir}/{ws}/{proj}/{exp}/others.jsonl"),
            self.make_path("{dir}/{ws}/{proj}/{exp}/system_details.json"),
            self.make_path("{dir}/{ws}/{proj}/{exp}/run/script.py"),
            self.make_path("{dir}/{ws}/{proj}/{exp}/assets/assets_metadata.jsonl"),
            self.make_path(
                "{dir}/{ws}/{proj}/{exp}/assets/model-element/model/keras_module.txt"
            ),
            self.make_path(
                "{dir}/{ws}/{proj}/{exp}/assets/model-element/model/model.h5"
            ),
            self.make_path("{dir}/{ws}/{proj}/{exp}/assets/asset/model.h5"),
            self.make_path("{dir}/{ws}/{proj}/{exp}/assets/asset/keras_module.txt"),
        ]
        not_results = [
            self.make_path("{dir}/{ws}/{proj}/project_metadata.json"),
        ]
        output = self.download(comet_path, capsys, output=self.DIR)
        assert len(output) > 0
        actual_results = command(["find", self.DIR])
        for result in results:
            assert result in actual_results
        for result in not_results:
            assert result not in actual_results

    def test_download_experiment_name(self, capsys):
        comet_path = "%s/%s/%s" % (
            self.WORKSPACE,
            self.PROJECT_NAME,
            self.EXPERIMENT_NAME,
        )
        self.DIR = tempfile.mkdtemp()
        results = [
            self.make_path("{dir}/{ws}/{proj}/{exp}/metrics.jsonl"),
            self.make_path("{dir}/{ws}/{proj}/{exp}/metadata.json"),
            self.make_path("{dir}/{ws}/{proj}/{exp}/parameters.json"),
            self.make_path("{dir}/{ws}/{proj}/{exp}/others.jsonl"),
            self.make_path("{dir}/{ws}/{proj}/{exp}/system_details.json"),
            self.make_path("{dir}/{ws}/{proj}/{exp}/run/script.py"),
            self.make_path("{dir}/{ws}/{proj}/{exp}/assets/assets_metadata.jsonl"),
            self.make_path(
                "{dir}/{ws}/{proj}/{exp}/assets/model-element/model/keras_module.txt"
            ),
            self.make_path(
                "{dir}/{ws}/{proj}/{exp}/assets/model-element/model/model.h5"
            ),
            self.make_path("{dir}/{ws}/{proj}/{exp}/assets/asset/model.h5"),
            self.make_path("{dir}/{ws}/{proj}/{exp}/assets/asset/keras_module.txt"),
        ]
        not_results = [
            self.make_path("{dir}/{ws}/{proj}/project_metadata.json"),
        ]
        output = self.download(comet_path, capsys, output=self.DIR)
        assert len(output) > 0
        actual_results = command(["find", self.DIR])
        for result in results:
            assert result in actual_results
        for result in not_results:
            assert result not in actual_results

    def test_download_experiment_flat(self, capsys):
        comet_path = "%s/%s/%s" % (
            self.WORKSPACE,
            self.PROJECT_NAME,
            self.EXPERIMENT_ID,
        )
        self.DIR = tempfile.mkdtemp()
        results = [
            self.make_path("{dir}/assets_metadata.jsonl"),
            self.make_path("{dir}/script.py"),
            self.make_path("{dir}/keras_module.txt"),
            self.make_path("{dir}/metadata.json"),
            self.make_path("{dir}/metrics.jsonl"),
            self.make_path("{dir}/model/keras_module.txt"),
            self.make_path("{dir}/model/model.h5"),
            self.make_path("{dir}/model.h5"),
            self.make_path("{dir}/others.jsonl"),
            self.make_path("{dir}/parameters.json"),
            self.make_path("{dir}/system_details.json"),
        ]
        output = self.download(comet_path, capsys, output=self.DIR, flat=True)
        assert len(output) > 0
        actual_results = command(["find", self.DIR])
        for result in results:
            assert result in actual_results

    def test_download_experiment_flat_ignore(self, capsys):
        comet_path = "%s/%s/%s" % (
            self.WORKSPACE,
            self.PROJECT_NAME,
            self.EXPERIMENT_ID,
        )
        self.DIR = tempfile.mkdtemp()
        results = [
            self.make_path("{dir}/assets_metadata.jsonl"),
            self.make_path("{dir}/keras_module.txt"),
            self.make_path("{dir}/metadata.json"),
            self.make_path("{dir}/metrics.jsonl"),
            self.make_path("{dir}/model/keras_module.txt"),
            self.make_path("{dir}/model/model.h5"),
            self.make_path("{dir}/model.h5"),
            self.make_path("{dir}/others.jsonl"),
            self.make_path("{dir}/parameters.json"),
            self.make_path("{dir}/system_details.json"),
        ]
        not_results = [
            self.make_path("{dir}/script.py"),
        ]
        output = self.download(
            comet_path,
            capsys,
            output=self.DIR,
            flat=True,
            ignore=["code"],
        )
        assert len(output) > 0
        actual_results = command(["find", self.DIR])
        for result in results:
            assert result in actual_results
        for result in not_results:
            assert result not in actual_results

    def test_download_model(self, capsys):
        comet_path = "%s/model-registry/%s" % (
            self.WORKSPACE,
            self.MODEL_NAME,
        )
        self.DIR = tempfile.mkdtemp()
        results = [
            self.make_path("{dir}/{ws}/model-registry/{mname}/model/keras_module.txt"),
            self.make_path("{dir}/{ws}/model-registry/{mname}/model/model.h5"),
        ]
        output = self.download(comet_path, capsys, output=self.DIR)
        assert len(output) > 0
        actual_results = command(["find", self.DIR])
        for result in results:
            assert result in actual_results

    def test_download_model_version(self, capsys):
        comet_path = "%s/model-registry/%s/1.0.0" % (
            self.WORKSPACE,
            self.MODEL_NAME,
        )
        self.DIR = tempfile.mkdtemp()
        results = [
            self.make_path("{dir}/{ws}/model-registry/{mname}/model/keras_module.txt"),
            self.make_path("{dir}/{ws}/model-registry/{mname}/model/model.h5"),
        ]
        output = self.download(comet_path, capsys, output=self.DIR)
        assert len(output) > 0
        actual_results = command(["find", self.DIR])
        for result in results:
            assert result in actual_results

    def test_download_model_tag(self, capsys):
        comet_path = "%s/model-registry/%s/production" % (
            self.WORKSPACE,
            self.MODEL_NAME,
        )
        self.DIR = tempfile.mkdtemp()
        results = [
            self.make_path("{dir}/{ws}/model-registry/{mname}/model/keras_module.txt"),
            self.make_path("{dir}/{ws}/model-registry/{mname}/model/model.h5"),
        ]
        output = self.download(comet_path, capsys, output=self.DIR)
        assert len(output) > 0
        actual_results = command(["find", self.DIR])
        for result in results:
            assert result in actual_results

    def test_download_artifact(self, capsys):
        comet_path = "%s/artifacts/%s" % (
            self.WORKSPACE,
            self.ARTIFACT_NAME,
        )
        self.DIR = tempfile.mkdtemp()
        results = [
            self.make_path("{dir}/{ws}/artifacts/{aname}/keras_module.txt"),
        ]
        output = self.download(comet_path, capsys, output=self.DIR)
        assert len(output) > 0
        actual_results = command(["find", self.DIR])
        for result in results:
            assert result in actual_results

    def test_download_artifact_latest(self, capsys):
        comet_path = "%s/artifacts/%s/latest" % (
            self.WORKSPACE,
            self.ARTIFACT_NAME,
        )
        self.DIR = tempfile.mkdtemp()
        results = [
            self.make_path("{dir}/{ws}/artifacts/{aname}/keras_module.txt"),
        ]
        output = self.download(comet_path, capsys, output=self.DIR)
        assert len(output) > 0
        actual_results = command(["find", self.DIR])
        for result in results:
            assert result in actual_results

    @patch("comet_ml.API.get_projects")
    def test_download_workspace(self, mock_get_projects, capsys):
        # Fake it, so we don't download all of self.USER
        # but just this one project
        mock_get_projects.return_value = [self.PROJECT_NAME]

        comet_path = "%s" % (self.WORKSPACE,)
        self.DIR = tempfile.mkdtemp()
        results = [
            self.make_path("{dir}/{ws}/{proj}/project_metadata.json"),
            self.make_path("{dir}/{ws}/{proj}/{exp}/metrics.jsonl"),
            self.make_path("{dir}/{ws}/{proj}/{exp}/metadata.json"),
            self.make_path("{dir}/{ws}/{proj}/{exp}/parameters.json"),
            self.make_path("{dir}/{ws}/{proj}/{exp}/others.jsonl"),
            self.make_path("{dir}/{ws}/{proj}/{exp}/system_details.json"),
            self.make_path("{dir}/{ws}/{proj}/{exp}/run/script.py"),
            self.make_path("{dir}/{ws}/{proj}/{exp}/assets/assets_metadata.jsonl"),
            self.make_path(
                "{dir}/{ws}/{proj}/{exp}/assets/model-element/model/keras_module.txt"
            ),
            self.make_path(
                "{dir}/{ws}/{proj}/{exp}/assets/model-element/model/model.h5"
            ),
            self.make_path("{dir}/{ws}/{proj}/{exp}/assets/asset/model.h5"),
            self.make_path("{dir}/{ws}/{proj}/{exp}/assets/asset/keras_module.txt"),
        ]
        output = self.download(comet_path, capsys, output=self.DIR)
        assert len(output) > 0
        actual_results = command(["find", self.DIR])
        for result in results:
            assert result in actual_results

    @patch("comet_ml.APIExperiment.get_git_patch")
    @patch("comet_ml.APIExperiment.get_git_metadata")
    def test_download_git(self, mock_get_git_metadata, mock_get_git_patch, capsys):

        mock_get_git_patch.return_value = DIFF_CONTENTS
        mock_get_git_metadata.return_value = {
            "user": self.USER,
            "root": "e16d344d042fda7e02cdefe337063bd902281962",
            "branch": "refs/heads/%s/branch-name" % self.USER,
            "parent": "0e9752d81146f299d344de2b6ac653c5cc601de2",
            "origin": "git@github.com:%s/repo-name" % self.USER,
        }

        comet_path = "%s/%s/%s" % (
            self.WORKSPACE,
            self.PROJECT_NAME,
            self.EXPERIMENT_ID,
        )
        self.DIR = tempfile.mkdtemp()
        results = [
            self.make_path("{dir}/{ws}/{proj}/{exp}/run/git_diff.patch"),
            self.make_path("{dir}/{ws}/{proj}/{exp}/run/git_metadata.json"),
            self.make_path("{dir}/{ws}/{proj}/{exp}/run/README.md"),
        ]
        output = self.download(comet_path, capsys, include=["git"], output=self.DIR)
        assert len(output) > 0
        actual_results = command(["find", self.DIR])
        for result in results:
            assert result in actual_results

    def test_download_cli(self, capsys):
        comet_path = "%s/%s" % (self.WORKSPACE, self.PROJECT_NAME)
        with environ({"COMET_API_KEY": self.API_KEY}):
            main([comet_path, "--list"])
        captured = capsys.readouterr()
        results = ["%s/%s/%s" % (self.WORKSPACE, self.PROJECT_NAME, self.EXPERIMENT_ID)]
        output = captured.out.strip().split("\n")
        for result in results:
            assert result in output
