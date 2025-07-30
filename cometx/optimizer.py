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

import collections
import functools
import hashlib
import json

from comet_ml import API, Experiment, Optimizer
from comet_ml.json_encoder import NestedEncoder
from comet_ml.query import Other

DATABASE = {}


def pid_from_params(params):
    """Compute an ID based on params"""
    # First convert float ints to ints to standardize:
    for key in params:
        if isinstance(params[key], float):
            if params[key] == int(params[key]):
                params[key] = int(params[key])
    return hashlib.sha1(
        json.dumps(
            params, separators=(",", ":"), sort_keys=True, cls=NestedEncoder
        ).encode("utf-8")
    ).hexdigest()


def update_dict(orig, update):
    """
    Recursive function to update a dict in place,
    and return it.
    """
    for key, value in update.items():
        if isinstance(value, collections.abc.Mapping):
            orig[key] = update_dict(orig.get(key, {}), value)
        else:
            orig[key] = value
    return orig


def get_value(other, summary):
    """
    Get the first item's field from a list of dicts
    """
    return [s["valueCurrent"] for s in summary if s["name"] == other][0]


def check_name(other, summary):
    """
    Check to see if other is in summary
    """
    return len([s for s in summary if s["name"] == other]) > 0


def optimizer_insert(self, opt_id, pid, trial, status, score):
    """
    Monkey-patched comet_ml/connection.py OptimizerAPI.optimizer_update()
    """
    p = DATABASE[pid]
    p.update(
        {
            "id": opt_id,
            "pid": pid,
            "trial": trial,
            "status": status,
            "score": score,
        }
    )
    payload = {
        "id": opt_id,
    }
    json_data = {
        "id": opt_id,
        "pid": p["pid"],
        "trial": p["trial"],
        "status": p["status"],
        "score": p["score"],
        "epoch": p["epoch"],
        "parameters": p["parameters"],
        "tries": p["tries"],
        "retries": p["retries"],
        "startTime": p["startTime"],
        "endTime": p["endTime"],
        "lastUpdateTime": p["lastUpdateTime"],
        "count": p["count"],
    }
    results = self.post_request("insert", payload=payload, json=json_data)
    return results


def optimizer_populate(workspace, project_name, config):
    # Clear the evidence:
    DATABASE.clear()
    # First, get the original config:
    original_optimizer = Optimizer(config["id"])
    original_config = original_optimizer.status()

    new_config = {
        "algorithm": original_config["algorithm"],
        "name": original_config["name"],
        "parameters": original_config["parameters"],
        "spec": original_config["spec"],
        "trials": original_config["trials"],
    }

    # Update the new_config with the given config:
    update_dict(new_config, config["config"])
    if "algorithm" in config:
        new_config["algorithm"] = config["algorithm"]

    print("New config:")
    print(new_config)

    new_optimizer = Optimizer(new_config)

    # Monkey patch to workaround broken insert
    original_func = new_optimizer._api.optimizer_update
    new_optimizer._api.optimizer_update = functools.partial(
        optimizer_insert, new_optimizer._api
    )
    # Will undo monkey patch after inserts

    api = API()
    other_names = [
        "optimizer_trial",
        "optimizer_metric",
        "optimizer_metric_value",
        "optimizer_parameters",
        "optimizer_trial",
    ]
    parameter_names = [
        "curr_step",
        "curr_epoch",
    ]

    # Gather evidence from previous sweeps:
    count = 0
    for e_workspace, e_project_name, optimizer_id in config["evidence"]:
        experiments = api.query(
            e_workspace, e_project_name, Other("optimizer_id") == optimizer_id
        )
        # Get possible evidence for new sweep:
        for experiment in experiments:
            others = experiment.get_others_summary()
            if not all([check_name(name, others) for name in other_names]):
                print(
                    "Missing other optimizer elements; skipping %r..." % experiment.name
                )
                continue
            parameters = experiment.get_parameters_summary()
            if not all([check_name(name, parameters) for name in parameter_names]):
                print("Missing parameter elements; skipping %r..." % experiment.name)
                continue
            metadata = experiment.get_metadata()
            # Metadata:
            start_time = metadata["startTimeMillis"]
            end_time = metadata["endTimeMillis"]
            # Parameters:
            step = int(get_value("curr_step", parameters))
            epoch = int(get_value("curr_epoch", parameters))
            # Others:
            optimizer_trial = int(get_value("optimizer_trial", others))
            metric_name = get_value("optimizer_metric", others)
            metric_value = get_value("optimizer_metric_value", others)
            optimizer_trial = int(get_value("optimizer_trial", others))
            optimizer_parameters = json.loads(get_value("optimizer_parameters", others))
            # FIXME: Make sure parameters are correct; what is criteria?
            # Metrics:
            metrics = experiment.get_metrics(metric_name)
            # Log it:
            new_pid = pid_from_params(optimizer_parameters)
            count += 1
            DATABASE[new_pid] = {
                "epoch": epoch,
                "parameters": optimizer_parameters,
                "retries": 0,
                "tries": 1,
                "startTime": start_time,
                "endTime": end_time,
                "lastUpdateTime": end_time,
                "count": count,
            }
            new_experiment = Experiment(
                workspace=workspace,
                project_name=project_name,
                log_code=False,
                log_env_details=False,
                auto_output_logging=None,
            )
            for metric in metrics:
                new_experiment.log_metric(
                    metric_name,
                    metric["metricValue"],
                    metric["step"],
                    metric["epoch"],
                    metric["timestamp"],
                )
            optimizer_data = {
                "id": new_optimizer.id,
                "pid": new_pid,
                "trial": optimizer_trial,
                "count": count,
                "parameters": optimizer_parameters,
            }
            new_experiment._set_optimizer_from_data(optimizer_data)
            new_experiment.optimizer["optimizer"] = new_optimizer
            new_experiment.log_metric(metric_name, float(metric_value), step=step)
            new_experiment.end()

    new_optimizer._api.optimizer_update = original_func

    print("Your new optimizer id is:")
    print("%r" % new_optimizer.id)
    print("%s/%s was populated with %d experiments" % (workspace, project_name, count))
    print("You are now ready to continue to run experiments.")
    return new_optimizer.id
