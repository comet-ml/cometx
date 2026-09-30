#!/usr/bin/env python
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
"""MPM presence for `cometx admin growth-report --mpm` (OPIK-8411).

The chargeback report carries no MPM data, so this collects it client-side
from existing endpoints and merges it into the chargeback payload's
`workspaces[]` as `mpmEnabled` / `monitoredModels` -- the shape
`admin_growth_users.parse_workspaces` reads.

A model counts as MPM-monitored when the registry flags it `is_monitored`
(and it is not pipeline-generated, a production-model reference, or deleted).
Both sources below apply exactly that predicate server-side:

1. `GET /api/mpm/v3/workspaces` -- one call returning every workspace the
   API key's user is a MEMBER of, with its monitored models.
2. For the remaining workspaces (an org admin need not be a member), the REST
   v2 registry: list the workspace's models, then read each model's details
   for its monitored flag. Org admins may read any workspace this way,
   private models included. One request per model, so it runs in a pool.

   Access caveat: for a non-member workspace, a user who is NOT an org admin
   gets only its public models, with no error -- so private monitored models
   are missed and the workspace still reads as checked. This is a different
   rule from chargeback's (server admin list, or org admin on-prem), so a key
   that passes chargeback is not necessarily complete here.

A workspace whose lookup fails is reported as unknown (`None`), never as
"no MPM": a false zero would read as a workspace that stopped using MPM.
`nb_models_registered` (the monthly usage report) is deliberately NOT used --
it counts every registry model and overstates MPM adoption.
"""

from __future__ import annotations

import concurrent.futures

from cometx.utils import admin_api_url

MPM_WORKSPACES_PATH = "/api/mpm/v3/workspaces"
DEFAULT_MAX_WORKERS = 8


def _mpm_url(api) -> str:
    """URL of `mpm/v3/workspaces`, beside (not under) the SDK's `/clientlib`
    root -- `admin_api_url` drops that segment and keeps any deployment
    prefix."""
    return admin_api_url(api.config["comet.url_override"], MPM_WORKSPACES_PATH)


def _fetch_member_workspaces(api) -> "dict[str, list[dict]] | None":
    """`mpm/v3/workspaces` -> {workspace_name: [{"id", "name"}, ...]} for the
    workspaces the caller belongs to. `None` on any failure (MPM disabled on
    the deployment, network error, unexpected shape): the registry path then
    covers every workspace instead."""
    try:
        url = _mpm_url(api)
        response = api._client.get(
            url, headers={"Authorization": api.api_key}, params={}
        )
        payload = response.json()
        out = {}
        for ws in payload["workspaces"]:
            name = ws.get("workspaceName")
            if not name:
                continue
            out[name] = [
                {"id": m.get("modelId"), "name": m.get("modelName")}
                for m in (ws.get("models") or [])
                if isinstance(m, dict)
            ]
        return out
    except Exception:
        return None


def _is_monitored(details) -> "bool | None":
    """The monitored flag from a registry-model details response. The Java
    field is `boolean isMonitored` without an explicit `@JsonProperty`, so
    Jackson may serialize it as `monitored`; accept either. `None` when
    neither is a real bool (the answer is unknown, not False)."""
    if not isinstance(details, dict):
        return None
    for key in ("isMonitored", "monitored"):
        value = details.get(key)
        if isinstance(value, bool):
            return value
    return None


def _registry_monitored_models(
    api, workspaces, max_workers=DEFAULT_MAX_WORKERS
) -> "dict[str, list[dict] | None]":
    """Monitored models per workspace via the REST v2 registry. Two parallel
    phases -- list every workspace's models, then read every model's details
    -- so neither many small workspaces nor one large one runs serially.

    A workspace is `None` when its listing fails, any of its detail lookups
    fails, or a model's flag cannot be read: a partial list would silently
    undercount."""

    def list_models(ws):
        return ws, api._client.get_registry_models(ws) or []

    def check(pair):
        ws, name = pair
        return ws, name, _is_monitored(api._client.get_registry_model_details(ws, name))

    out: "dict[str, list[dict] | None]" = {}
    ids: "dict[tuple, str]" = {}
    pairs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        listed = {pool.submit(list_models, ws): ws for ws in workspaces}
        for future in concurrent.futures.as_completed(listed):
            ws = listed[future]
            try:
                _ws, models = future.result()
            except Exception:
                out[ws] = None
                continue
            out[ws] = []
            for m in models:
                name = m.get("modelName") if isinstance(m, dict) else None
                if name:
                    ids[(ws, name)] = m.get("registryModelId")
                    pairs.append((ws, name))

        checked = {pool.submit(check, pair): pair for pair in pairs}
        for future in concurrent.futures.as_completed(checked):
            ws, name = checked[future]
            if out.get(ws) is None:
                continue  # already failed
            try:
                _ws, _name, flag = future.result()
            except Exception:
                flag = None
            if flag is None:
                out[ws] = None
            elif flag:
                out[ws].append({"id": ids.get((ws, name)), "name": name})

    # Stable order regardless of completion order.
    for ws, models in out.items():
        if models:
            models.sort(key=lambda m: m["name"])
    return out


def fetch_mpm_presence(
    api, workspace_names, max_workers=DEFAULT_MAX_WORKERS
) -> "dict[str, list[dict] | None]":
    """Monitored models per workspace: {name: [{"id", "name"}, ...]}, or
    {name: None} where it could not be determined. Workspaces in
    `mpm/v3/workspaces` are answered from that single call; the rest fall
    back to the registry."""
    names = list(dict.fromkeys(n for n in workspace_names if n))
    member = _fetch_member_workspaces(api) or {}
    presence = {n: member[n] for n in names if n in member}
    remaining = [n for n in names if n not in presence]
    if remaining:
        print(
            "Checking MPM models in %d workspace(s) via the model registry..."
            % len(remaining)
        )
        presence.update(
            _registry_monitored_models(api, remaining, max_workers=max_workers)
        )
    return presence


def apply_mpm_presence(chargeback, presence) -> dict:
    """Return a copy of `chargeback` whose workspaces carry `mpmEnabled` /
    `monitoredModels` from `presence`. Workspaces that are unknown (`None`)
    or absent from `presence` end up WITHOUT the fields -- any already in the
    input (e.g. a hand-edited `--chargeback-report` file) are removed rather
    than passed through as current -- which `parse_workspaces` reads as
    "not reported"."""
    workspaces = []
    for w in chargeback.get("workspaces") or []:
        models = presence.get(w.get("name"))
        if models is None:
            workspaces.append(
                {
                    k: v
                    for k, v in w.items()
                    if k not in ("mpmEnabled", "monitoredModels")
                }
            )
        else:
            workspaces.append(
                {**w, "mpmEnabled": bool(models), "monitoredModels": list(models)}
            )
    return {**chargeback, "workspaces": workspaces}
