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
   API key's user is a MEMBER of, with its monitored models. The path is
   fixed on purpose: it is a backend-react route (registered in
   ReactWebappServerApplication), not an MPM-service one, so it resolves the
   same way on Cloud and on chart deployments. If it fails, the failure is
   classified (refused / not_found / error), warned about, and recorded, so
   "MPM not installed" is distinguishable from "key refused".
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

from cometx.utils import (
    admin_api_url,
    exception_text,
    http_error_status,
    redact_url_userinfo,
)

MPM_WORKSPACES_PATH = "/api/mpm/v3/workspaces"
DEFAULT_MAX_WORKERS = 8

# Outcome of the `mpm/v3/workspaces` call, kept so the report and CSV can tell
# "MPM not installed" apart from "the key was refused". Every value but
# LOOKUP_OK means the registry covered every workspace instead.
LOOKUP_OK = "ok"
LOOKUP_REFUSED = "refused"  # 401/403
LOOKUP_NOT_FOUND = "not_found"  # 404: MPM not installed, or not routed
LOOKUP_ERROR = "error"  # anything else: 5xx, network, unexpected shape


def _mpm_url(api) -> str:
    """URL of `mpm/v3/workspaces`, beside (not under) the SDK's `/clientlib`
    root -- `admin_api_url` drops that segment and keeps any deployment
    prefix."""
    return admin_api_url(api.config["comet.url_override"], MPM_WORKSPACES_PATH)


def _lookup_warning(lookup, exc, url) -> str:
    """Operator-facing warning for a failed `mpm/v3/workspaces` call."""
    where = redact_url_userinfo(url) if url else MPM_WORKSPACES_PATH
    detail = exception_text(exc)
    if lookup == LOOKUP_REFUSED:
        return (
            "Warning: the MPM API refused this key at %s (%s). Checking every "
            "workspace via the model registry instead. There, workspaces the "
            "key's user is not a member of show only public models unless "
            "the user is an organization admin, so counts may be low." % (where, detail)
        )
    if lookup == LOOKUP_NOT_FOUND:
        return (
            "Warning: the MPM API was not found at %s (%s); MPM may not be "
            "installed or routed on this deployment. Checking every workspace "
            "via the model registry instead." % (where, detail)
        )
    return (
        "Warning: the MPM API call to %s failed (%s). Checking every "
        "workspace via the model registry instead." % (where, detail)
    )


def _fetch_member_workspaces(api) -> "tuple[dict[str, list[dict]] | None, str]":
    """`mpm/v3/workspaces` -> ({workspace_name: [{"id", "name"}, ...]}, lookup)
    for the workspaces the caller belongs to.

    On failure the dict is `None` (the registry path then covers every
    workspace) and `lookup` says why -- LOOKUP_REFUSED, LOOKUP_NOT_FOUND or
    LOOKUP_ERROR -- with a warning printed. Failures are classified rather
    than swallowed because the fallback is only as complete as the key's
    registry access: a refused key that also isn't an org admin would
    otherwise yield a confident-looking low count instead of a visible
    problem."""
    url = None
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
    except Exception as exc:
        status = http_error_status(exc)
        if status in (401, 403):
            lookup = LOOKUP_REFUSED
        elif status == 404:
            lookup = LOOKUP_NOT_FOUND
        else:
            lookup = LOOKUP_ERROR
        print(_lookup_warning(lookup, exc, url))
        return None, lookup
    return out, LOOKUP_OK


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
) -> "tuple[dict[str, list[dict] | None], str]":
    """(presence, lookup). `presence` maps each workspace to its monitored
    models, [{"id", "name"}, ...], or to `None` where they could not be
    determined. Workspaces in `mpm/v3/workspaces` are answered from that
    single call; the rest fall back to the registry. `lookup` is how that
    call went (LOOKUP_OK or a failure class; see `_fetch_member_workspaces`)."""
    names = list(dict.fromkeys(n for n in workspace_names if n))
    member, lookup = _fetch_member_workspaces(api)
    member = member or {}
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
    return presence, lookup


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
