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
"""cometx admin growth-report — cross-platform use-case growth & rates, per
workspace/department (Opik + EM + MPM), rendered as a self-contained HTML page.

Distinct from `admin usage-report` (experiment counts over time, PDF/Streamlit):
growth-report tracks cross-platform use-case creation growth and rates.
"""

from __future__ import annotations


def generate_growth_report(
    api,
    workspaces,
    *,
    window="7d",
    units="month",
    platforms="em,opik,mpm",
    output="growth_report.html",
    no_open=False,
    limit=None,
):
    reporter = GrowthReporter(
        api, window=window, units=units, platforms=platforms, limit=limit
    )
    report_data = reporter.build(workspaces)  # events + usage -> report_data (C2-C7)
    path = write_growth_html(report_data, output)  # C8
    if not no_open:
        _open(path)
    return path


class GrowthReporter:
    def __init__(self, api, *, window, units, platforms, limit=None):
        self.api = api
        self.window = window
        self.units = units
        self.platforms = platforms
        self.limit = limit

    def build(self, workspaces):
        raise NotImplementedError  # filled in C2-C7


def write_growth_html(report_data, output):
    raise NotImplementedError  # filled in C8


def _open(path):
    import webbrowser

    webbrowser.open(f"file://{path}")
