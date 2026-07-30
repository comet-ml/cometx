def test_lines_kind_registered_and_panel_renders():
    from cometx.cli.admin_growth_render import build_html

    report = {
        "meta": {"title": "T", "generated": "2026-07-01", "source": "x"},
        "window": {"label": "w"},
        "collectors": {"opik": False, "em": False, "mpm": False},
        "sections": {
            "people": {
                "title": "Users / People",
                "kpis": [],
                "charts": [
                    {
                        "id": "chart-people-adoption",
                        "kind": "lines",
                        "title": "Adoption rate",
                        "data": {
                            "categories": ["overall", "opik"],
                            "labels": {"overall": "Overall", "opik": "Opik"},
                            "colors": ["--accent", "--ok"],
                            "points": [
                                {
                                    "key": "2026-06",
                                    "values": {"overall": 10.0, "opik": 4.0},
                                },
                                {
                                    "key": "2026-07",
                                    "values": {"overall": 12.0, "opik": 6.0},
                                },
                            ],
                            "window_start": None,
                            "window_end": None,
                        },
                    }
                ],
                "table": None,
            }
        },
    }
    html = build_html(report)
    assert 'id="chart-people-adoption"' in html
    assert 'data-kind="lines"' in html
    assert "function drawLines" in html
    assert 'c.kind === "lines"' in html


def test_barsline_kind_registered_and_panel_renders():
    from cometx.cli.admin_growth_render import build_html

    report = {
        "meta": {"title": "T", "generated": "2026-07-01", "source": "x"},
        "window": {"label": "w"},
        "sections": {
            "unified": {
                "title": "Organization overview (chargeback)",
                "kpis": [],
                "charts": [
                    {
                        "id": "chart-unified-workspace-churn",
                        "kind": "barsLine",
                        "title": "Workspaces added vs. deleted",
                        "data": {
                            "points": [
                                {
                                    "key": "2026-06",
                                    "values": {
                                        "added": 3,
                                        "deleted": 1,
                                        "rate": 50.0,
                                    },
                                },
                                {
                                    "key": "2026-07",
                                    "values": {
                                        "added": 2,
                                        "deleted": 0,
                                        "rate": 33.3,
                                    },
                                },
                            ],
                            "bars": ["added", "deleted"],
                            "line": "rate",
                            "bar_colors": ["--ok", "--warn"],
                            "line_color": "--accent",
                            "line_suffix": "%",
                        },
                    }
                ],
                "table": None,
            }
        },
    }
    html = build_html(report)
    assert 'id="chart-unified-workspace-churn"' in html
    assert 'data-kind="barsLine"' in html
    assert "function drawBarsLine" in html
    assert 'c.kind === "barsLine"' in html
    # right-axis helper for the secondary (rate) scale is present and wired
    assert "function rAxis" in html
    assert "rAxis(svg, YR" in html
    # tooltips are attached to the bars+line chart
    assert "attachTip(host, svg, cols)" in html


def test_groupedbarsh_kind_registered_and_panel_renders():
    from cometx.cli.admin_growth_render import build_html

    report = {
        "meta": {"title": "T", "generated": "2026-07-01", "source": "x"},
        "window": {"label": "w"},
        "sections": {
            "unified": {
                "title": "Organization overview (chargeback)",
                "kpis": [],
                "charts": [
                    {
                        "id": "chart-unified-platform-mix",
                        "kind": "groupedBarsH",
                        "title": "Workspace platform mix",
                        "data": {
                            "rows": [
                                {"label": "EM only", "value": 5},
                                {"label": "Opik only", "value": 2},
                            ]
                        },
                    }
                ],
                "table": None,
            }
        },
    }
    html = build_html(report)
    assert 'id="chart-unified-platform-mix"' in html
    assert 'data-kind="groupedBarsH"' in html
    assert "function drawGroupedH" in html
    assert 'c.kind === "groupedBarsH"' in html


def test_leaderboards_and_personal_vs_service_sections_reach_body():
    from cometx.cli.admin_growth_render import build_html

    report = {
        "meta": {"title": "T", "generated": "2026-07-01", "source": "x"},
        "window": {"label": "w"},
        "sections": {
            "leaderboards": {
                "title": "Leaderboards",
                "kpis": [],
                "charts": [],
                "table": {
                    "title": "Top users by activity",
                    "headers": ["User", "Experiments"],
                    "rows": [["alice", 42]],
                },
            },
            "personal_vs_service": {
                "title": "Personal vs. service accounts",
                "kpis": [],
                "charts": [],
                "table": {
                    "title": "Split",
                    "headers": ["Type", "Count"],
                    "rows": [["Personal", 10], ["Service", 3]],
                },
            },
        },
    }
    html = build_html(report)
    body = html.split('id="report-data"')[0]
    # Both section titles and their table content land in the rendered body,
    # not merely in the embedded JSON payload.
    assert "Leaderboards" in body
    assert "Top users by activity" in body
    assert "alice" in body
    assert "Personal vs. service accounts" in body
    assert "Service" in body
