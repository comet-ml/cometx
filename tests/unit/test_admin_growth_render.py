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
                "charts": [{
                    "id": "chart-people-adoption",
                    "kind": "lines",
                    "title": "Adoption rate",
                    "data": {
                        "categories": ["overall", "opik"],
                        "labels": {"overall": "Overall", "opik": "Opik"},
                        "colors": ["--accent", "--ok"],
                        "points": [
                            {"key": "2026-06", "values": {"overall": 10.0, "opik": 4.0}},
                            {"key": "2026-07", "values": {"overall": 12.0, "opik": 6.0}},
                        ],
                        "window_start": None, "window_end": None,
                    },
                }],
                "table": None,
            }
        },
    }
    html = build_html(report)
    assert 'id="chart-people-adoption"' in html
    assert 'data-kind="lines"' in html
    assert "function drawLines" in html
    assert 'c.kind === "lines"' in html
