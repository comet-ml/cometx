# -*- coding: utf-8 -*-
import json

import pytest
from comet_ml.api_objects.view import View

from cometx.views import (
    GRID_WIDTH,
    Section,
    all_of,
    any_of,
    bar_panel,
    build_experiment_view,
    build_view,
    custom_panel,
    data_panel,
    describe_view,
    global_config,
    image_panel,
    javascript_panel,
    legend_key,
    line_panel,
    new_id,
    parallel_panel,
    python_panel,
    query_state,
    rule,
    scalar_panel,
    scatter_panel,
    table_state,
)


class TestNewId:
    def test_new_id__default__six_alphanumeric_chars(self):
        value = new_id()

        assert len(value) == 6
        assert value.isalnum()

    def test_new_id__repeated_calls__unique(self):
        assert len({new_id() for _ in range(100)}) == 100


class TestPanelFactories:
    def test_line_panel__single_metric__wrapped_in_list(self):
        panel = line_panel("loss")

        assert panel["chartType"] == "line"
        assert panel["metricNames"] == ["loss"]
        assert panel["selectedYAxis"] == "loss"
        assert panel["selectedXAxis"] == "step"

    def test_line_panel__several_metrics__first_is_y_axis(self):
        panel = line_panel(["loss", "val_loss"], smoothing=0.6, transform_y="log")

        assert panel["metricNames"] == ["loss", "val_loss"]
        assert panel["selectedYAxis"] == "loss"
        assert panel["smoothingY"] == 0.6
        assert panel["transformY"] == "log scale"

    def test_line_panel__no_legend__defaults_to_experiment_name(self):
        assert line_panel("loss")["selectedLegendKeys"] == [legend_key("Name")]

    def test_line_panel__empty_y__raises_value_error(self):
        with pytest.raises(ValueError):
            line_panel([])

    @pytest.mark.parametrize(
        "factory",
        [
            lambda **kw: line_panel("loss", **kw),
            lambda **kw: bar_panel("accuracy", **kw),
            lambda **kw: scatter_panel("x", "y", **kw),
        ],
    )
    def test_panel_factories__empty_legend__kept_empty(self, factory):
        assert factory(legend=[])["selectedLegendKeys"] == []

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: line_panel("loss"),
            lambda: bar_panel("accuracy"),
            lambda: scatter_panel("x", "y"),
        ],
    )
    def test_panel_factories__default_legend_mutated__other_panels_unaffected(
        self, factory
    ):
        first = factory()
        first["selectedLegendKeys"][0]["label"] = "changed"
        first["selectedLegendKeys"].append(legend_key("lr", source="params"))

        assert factory()["selectedLegendKeys"] == [legend_key("Name")]

    def test_panel_factories__each_call__fresh_chart_id(self):
        assert line_panel("loss")["chartId"] != line_panel("loss")["chartId"]

    def test_bar_panel__group_by__grouping_enabled(self):
        panel = bar_panel(
            "accuracy",
            plot_type="BOX",
            group_by="optimizer",
            group_by_aggregation="median",
        )

        assert panel["plotType"] == "BOX_PLOT"
        assert panel["grouping"] == {
            "enabled": True,
            "plotType": "BOX_PLOT",
            "aggregation": "median",
            "groupByParameter": "optimizer",
        }

    def test_bar_panel__plot_types__stored_as_ui_values(self):
        assert bar_panel("a", plot_type="violin")["plotType"] == "VIOLIN_PLOT"
        assert bar_panel("a", plot_type="STACKED_BAR")["plotType"] == "STACKED_BAR"
        assert bar_panel("a", plot_type="BOX_PLOT")["plotType"] == "BOX_PLOT"

    def test_bar_panel__unknown_plot_type__raises(self):
        with pytest.raises(ValueError):
            bar_panel("a", plot_type="PIE")

    def test_bar_panel__unknown_group_by_aggregation__raises(self):
        with pytest.raises(ValueError):
            bar_panel("a", group_by="optimizer", group_by_aggregation="sum")

    def test_line_panel__transforms__stored_as_ui_values(self):
        assert line_panel("a", transform_y="ln")["transformY"] == "log(value)"
        assert line_panel("a", transform_y="log10")["transformY"] == "log(10)"
        assert line_panel("a", transform_y="log scale")["transformY"] == "log scale"
        assert line_panel("a")["transformY"] is None

    def test_line_panel__unknown_transform__raises(self):
        with pytest.raises(ValueError):
            line_panel("a", transform_y="sqrt")

    def test_line_panel__hide_outliers__stored_as_ignore(self):
        assert line_panel("a", outliers="hide")["selectedOutliers"] == "ignore"

    def test_scalar_panel__empty_description__kept_as_custom(self):
        panel = scalar_panel("a", description="")

        assert panel["description"] == ""
        assert panel["isCustomDescription"] is True

    def test_scalar_panel__average__stored_as_avg(self):
        assert scalar_panel("a", aggregation="average")["aggregation"] == "avg"

    def test_scalar_panel__unknown_aggregation__raises(self):
        with pytest.raises(ValueError):
            scalar_panel("a", aggregation="biggest")

    def test_global_config__log__stored_as_log_scale(self):
        assert global_config(transform_y="log")["transformY"] == "log scale"

    def test_bar_panel__no_grouping__grouping_disabled(self):
        assert bar_panel("accuracy")["grouping"]["enabled"] is False

    def test_scatter_panel__no_z__two_dimensional(self):
        panel = scatter_panel("learning_rate", "accuracy")

        assert "selectedZAxis" not in panel
        assert panel["metricNames"] == ["learning_rate", "accuracy"]

    def test_scatter_panel__z__three_dimensional(self):
        panel = scatter_panel("a", "b", z="c", aggregation="max")

        assert panel["selectedZAxis"] == "c"
        assert panel["aggregationZ"] == "max"
        assert panel["metricNames"] == ["a", "b", "c"]

    def test_scatter_panel__explicit_metrics__used_as_metric_names(self):
        panel = scatter_panel("lr", "acc", metrics=["acc"], params=["lr"])

        assert panel["metricNames"] == ["acc"]
        assert panel["params"] == ["lr"]

    def test_parallel_panel__no_target__first_metric_is_target(self):
        panel = parallel_panel(["lr", "batch_size"], metrics=["loss", "acc"])

        assert panel["selectedTargetVariable"] == "loss"
        assert panel["params"] == ["lr", "batch_size"]

    def test_parallel_panel__no_metrics__empty_target(self):
        assert parallel_panel(["lr"])["selectedTargetVariable"] == ""

    def test_scalar_panel__no_description__generated_wording(self):
        panel = scalar_panel("accuracy", aggregation="max")

        assert panel["description"] == "The max value of accuracy metric"
        assert panel["isCustomDescription"] is False

    def test_scalar_panel__description__marked_custom(self):
        panel = scalar_panel("accuracy", description="Best accuracy")

        assert panel["description"] == "Best accuracy"
        assert panel["isCustomDescription"] is True

    def test_image_panel__happyflow(self):
        panel = image_panel(["grid.png"], step=10)

        assert panel["chartType"] == "image"
        assert panel["images"] == ["grid.png"]
        assert panel["step"] == 10

    def test_data_panel__happyflow(self):
        panel = data_panel("metrics.csv", axis=1, join_type="inner")

        assert panel["chartType"] == "data"
        assert panel["fileName"] == "metrics.csv"
        assert panel["axis"] == 1
        assert panel["joinType"] == "inner"

    def test_custom_panel__no_name__omits_optional_keys(self):
        panel = custom_panel("inst-1")

        assert panel["instanceId"] == "inst-1"
        assert "chartName" not in panel
        assert "defaultConfig" not in panel

    def test_custom_panel__name_and_config__included(self):
        panel = custom_panel("inst-1", name="Mine", default_config="{}")

        assert panel["chartName"] == "Mine"
        assert panel["defaultConfig"] == "{}"

    def test_python_panel__happyflow(self):
        panel = python_panel("tmpl-1", revision_id="rev-1")

        assert panel["chartType"] == "python"
        assert panel["templateId"] == "tmpl-1"
        assert panel["revisionId"] == "rev-1"
        assert panel["code"] is None

    def test_python_panel__code__stored_in_panel(self):
        panel = python_panel(code="print(1)", name="Mine")

        assert panel["chartType"] == "python"
        assert panel["chartName"] == "Mine"
        assert panel["code"] == "print(1)"
        assert panel["templateId"] == ""
        assert panel["revisionId"] == ""

    def test_python_panel__code__version_follows_code(self):
        first = python_panel(code="print(1)")
        same = python_panel(code="print(1)")
        changed = python_panel(code="print(2)")
        given = python_panel(code="print(1)", code_version="v1")

        assert first["codeVersion"]
        assert first["codeVersion"] == same["codeVersion"]
        assert first["codeVersion"] != changed["codeVersion"]
        assert given["codeVersion"] == "v1"

    def test_python_panel__code_and_template__raises(self):
        with pytest.raises(ValueError):
            python_panel("tmpl-1", code="print(1)")

    def test_python_panel__neither_code_nor_template__raises(self):
        with pytest.raises(ValueError):
            python_panel()

    def test_javascript_panel__defaults__stores_code_in_panel(self):
        panel = javascript_panel("class P extends Comet.Panel {}")

        assert panel["chartType"] == "javascript"
        assert panel["chartName"] == "JavaScript panel"
        assert panel["storeId"] == panel["chartId"]
        assert panel["code"] == {
            "code": "class P extends Comet.Panel {}",
            "html": '<div id="chart-mount"></div>',
            "css": "",
            "resources": [],
        }

    def test_javascript_panel__bare_urls__type_guessed(self):
        panel = javascript_panel(
            "",
            resources=[
                "https://cdn.example.com/lib.js",
                "https://cdn.example.com/style.CSS?v=2",
                "https://fonts.googleapis.com/css2?family=Roboto",
            ],
        )

        assert [r["type"] for r in panel["code"]["resources"]] == [
            "script",
            "stylesheet",
            "script",
        ]

    def test_javascript_panel__typed_resource__kept(self):
        resource = {
            "url": "https://fonts.googleapis.com/css2?family=Roboto",
            "type": "stylesheet",
        }

        panel = javascript_panel("", resources=[resource])

        assert panel["code"]["resources"] == [resource]

    @pytest.mark.parametrize(
        "resource",
        [{"url": "https://x/a.js", "type": "module"}, {"type": "script"}],
    )
    def test_javascript_panel__bad_resource__raises(self, resource):
        with pytest.raises(ValueError):
            javascript_panel("", resources=[resource])

    def test_legend_key__no_label__label_is_value(self):
        assert legend_key("learning_rate", source="params") == {
            "value": "learning_rate",
            "label": "learning_rate",
            "source": "params",
        }


class TestSection:
    @pytest.mark.parametrize("columns", [0, -2, -6, 4, 5, 12, True])
    def test_section__invalid_columns__raises_value_error(self, columns):
        with pytest.raises(ValueError, match="columns"):
            Section("Bad", columns=columns)

    @pytest.mark.parametrize("height", [0, -1, 4, 1.5, True])
    def test_section__invalid_height__raises_value_error(self, height):
        with pytest.raises(ValueError, match="height"):
            Section("Bad", height=height)

    @pytest.mark.parametrize("columns", [1, 2, 3, 6])
    def test_section_layout__columns__panels_flow_left_to_right(self, columns):
        panels = [line_panel("m%d" % i) for i in range(7)]
        section = Section("S", panels, columns=columns, height=2)

        layout = section.layout()

        width = GRID_WIDTH // columns
        for index, (entry, panel) in enumerate(zip(layout, panels)):
            assert entry["i"] == panel["chartId"]
            assert entry["x"] == (index % columns) * width
            assert entry["y"] == (index // columns) * 2
            assert entry["w"] == width
            assert entry["h"] == 2

    def test_section_add__chained__appends_in_order(self):
        first, second = line_panel("a"), line_panel("b")
        section = Section("S")

        assert section.add(first).add(second) is section
        assert section.panels == [first, second]

    def test_section_to_dict__happyflow(self):
        panel = scalar_panel("accuracy")
        section = Section("At a glance", [panel], expanded=False)

        result = section.to_dict()

        assert result["id"] == section.id
        assert result["title"] == "At a glance"
        assert result["expanded"] is False
        assert result["autogenerated"] is False
        assert result["panels"] == [panel]
        assert len(result["layout"]) == 1


class TestStateHelpers:
    def test_global_config__happyflow(self):
        assert global_config(x_axis="epoch", smoothing=0.3)["selectedXAxis"] == "epoch"

    def test_table_state__columns__fixed_columns_first(self):
        state = json.loads(table_state(["Name", "accuracy"], sort_by="accuracy"))

        assert state["columnOrders"] == [
            "pinned",
            "singleExperimentBtn",
            "isVisibleOnDashboard",
            "Name",
            "accuracy",
        ]
        assert state["columnSorting"] == [
            {"columnName": "accuracy", "direction": "desc"}
        ]
        assert [w["columnName"] for w in state["columnWidths"]] == ["Name", "accuracy"]

    def test_table_state__no_sort__empty_sorting(self):
        assert json.loads(table_state())["columnSorting"] == []

    @pytest.mark.parametrize(
        "value, expected",
        [(True, "boolean"), (0.9, "double"), (3, "double"), ("adam", "string")],
    )
    def test_rule__no_type__inferred_from_value(self, value, expected):
        assert rule("field", "equal", value)["type"] == expected

    def test_rule__explicit_type__kept(self):
        assert rule("field", "equal", "1", type_="double")["type"] == "double"

    def test_any_of__no_groups__one_empty_and_group(self):
        assert any_of() == {"condition": "OR", "rules": [all_of()]}

    def test_query_state__rules_tree__serialized(self):
        tree = any_of(all_of(rule("accuracy", "greater", 0.9)))

        state = json.loads(query_state(tree, segment_id="seg-1"))

        assert state["rulesTree"] == tree
        assert state["activeSegment"] == {"segment_id": "seg-1"}


class TestBuildView:
    def test_build_view__sections__in_v3(self):
        sections = [
            Section("A", [scalar_panel("acc")]),
            Section("B", [line_panel("loss")]),
        ]

        view = build_view("Dash", sections, auto_generate=True, personal=True)

        assert isinstance(view, View)
        assert view.name == "Dash"
        assert [s["title"] for s in view.v3["sections"]] == ["A", "B"]
        assert view.v3["autoGenerate"] is True
        assert view.v3["isPersonalView"] is True
        assert view.v3["globalConfig"] == global_config()

    def test_build_view__panels__mirrored_in_legacy_chart_state(self):
        scalar, line = scalar_panel("acc"), line_panel("loss")

        view = build_view("Dash", [Section("A", [scalar, line])])

        charts = json.loads(view.chart_state)["charts"]
        assert charts["scalar"] == [scalar]
        assert charts["line"] == [line]
        assert charts["video"] == []
        assert charts["numberOfCharts"] == 2

    def test_build_view__no_filters__empty_query_state(self):
        view = build_view("Dash", [])

        assert json.loads(view.query_state)["rulesTree"] == any_of()
        assert view.table_state is None

    def test_build_view__filters_and_table__passed_through(self):
        filters = query_state(any_of(all_of(rule("acc", "greater", 0.9))))
        table = table_state(["Name"])

        view = build_view("Dash", [], filters=filters, table=table)

        assert view.query_state == filters
        assert view.table_state == table


class TestBuildExperimentView:
    def test_build_experiment_view__sections__in_v2_panels(self):
        view = build_experiment_view(
            "Run", [Section("Curves", [line_panel("loss")])], x_axis="epoch"
        )

        assert view.v3 is None
        assert view.v2["template_name"] == "Run"
        assert view.v2["x_axis"] == "epoch"
        assert view.v2["panels"]["sections"][0]["title"] == "Curves"
        assert view.v2["panels"]["compareXAxis"] == "epoch"

    def test_build_experiment_view__happyflow__experiment_key_omitted(self):
        view = build_experiment_view("Run", [])

        assert view.experiment_key is None
        assert "experimentKey" not in view.v2


class TestDescribeView:
    def test_describe_view__project_view__lists_sections_panels_filters(self):
        view = build_view(
            "Dash",
            [Section("Curves", [line_panel(["loss", "val_loss"], name="Loss")])],
            filters=query_state(any_of(all_of(rule("acc", "greater", 0.9)))),
        )

        text = describe_view(view)

        assert text.splitlines()[0] == "Dash"
        assert "- section 'Curves'" in text
        assert "line      Loss  ['loss', 'val_loss']" in text
        assert "filter: acc greater 0.9" in text

    def test_describe_view__experiment_view__reads_v2_sections(self):
        view = build_experiment_view("Run", [Section("Curves", [data_panel("t.csv")])])

        text = describe_view(view)

        assert "- section 'Curves'" in text
        assert "t.csv" in text

    def test_describe_view__legacy_chart_state__lists_panels(self):
        chart_state = json.dumps({"charts": {"bar": [bar_panel("accuracy")]}})
        view = View(name="Old", chart_state=chart_state)

        text = describe_view(view)

        assert "(legacy chart_state, no sections)" in text
        assert "accuracy" in text

    def test_describe_view__empty__no_panels(self):
        assert "(no panels)" in describe_view(View(name="Empty"))

    def test_describe_view__metadata__shown(self):
        view = View(
            name="Dash",
            template_id="tmpl-1",
            experiment_key="abcdef123456",
            view_source="chart_template",
            project_default=True,
        )

        text = describe_view(view)

        assert (
            "[chart_template, id=tmpl-1, experiment=abcdef12, project default]" in text
        )
