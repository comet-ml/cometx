import os
from types import SimpleNamespace
from unittest import mock

import pytest

from cometx.panel_engine import process_magic_commands
from cometx.panel_studio import (
    STUDIO_SECTION,
    clientlib_url,
    default_selection,
    display_path,
    experiment_colors,
    lighten_color,
    list_directory,
    panel_config,
    project_url,
    put_panel_in_layout,
    sort_entries,
    upload_to_dashboard,
    upload_to_library,
    write_panel_page,
)
from cometx.views import Section, build_view, python_panel, scalar_panel


@pytest.mark.parametrize(
    "typed",
    [
        "https://www.comet.com",
        "https://www.comet.com/",
        "https://www.comet.com/clientlib",
        "https://www.comet.com/clientlib/",
        "https://www.comet.com//clientlib//",
        "  https://www.comet.com  ",
    ],
)
def test_clientlib_url__however_typed__one_clientlib(typed):
    assert clientlib_url(typed) == "https://www.comet.com/clientlib/"


def test_project_url__clientlib_typed__project_page():
    url = project_url("https://comet.example.com/clientlib/", "ws", "proj")

    assert url == "https://comet.example.com/ws/proj"


def test_lighten_color__frontend_formula__matches():
    # The frontend's lightenColor("#fb9341", 0.8)
    assert lighten_color("#fb9341") == "#fee9d9"


def test_experiment_colors__last_hex_digit__picks_chart_color():
    colors = experiment_colors(["abc0", "abc1", "abcc"])

    assert colors["abc0"]["primary"] == "#fb9341"
    assert colors["abc1"]["primary"] == "#5899da"
    assert colors["abcc"]["primary"] == "#fb9341"  # 12 wraps around
    assert colors["abc1"]["light"] == lighten_color("#5899da")


def test_panel_config__happyflow__comet_keys():
    cfg = panel_config("URL", "KEY", "ws", "proj", ["k1"], {"a": 1}, 700, 500, "id")

    assert cfg["comet.api_key"] == "KEY"
    assert cfg["comet.workspace"] == "ws"
    assert cfg["COMET_EXPERIMENT_KEYS"] == ["k1"]
    assert cfg["COMET_PANEL_OPTIONS"] == {"a": 1}
    assert set(cfg["COMET_PANEL_EXPERIMENT_COLORS"]) == {"k1"}


def test_write_panel_page__happyflow__page_holds_config_and_code(tmp_path):
    pages = tmp_path / "pages"
    pages.mkdir()
    cfg = panel_config("URL", "KEY", "ws", "proj", [], {}, None, None, "id")

    path = write_panel_page("ui.display('hi')\n", str(pages), "page_abc", cfg)

    source = open(path).read()
    assert os.path.basename(path) == "page_abc.py"
    assert 'st.session_state["comet_config_override"][key] = config' in source
    assert "\"comet.workspace\": 'ws'" in source
    assert "    ui.display('hi')" in source
    compile(source, path, "exec")


def test_write_panel_page__name_without_page_prefix__raises(tmp_path):
    with pytest.raises(ValueError):
        write_panel_page("x = 1\n", str(tmp_path), "panel", {})


def test_process_magic_commands__pip_install__moved_to_install_code(tmp_path):
    user_code, pip_code = process_magic_commands(
        "%pip install pandas\nimport pandas\n",
        str(tmp_path / "cache"),
        str(tmp_path / "pip.lock"),
    )

    assert "%pip" not in user_code
    assert "import pandas" in user_code
    assert "process_args('''pandas''')" in pip_code
    compile("import streamlit as st\n" + pip_code, "pip", "exec")


def _v3(*sections):
    return build_view("D", sections=list(sections)).v3


def test_put_panel_in_layout__new_panel__added_in_studio_section():
    v3 = _v3(Section("Existing", [scalar_panel("loss")]))

    updated = put_panel_in_layout(v3, "Mine", "print(1)")

    titles = [s["title"] for s in updated["sections"]]
    assert titles == ["Existing", STUDIO_SECTION]
    studio = updated["sections"][1]
    assert studio["panels"][0]["code"] == "print(1)"
    assert studio["layout"][0]["i"] == studio["panels"][0]["chartId"]
    assert v3["sections"][0] == updated["sections"][0]
    assert len(v3["sections"]) == 1  # the input is not modified


def test_put_panel_in_layout__second_panel__placed_below_first():
    v3 = put_panel_in_layout(_v3(), "One", "a = 1")

    updated = put_panel_in_layout(v3, "Two", "b = 2")

    layout = updated["sections"][-1]["layout"]
    assert [item["y"] for item in layout] == [0, 2]


def test_put_panel_in_layout__same_name__code_replaced_in_place():
    v3 = _v3(Section("Mine", [python_panel(code="old", name="P")]))
    chart_id = v3["sections"][0]["panels"][0]["chartId"]

    updated = put_panel_in_layout(v3, "P", "new")

    panels = [p for s in updated["sections"] for p in s["panels"]]
    assert len(panels) == 1
    assert panels[0]["chartId"] == chart_id
    assert panels[0]["code"] == "new"


def test_upload_to_dashboard__no_such_dashboard__created():
    api = mock.MagicMock()
    api.get_views.return_value = []

    _, created = upload_to_dashboard(api, "ws", "proj", "D", "P", "code")

    assert created is True
    view = api.create_view.call_args[0][2]
    assert view.v3["sections"][0]["panels"][0]["code"] == "code"


def test_upload_to_dashboard__existing__updated_with_its_template_id():
    existing = build_view("D", sections=[Section("S", [scalar_panel("loss")])])
    existing.template_id = "tmpl"
    api = mock.MagicMock()
    api.get_views.return_value = [existing]

    _, created = upload_to_dashboard(api, "ws", "proj", "D", "P", "code")

    assert created is False
    view = api.update_view.call_args[0][2]
    assert view.template_id == "tmpl"
    assert len(view.v3["sections"]) == 2


def test_upload_to_dashboard__old_format__raises_value_error():
    api = mock.MagicMock()
    api.get_views.return_value = [SimpleNamespace(name="D", v3=None)]

    with pytest.raises(ValueError, match="older format"):
        upload_to_dashboard(api, "ws", "proj", "D", "P", "code")


def test_upload_to_library__existing_name__overwritten():
    api = mock.MagicMock()
    api.get_panels.return_value = [{"templateName": "P", "templateId": "t1"}]
    api.upload_panel_code.return_value = {"templateId": "t1"}

    template_id, created = upload_to_library(api, "ws", "P", "code")

    assert (template_id, created) == ("t1", False)
    api.upload_panel_code.assert_called_once_with("ws", "P", "code", template_id="t1")


def test_list_directory__mixed_entries__folders_and_py_files(tmp_path):
    (tmp_path / "b.py").write_text("")
    (tmp_path / "A.py").write_text("")
    (tmp_path / "notes.txt").write_text("")
    (tmp_path / ".hidden.py").write_text("")
    (tmp_path / "panels").mkdir()
    (tmp_path / ".git").mkdir()

    folders, files = list_directory(str(tmp_path))

    assert folders == ["panels"]
    assert files == ["A.py", "b.py"]


def test_list_directory__missing_folder__raises_os_error(tmp_path):
    with pytest.raises(OSError):
        list_directory(str(tmp_path / "missing"))


def test_display_path__inside_and_outside_root():
    assert display_path("/w/panels/p.py", "/w") == os.path.join("panels", "p.py")
    assert display_path("/other/p.py", "/w") == "/other/p.py"
    assert display_path("", "/w") == ""


EXPERIMENTS = [
    ("k1", "beta", 300),
    ("k2", "Alpha", 100),
    ("k3", "gamma", 0),
    ("k4", "delta", 200),
]


def test_sort_entries__by_name__case_insensitive():
    names = [name for _, name, _ in sort_entries(EXPERIMENTS, "name")]

    assert names == ["Alpha", "beta", "delta", "gamma"]


def test_sort_entries__by_updated__newest_first():
    keys = [key for key, _, _ in sort_entries(EXPERIMENTS, "updated")]

    assert keys == ["k1", "k4", "k2", "k3"]


def test_sort_entries__unknown_order__raises():
    with pytest.raises(ValueError):
        sort_entries(EXPERIMENTS, "size")


def test_default_selection__more_than_a_page__first_25():
    keys = ["k%d" % i for i in range(60)]

    assert default_selection(keys) == keys[:25]
    assert default_selection(keys[:3]) == keys[:3]
