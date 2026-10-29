import argparse
import os
from types import SimpleNamespace
from unittest import mock

import pytest

from cometx import panel_studio
from cometx.cli import panel_download, panel_preview, panel_upload
from cometx.cli.panel_studio import apply_credentials, port_number
from cometx.panel_studio import (
    choose_experiments,
    file_signature,
    library_panel_code,
    page_name_for,
    read_settled,
    resolve_workspace,
    view_panel_code,
)
from cometx.views import Section, build_view, python_panel, scalar_panel

ENTRIES = [("abc1", "one", 3), ("abd2", "two", 2), ("xyz3", "one", 1)]


def test_page_name_for__file_name__readable_page_name():
    assert page_name_for("/x/loss-curves.py") == "page_loss_curves"
    assert page_name_for("---.py") == "page_panel"


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("all", ["abc1", "abd2", "xyz3"]),
        ("2", ["abc1", "abd2"]),
        ("one", ["abc1", "xyz3"]),  # a name shared by two experiments
        ("abd", ["abd2"]),  # a unique key prefix
        ("two, xyz3", ["abd2", "xyz3"]),
    ],
)
def test_choose_experiments__specs__matching_keys(spec, expected):
    assert choose_experiments(ENTRIES, spec) == expected


def test_choose_experiments__ambiguous_prefix__raises():
    with pytest.raises(ValueError, match="give more of the key"):
        choose_experiments(ENTRIES, "ab")


def test_choose_experiments__no_match__raises():
    with pytest.raises(ValueError, match="No experiment matches"):
        choose_experiments(ENTRIES, "nope")


def test_read_settled__steady_file__text_and_signature(tmp_path):
    path = tmp_path / "p.py"
    path.write_text("x = 1\n")

    signature, text = read_settled(str(path), settle=0)

    assert text == "x = 1\n"
    assert signature == file_signature(str(path))


def test_read_settled__written_while_reading__none(tmp_path):
    path = tmp_path / "p.py"
    path.write_text("")

    def finish_writing(_):
        path.write_text("x = 1\n")

    with mock.patch.object(panel_studio.time, "sleep", finish_writing):
        assert read_settled(str(path)) is None


def test_read_settled__missing_file__none(tmp_path):
    assert read_settled(str(tmp_path / "missing.py"), settle=0) is None


def test_resolve_workspace__order__given_then_config_then_default():
    api = mock.MagicMock()
    api.get_default_workspace.return_value = "default-ws"
    with mock.patch.object(panel_studio, "get_config", return_value="config-ws"):
        assert resolve_workspace(api, "given") == "given"
        assert resolve_workspace(api) == "config-ws"
    with mock.patch.object(panel_studio, "get_config", return_value=None):
        assert resolve_workspace(api) == "default-ws"


def _api_with_view(*panels):
    view = build_view("D", sections=[Section("S", list(panels))])
    api = mock.MagicMock()
    api.get_views.return_value = [view]
    return api


def test_view_panel_code__single_python_panel__name_optional():
    api = _api_with_view(scalar_panel("loss"), python_panel(code="a = 1", name="P"))

    assert view_panel_code(api, "ws", "proj", "D") == ("P", "a = 1")


def test_view_panel_code__several_panels__needs_name():
    api = _api_with_view(
        python_panel(code="a = 1", name="P"), python_panel(code="b = 2", name="Q")
    )

    with pytest.raises(ValueError, match="pick one with --name: P, Q"):
        view_panel_code(api, "ws", "proj", "D")
    assert view_panel_code(api, "ws", "proj", "D", "Q") == ("Q", "b = 2")


def test_view_panel_code__panel_added_in_ui__explains():
    api = _api_with_view(python_panel("tmpl", revision_id="rev", name="P"))

    with pytest.raises(ValueError, match="added in the Comet UI"):
        view_panel_code(api, "ws", "proj", "D")


def test_view_panel_code__no_such_view__lists_views():
    api = _api_with_view(python_panel(code="a = 1", name="P"))

    with pytest.raises(ValueError, match="Views: D"):
        view_panel_code(api, "ws", "proj", "Other")


def test_library_panel_code__by_name__code():
    api = mock.MagicMock()
    api.get_panels.return_value = [{"templateName": "P", "templateId": "t1"}]
    api.get_panel_code.return_value = "a = 1"

    assert library_panel_code(api, "ws", "P") == "a = 1"
    api.get_panel_code.assert_called_once_with("t1")
    with pytest.raises(ValueError, match="No panel"):
        library_panel_code(api, "ws", "missing")


def _parse(module, args):
    parser = argparse.ArgumentParser()
    module.get_parser_arguments(parser)
    return parser.parse_args(args)


def test_panel_upload__view__uploads_file_named_after_it(tmp_path, capsys):
    path = tmp_path / "loss_curves.py"
    path.write_text("a = 1\n")
    api = mock.MagicMock(server_url="https://comet.example.com")
    view = SimpleNamespace(template_id="v1")
    args = _parse(
        panel_upload,
        [str(path), "--workspace", "ws", "--project", "proj", "--view", "D"],
    )

    with mock.patch.object(panel_upload, "API", return_value=api), mock.patch.object(
        panel_upload, "upload_to_view", return_value=(view, True)
    ) as upload:
        panel_upload.panel_upload(args)

    upload.assert_called_once_with(api, "ws", "proj", "D", "loss_curves", "a = 1\n")
    assert "https://comet.example.com/ws/proj/view/v1/panels" in capsys.readouterr().out


def test_panel_upload__view_without_project__exits(tmp_path):
    path = tmp_path / "p.py"
    path.write_text("")
    args = _parse(panel_upload, [str(path), "--view", "D"])

    with pytest.raises(SystemExit, match="--project is required"):
        panel_upload.panel_upload(args)


def test_panel_download__library_to_file__written(tmp_path):
    out = tmp_path / "p.py"
    args = _parse(
        panel_download,
        ["--library", "--workspace", "ws", "--name", "P", "-o", str(out)],
    )

    with mock.patch.object(panel_download, "API"), mock.patch.object(
        panel_download, "library_panel_code", return_value="a = 1\n"
    ):
        panel_download.panel_download(args)

    assert out.read_text() == "a = 1\n"


def test_panel_download__existing_output__refused_without_force(tmp_path):
    out = tmp_path / "p.py"
    out.write_text("keep me")
    args = _parse(panel_download, ["--library", "--name", "P", "-o", str(out)])

    with pytest.raises(SystemExit, match="--force"):
        panel_download.panel_download(args)
    assert out.read_text() == "keep me"


def test_file_signature__same_time_and_size__contents_still_differ(tmp_path):
    path = tmp_path / "p.py"
    path.write_text("x = 1\n")
    stat = os.stat(path)
    before = file_signature(str(path))

    path.write_text("x = 2\n")  # same size
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))  # same time

    assert file_signature(str(path)) != before


@pytest.mark.parametrize("port", ["0", "65536", "-1", "http"])
def test_port_number__out_of_range__parse_error(port):
    parser = argparse.ArgumentParser()
    panel_preview.get_parser_arguments(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["p.py", "--project", "proj", "--port", port])


def test_port_number__in_range__int():
    assert port_number("8502") == 8502


def test_credentials__given_before_command__kept():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key")
    sub = parser.add_subparsers().add_parser("panel-preview")
    panel_preview.get_parser_arguments(sub)

    args = parser.parse_args(
        ["--api-key", "KEY", "panel-preview", "p.py", "--project", "proj"]
    )

    assert args.api_key == "KEY"


def test_credentials__given_after_command__used(monkeypatch):
    monkeypatch.delenv("COMET_API_KEY", raising=False)
    args = _parse(panel_preview, ["p.py", "--project", "proj", "--api-key", "KEY"])

    apply_credentials(args)

    assert os.environ["COMET_API_KEY"] == "KEY"


def test_read_first__folder_or_missing__none(tmp_path):
    assert panel_preview.read_first(str(tmp_path)) is None
    assert panel_preview.read_first(str(tmp_path / "missing.py")) is None


@pytest.mark.parametrize("module", [panel_upload, panel_download])
def test_panel_commands__older_dashboard_option__same_as_view(module):
    head = ["p.py"] if module is panel_upload else []
    parser = argparse.ArgumentParser()
    module.get_parser_arguments(parser)

    assert parser.parse_args(head + ["--dashboard", "D"]).view == "D"
    assert "--dashboard" not in parser.format_help()


def test_dashboard_panel_code__older_name__same_as_view_panel_code():
    api = _api_with_view(python_panel(code="a = 1", name="P"))

    assert panel_studio.dashboard_panel_code(api, "ws", "proj", dashboard="D") == (
        "P",
        "a = 1",
    )
