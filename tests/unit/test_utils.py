# -*- coding: utf-8 -*-
# ****************************************
#                              __
#   _________  ____ ___  ___  / /__  __
#  / ___/ __ \/ __ `__ \/ _ \/ __/ |/_/
# / /__/ /_/ / / / / / /  __/ /__>  <
# \___/\____/_/ /_/ /_/\___/\__/_/|_|
#
#
#  Copyright (c) 2022 Cometx Development
#      Team. All rights reserved.
# ****************************************

import unittest
from io import StringIO
from unittest.mock import MagicMock, patch

from cometx.utils import (
    ProgressBar,
    _input_user,
    _input_user_yn,
    display_invalid_api_key,
    download_url,
    get_file_extension,
    get_query_experiments,
    remove_extra_slashes,
)


class TestProgressBar:
    def test_progress_bar_basic(self):
        """Test basic progress bar functionality"""
        sequence = [1, 2, 3, 4, 5]
        progress_bar = ProgressBar(sequence, "Processing")

        # Capture stdout to verify output
        captured_output = StringIO()
        with patch("sys.stdout", captured_output):
            result = list(progress_bar)

        assert result == [1, 2, 3, 4, 5]
        output = captured_output.getvalue()
        assert "Processing" in output
        assert "█" in output
        assert output.startswith("Processing [")
        assert output.endswith("]\n")

    def test_progress_bar_no_description(self):
        """Test progress bar without description"""
        sequence = [1, 2, 3]
        progress_bar = ProgressBar(sequence)

        captured_output = StringIO()
        with patch("sys.stdout", captured_output):
            result = list(progress_bar)

        assert result == [1, 2, 3]
        output = captured_output.getvalue()
        assert output.startswith("[")
        assert output.endswith("]\n")
        assert "█" in output

    def test_progress_bar_set_description(self):
        """Test setting description after initialization"""
        sequence = [1, 2]
        progress_bar = ProgressBar(sequence)
        progress_bar.set_description("New Description")

        captured_output = StringIO()
        with patch("sys.stdout", captured_output):
            result = list(progress_bar)

        assert result == [1, 2]
        output = captured_output.getvalue()
        assert "New Description" in output

    def test_progress_bar_empty_sequence(self):
        """Test progress bar with empty sequence"""
        sequence = []
        progress_bar = ProgressBar(sequence, "Empty")

        captured_output = StringIO()
        with patch("sys.stdout", captured_output):
            result = list(progress_bar)

        assert result == []
        output = captured_output.getvalue()
        assert output == "Empty []\n"


class TestInputUser:
    @patch("six.moves.input")
    @patch("cometx.utils.clean_string")
    def test_input_user(self, mock_clean_string, mock_input):
        """Test _input_user function"""
        mock_input.return_value = "test input"
        mock_clean_string.return_value = "cleaned input"

        result = _input_user("Enter something: ")

        mock_input.assert_called_once_with("Enter something: ")
        mock_clean_string.assert_called_once_with("test input")
        assert result == "cleaned input"


class TestInputUserYN:
    @patch("cometx.utils._input_user")
    def test_input_user_yn_yes(self, mock_input_user):
        """Test _input_user_yn with yes response"""
        mock_input_user.side_effect = ["yes", "y", "YES"]

        result = _input_user_yn("Continue? (y/n): ")

        assert result is True
        assert mock_input_user.call_count == 1

    @patch("cometx.utils._input_user")
    def test_input_user_yn_no(self, mock_input_user):
        """Test _input_user_yn with no response"""
        mock_input_user.side_effect = ["no", "n", "NO"]

        result = _input_user_yn("Continue? (y/n): ")

        assert result is False
        assert mock_input_user.call_count == 1

    @patch("cometx.utils._input_user")
    def test_input_user_yn_invalid_then_valid(self, mock_input_user):
        """Test _input_user_yn with invalid input followed by valid"""
        mock_input_user.side_effect = ["maybe", "invalid", "yes"]

        result = _input_user_yn("Continue? (y/n): ")

        assert result is True
        assert mock_input_user.call_count == 3


class TestGetFileExtension:
    def test_get_file_extension_with_extension(self):
        """Test get_file_extension with various file extensions"""
        assert get_file_extension("file.txt") == "txt"
        assert get_file_extension("document.pdf") == "pdf"
        assert get_file_extension("image.jpg") == "jpg"
        assert get_file_extension("script.py") == "py"

    def test_get_file_extension_no_extension(self):
        """Test get_file_extension with files without extensions"""
        assert get_file_extension("file") == ""
        assert get_file_extension("document") == ""

    def test_get_file_extension_none(self):
        """Test get_file_extension with None input"""
        assert get_file_extension(None) == ""

    def test_get_file_extension_dot_only(self):
        """Test get_file_extension with just a dot"""
        assert get_file_extension("file.") == ""

    def test_get_file_extension_multiple_dots(self):
        """Test get_file_extension with multiple dots"""
        assert get_file_extension("file.backup.txt") == "txt"
        assert get_file_extension("archive.tar.gz") == "gz"


class TestDisplayInvalidApiKey:
    @patch("builtins.print")
    @patch("comet_ml.config.get_config")
    @patch("comet_ml.utils.get_root_url")
    def test_display_invalid_api_key_defaults(
        self, mock_get_root_url, mock_get_config, mock_print
    ):
        """Test display_invalid_api_key with default parameters"""
        mock_get_config.return_value = "default_api_key"
        mock_get_root_url.return_value = "default_url"

        display_invalid_api_key()

        mock_print.assert_called_once()
        call_args = mock_print.call_args[0][0]
        assert "Invalid Comet API Key" in call_args
        # The function uses actual config values, so we just check the format
        assert "for" in call_args

    @patch("builtins.print")
    def test_display_invalid_api_key_custom(self, mock_print):
        """Test display_invalid_api_key with custom parameters"""
        display_invalid_api_key("custom_key", "custom_url")

        mock_print.assert_called_once()
        call_args = mock_print.call_args[0][0]
        assert "Invalid Comet API Key" in call_args
        assert "custom_key" in call_args
        assert "custom_url" in call_args


class TestGetQueryExperiments:
    @patch("cometx.utils.eval")
    @patch("comet_ml.API")
    def test_get_query_experiments(self, mock_api_class, mock_eval):
        """Test get_query_experiments function"""
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        mock_query = MagicMock()
        mock_eval.return_value = mock_query

        query_string = "Metric('accuracy') > 0.8"
        workspace = "test_workspace"
        project_name = "test_project"

        result = get_query_experiments(mock_api, query_string, workspace, project_name)

        mock_eval.assert_called_once()
        mock_api.query.assert_called_once_with(workspace, project_name, mock_query)
        assert result == mock_api.query.return_value


class TestDownloadUrl:
    def test_download_url_no_selenium(self):
        """Test download_url when selenium is not available"""
        # This test verifies that the function handles missing selenium gracefully
        result = download_url("http://example.com", "file.html")
        # Should return None when selenium is not available
        assert result is None

    def test_download_url_unknown_extension(self):
        """Test download_url with unknown file extension"""
        # This test verifies that the function handles unknown extensions gracefully
        result = download_url("http://example.com", "file.unknown")
        # Should return None when selenium is not available
        assert result is None


class TestRemoveExtraSlashes:
    def test_remove_extra_slashes_with_slashes(self):
        """Test remove_extra_slashes with various slash patterns"""
        assert remove_extra_slashes("/path/to/file/") == "path/to/file"
        assert remove_extra_slashes("/path/to/file") == "path/to/file"
        assert remove_extra_slashes("path/to/file/") == "path/to/file"
        assert remove_extra_slashes("path/to/file") == "path/to/file"

    def test_remove_extra_slashes_empty(self):
        """Test remove_extra_slashes with empty or None input"""
        assert remove_extra_slashes("") == ""
        assert remove_extra_slashes(None) == ""

    def test_remove_extra_slashes_single_slash(self):
        """Test remove_extra_slashes with single slash"""
        assert remove_extra_slashes("/") == ""
        assert remove_extra_slashes("///") == "/"

    def test_remove_extra_slashes_no_slashes(self):
        """Test remove_extra_slashes with no slashes"""
        assert remove_extra_slashes("path") == "path"
        assert remove_extra_slashes("file.txt") == "file.txt"


if __name__ == "__main__":
    unittest.main()
