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

import pathlib

import pytest

from cometx.utils import get_path_parts


class TestGetPathParts:
    """Test cases for the get_path_parts function."""

    def test_none_input(self):
        """Test that None input returns empty list."""
        result = get_path_parts(None)
        assert result == []

    def test_empty_string(self):
        """Test that empty string returns empty list."""
        result = get_path_parts("")
        assert result == []

    def test_single_component(self):
        """Test single path component."""
        result = get_path_parts("workspace")
        assert result == ["workspace"]

    def test_two_components(self):
        """Test two path components."""
        result = get_path_parts("workspace/project")
        assert result == ["workspace", "project"]

    def test_three_components(self):
        """Test three path components."""
        result = get_path_parts("workspace/project/experiment")
        assert result == ["workspace", "project", "experiment"]

    def test_leading_slash(self):
        """Test path with leading slash."""
        result = get_path_parts("/workspace/project")
        assert result == ["workspace", "project"]

    def test_trailing_slash(self):
        """Test path with trailing slash."""
        result = get_path_parts("workspace/project/")
        assert result == ["workspace", "project"]

    def test_both_leading_and_trailing_slash(self):
        """Test path with both leading and trailing slashes."""
        result = get_path_parts("/workspace/project/")
        assert result == ["workspace", "project"]

    def test_multiple_slashes(self):
        """Test path with multiple consecutive slashes."""
        result = get_path_parts("workspace//project///experiment")
        assert result == ["workspace", "project", "experiment"]

    def test_windows_path_separators(self):
        """Test Windows-style path separators."""
        result = get_path_parts("workspace\\project\\experiment")
        assert result == ["workspace", "project", "experiment"]

    def test_mixed_path_separators(self):
        """Test mixed forward and backward slashes."""
        result = get_path_parts("workspace\\project/experiment")
        assert result == ["workspace", "project", "experiment"]

    def test_windows_absolute_path(self):
        """Test Windows absolute path."""
        result = get_path_parts("C:\\workspace\\project\\experiment")
        assert result == ["workspace", "project", "experiment"]

    def test_unix_absolute_path(self):
        """Test Unix absolute path."""
        result = get_path_parts("/home/user/workspace/project")
        assert result == ["home", "user", "workspace", "project"]

    def test_path_with_spaces(self):
        """Test path with spaces in components."""
        result = get_path_parts("workspace name/project name/experiment name")
        assert result == ["workspace name", "project name", "experiment name"]

    def test_path_with_special_characters(self):
        """Test path with special characters."""
        result = get_path_parts("workspace-name/project_name/experiment@123")
        assert result == ["workspace-name", "project_name", "experiment@123"]

    def test_path_with_unicode(self):
        """Test path with unicode characters."""
        result = get_path_parts("wörkspace/pröject/expériment")
        assert result == ["wörkspace", "pröject", "expériment"]

    def test_pathlib_path_object(self):
        """Test that function works with pathlib.Path objects."""
        path_obj = pathlib.Path("workspace/project/experiment")
        result = get_path_parts(path_obj)
        assert result == ["workspace", "project", "experiment"]

    def test_nested_paths(self):
        """Test deeply nested paths."""
        result = get_path_parts("workspace/project/subfolder/experiment/data")
        assert result == ["workspace", "project", "subfolder", "experiment", "data"]

    def test_current_directory(self):
        """Test current directory notation."""
        result = get_path_parts(".")
        assert result == []

    def test_path_with_numbers(self):
        """Test path with numeric components."""
        result = get_path_parts("workspace123/project456/experiment789")
        assert result == ["workspace123", "project456", "experiment789"]

    def test_empty_components(self):
        """Test path with empty components (should be filtered out)."""
        result = get_path_parts("workspace//project///experiment")
        assert result == ["workspace", "project", "experiment"]

    def test_only_slashes(self):
        """Test path with only slashes."""
        result = get_path_parts("///")
        assert result == []

    def test_single_slash(self):
        """Test path with single slash."""
        result = get_path_parts("/")
        assert result == []

    def test_backslash_only(self):
        """Test path with only backslashes."""
        result = get_path_parts("\\\\\\")
        assert result == []

    def test_mixed_slashes_with_empty_components(self):
        """Test mixed slashes with empty components."""
        result = get_path_parts("workspace\\/project//\\experiment")
        assert result == ["workspace", "project", "experiment"]

    def test_root_path(self):
        """Test root path on Unix systems."""
        result = get_path_parts("/")
        assert result == []

    def test_relative_path_with_dots(self):
        """Test relative path with dot notation."""
        result = get_path_parts("./workspace/project")
        assert result == ["workspace", "project"]

    def test_path_with_quotes(self):
        """Test path with quoted components."""
        result = get_path_parts('workspace/"project name"/experiment')
        assert result == ["workspace", '"project name"', "experiment"]

    def test_path_with_escaped_characters(self):
        """Test path with escaped characters."""
        result = get_path_parts("workspace\\project name\\experiment")
        assert result == ["workspace", "project name", "experiment"]

    # Tests for invalid paths containing ".."
    def test_path_with_double_dots_raises_exception(self):
        """Test that paths with '..' raise ValueError."""
        with pytest.raises(ValueError, match="Path contains '..' which is not allowed"):
            get_path_parts("workspace/../project")

    def test_path_with_double_dots_at_start_raises_exception(self):
        """Test that paths starting with '..' raise ValueError."""
        with pytest.raises(ValueError, match="Path contains '..' which is not allowed"):
            get_path_parts("../workspace/project")

    def test_path_with_double_dots_at_end_raises_exception(self):
        """Test that paths ending with '..' raise ValueError."""
        with pytest.raises(ValueError, match="Path contains '..' which is not allowed"):
            get_path_parts("workspace/project/..")

    def test_path_with_multiple_double_dots_raises_exception(self):
        """Test that paths with multiple '..' raise ValueError."""
        with pytest.raises(ValueError, match="Path contains '..' which is not allowed"):
            get_path_parts("workspace/../project/../experiment")

    def test_path_with_double_dots_in_middle_raises_exception(self):
        """Test that paths with '..' in the middle raise ValueError."""
        with pytest.raises(ValueError, match="Path contains '..' which is not allowed"):
            get_path_parts("workspace/project/../experiment")

    def test_path_with_double_dots_windows_style_raises_exception(self):
        """Test that Windows-style paths with '..' raise ValueError."""
        with pytest.raises(ValueError, match="Path contains '..' which is not allowed"):
            get_path_parts("workspace\\..\\project")

    def test_path_with_double_dots_mixed_separators_raises_exception(self):
        """Test that mixed separator paths with '..' raise ValueError."""
        with pytest.raises(ValueError, match="Path contains '..' which is not allowed"):
            get_path_parts("workspace\\../project")

    def test_path_with_double_dots_absolute_path_raises_exception(self):
        """Test that absolute paths with '..' raise ValueError."""
        with pytest.raises(ValueError, match="Path contains '..' which is not allowed"):
            get_path_parts("/workspace/../project")

    def test_path_with_double_dots_windows_absolute_raises_exception(self):
        """Test that Windows absolute paths with '..' raise ValueError."""
        with pytest.raises(ValueError, match="Path contains '..' which is not allowed"):
            get_path_parts("C:\\workspace\\..\\project")

    def test_path_with_double_dots_and_spaces_raises_exception(self):
        """Test that paths with '..' and spaces raise ValueError."""
        with pytest.raises(ValueError, match="Path contains '..' which is not allowed"):
            get_path_parts("workspace name/../project name")

    def test_path_with_double_dots_and_special_chars_raises_exception(self):
        """Test that paths with '..' and special characters raise ValueError."""
        with pytest.raises(ValueError, match="Path contains '..' which is not allowed"):
            get_path_parts("workspace-name/../project_name")

    def test_path_with_double_dots_and_unicode_raises_exception(self):
        """Test that paths with '..' and unicode characters raise ValueError."""
        with pytest.raises(ValueError, match="Path contains '..' which is not allowed"):
            get_path_parts("wörkspace/../pröject")

    def test_path_with_double_dots_and_quotes_raises_exception(self):
        """Test that paths with '..' and quotes raise ValueError."""
        with pytest.raises(ValueError, match="Path contains '..' which is not allowed"):
            get_path_parts('workspace/"../project"')

    def test_path_with_double_dots_and_escaped_chars_raises_exception(self):
        """Test that paths with '..' and escaped characters raise ValueError."""
        with pytest.raises(ValueError, match="Path contains '..' which is not allowed"):
            get_path_parts("workspace\\../project")

    def test_path_with_double_dots_and_numbers_raises_exception(self):
        """Test that paths with '..' and numbers raise ValueError."""
        with pytest.raises(ValueError, match="Path contains '..' which is not allowed"):
            get_path_parts("workspace123/../project456")

    def test_path_with_double_dots_and_empty_components_raises_exception(self):
        """Test that paths with '..' and empty components raise ValueError."""
        with pytest.raises(ValueError, match="Path contains '..' which is not allowed"):
            get_path_parts("workspace//../project")

    def test_path_with_double_dots_and_multiple_slashes_raises_exception(self):
        """Test that paths with '..' and multiple slashes raise ValueError."""
        with pytest.raises(ValueError, match="Path contains '..' which is not allowed"):
            get_path_parts("workspace///../project")

    def test_path_with_double_dots_and_mixed_slashes_raises_exception(self):
        """Test that paths with '..' and mixed slashes raise ValueError."""
        with pytest.raises(ValueError, match="Path contains '..' which is not allowed"):
            get_path_parts("workspace\\/../project")

    def test_path_with_double_dots_and_root_raises_exception(self):
        """Test that root paths with '..' raise ValueError."""
        with pytest.raises(ValueError, match="Path contains '..' which is not allowed"):
            get_path_parts("/../workspace")

    def test_path_with_double_dots_and_current_dir_raises_exception(self):
        """Test that paths with '..' and current directory notation raise ValueError."""
        with pytest.raises(ValueError, match="Path contains '..' which is not allowed"):
            get_path_parts("./../workspace")

    def test_path_with_double_dots_and_pathlib_object_raises_exception(self):
        """Test that pathlib.Path objects with '..' raise ValueError."""
        path_obj = pathlib.Path("workspace/../project")
        with pytest.raises(ValueError, match="Path contains '..' which is not allowed"):
            get_path_parts(path_obj)
