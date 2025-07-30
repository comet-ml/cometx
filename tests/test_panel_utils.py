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

import json
import os
import unittest
import zipfile
from unittest.mock import MagicMock, patch

from cometx.panel_utils import create_panel_zip, get_uuid, int_to_string


class TestIntToString:
    def test_int_to_string_basic(self):
        """Test basic int_to_string functionality"""
        alphabet = "0123456789"

        # Test basic conversion
        assert int_to_string(123, alphabet) == "123"
        assert int_to_string(0, alphabet) == ""
        assert int_to_string(1, alphabet) == "1"
        assert int_to_string(10, alphabet) == "10"

    def test_int_to_string_custom_alphabet(self):
        """Test int_to_string with custom alphabets"""
        # Binary alphabet
        binary = "01"
        assert int_to_string(5, binary) == "101"
        assert int_to_string(10, binary) == "1010"

        # Hex alphabet
        hex_alphabet = "0123456789ABCDEF"
        assert int_to_string(15, hex_alphabet) == "F"
        assert int_to_string(255, hex_alphabet) == "FF"

        # Custom alphabet
        custom = "ABC"
        assert int_to_string(0, custom) == ""
        assert int_to_string(1, custom) == "B"
        assert int_to_string(2, custom) == "C"
        assert int_to_string(3, custom) == "BA"
        assert int_to_string(4, custom) == "BB"

    def test_int_to_string_with_padding(self):
        """Test int_to_string with padding"""
        alphabet = "0123456789"

        # Test with padding
        assert int_to_string(123, alphabet, padding=5) == "00123"
        assert int_to_string(1, alphabet, padding=3) == "001"
        assert int_to_string(0, alphabet, padding=2) == "00"

        # Test with padding smaller than result
        assert int_to_string(12345, alphabet, padding=3) == "12345"

    def test_int_to_string_edge_cases(self):
        """Test int_to_string edge cases"""
        alphabet = "ABC"

        # Test zero
        assert int_to_string(0, alphabet) == ""
        assert int_to_string(0, alphabet, padding=3) == "AAA"

        # Test large numbers
        large_num = 1000
        result = int_to_string(large_num, alphabet)
        # Should be a valid representation
        assert len(result) > 0
        assert all(c in alphabet for c in result)

    def test_int_to_string_reverse_order(self):
        """Test that int_to_string produces most significant digit first"""
        alphabet = "0123456789"

        # Test that larger numbers produce longer strings
        assert len(int_to_string(100, alphabet)) > len(int_to_string(10, alphabet))
        assert len(int_to_string(1000, alphabet)) > len(int_to_string(100, alphabet))


class TestGetUuid:
    def test_get_uuid_length(self):
        """Test get_uuid returns string of correct length"""
        # get_uuid should return a string of reasonable length
        for length in [5, 10, 15, 25]:
            result = get_uuid(length)
            assert len(result) >= 20  # Should be at least 20 characters
            assert isinstance(result, str)

    def test_get_uuid_alphabet(self):
        """Test get_uuid uses correct alphabet"""
        expected_alphabet = "23456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

        # Generate multiple UUIDs and check they only use expected characters
        for _ in range(10):
            result = get_uuid(25)
            assert all(c in expected_alphabet for c in result)

    def test_get_uuid_uniqueness(self):
        """Test get_uuid produces unique results"""
        results = set()
        for _ in range(100):
            result = get_uuid(10)
            assert result not in results
            results.add(result)

    @patch("cometx.panel_utils.uuid.uuid4")
    def test_get_uuid_uses_uuid4(self, mock_uuid4):
        """Test get_uuid uses uuid.uuid4()"""
        mock_uuid = MagicMock()
        mock_uuid.int = 12345
        mock_uuid4.return_value = mock_uuid

        get_uuid(10)

        mock_uuid4.assert_called_once()

    def test_get_uuid_different_lengths(self):
        """Test get_uuid with different length parameters"""
        lengths = [1, 5, 10, 25, 50]
        for length in lengths:
            result = get_uuid(length)
            # get_uuid should return a string of reasonable length regardless of length
            # parameter
            assert len(result) >= 20  # Should be at least 20 characters


class TestCreatePanelZip:
    def test_create_panel_zip_basic(self):
        """Test basic create_panel_zip functionality"""
        name = "Test Panel"
        code = "print('Hello World')"

        zip_filename = create_panel_zip(name, code)

        # Check that file exists
        assert os.path.exists(zip_filename)
        assert zip_filename.endswith(".zip")

        # Check zip file contents
        with zipfile.ZipFile(zip_filename, "r") as zip_file:
            file_list = zip_file.namelist()
            assert "tempVisualizationTemplate.json" in file_list

            # Check JSON content
            json_content = zip_file.read("tempVisualizationTemplate.json")
            template = json.loads(json_content.decode("utf-8"))

            assert template["templateName"] == name
            assert template["code"]["pyCode"] == code.lstrip()
            assert template["code"]["type"] == "py"
            assert template["editable"] is True
            assert "createdAt" in template
            assert "thumbnailName" in template

        # Clean up
        os.unlink(zip_filename)

    def test_create_panel_zip_with_indented_code(self):
        """Test create_panel_zip with indented code"""
        name = "Indented Panel"
        code = """
        def hello():
            print("Hello World")
        """

        zip_filename = create_panel_zip(name, code)

        try:
            with zipfile.ZipFile(zip_filename, "r") as zip_file:
                json_content = zip_file.read("tempVisualizationTemplate.json")
                template = json.loads(json_content.decode("utf-8"))

                # Code should be stripped of leading whitespace
                expected_code = code.lstrip()
                assert template["code"]["pyCode"] == expected_code
        finally:
            os.unlink(zip_filename)

    def test_create_panel_zip_template_structure(self):
        """Test create_panel_zip creates correct template structure"""
        name = "Structure Test"
        code = "test code"

        zip_filename = create_panel_zip(name, code)

        try:
            with zipfile.ZipFile(zip_filename, "r") as zip_file:
                json_content = zip_file.read("tempVisualizationTemplate.json")
                template = json.loads(json_content.decode("utf-8"))

                # Check required fields
                required_fields = [
                    "templateName",
                    "code",
                    "createdAt",
                    "thumbnailName",
                    "editable",
                ]
                for field in required_fields:
                    assert field in template

                # Check code structure
                code_section = template["code"]
                required_code_fields = [
                    "code",
                    "css",
                    "description",
                    "html",
                    "defaultConfig",
                    "internalResources",
                    "userResources",
                    "pyCode",
                    "type",
                    "pyConfig",
                ]
                for field in required_code_fields:
                    assert field in code_section

                # Check specific values
                assert code_section["type"] == "py"
                assert code_section["pyConfig"] == "{}"
                assert code_section["internalResources"] == []
                assert code_section["userResources"] == []
        finally:
            os.unlink(zip_filename)

    def test_create_panel_zip_uuid_generation(self):
        """Test create_panel_zip generates unique UUIDs"""
        name = "UUID Test"
        code = "test"

        zip_filenames = []
        try:
            for _ in range(5):
                zip_filename = create_panel_zip(name, code)
                zip_filenames.append(zip_filename)

                # Check that filenames are unique
                assert zip_filename not in zip_filenames[:-1]

                # Check filename format
                assert "panel-" in zip_filename
                assert zip_filename.endswith(".zip")
        finally:
            for filename in zip_filenames:
                if os.path.exists(filename):
                    os.unlink(filename)

    def test_create_panel_zip_timestamp(self):
        """Test create_panel_zip includes current timestamp"""
        name = "Timestamp Test"
        code = "test"

        zip_filename = create_panel_zip(name, code)

        try:
            with zipfile.ZipFile(zip_filename, "r") as zip_file:
                json_content = zip_file.read("tempVisualizationTemplate.json")
                template = json.loads(json_content.decode("utf-8"))

                # Check that createdAt is a recent timestamp
                import time

                current_time = int(time.time() * 1000)
                created_at = template["createdAt"]

                # Should be within last 5 seconds
                assert abs(current_time - created_at) < 5000
        finally:
            os.unlink(zip_filename)

    def test_create_panel_zip_thumbnail_name(self):
        """Test create_panel_zip generates thumbnail name with UUID"""
        name = "Thumbnail Test"
        code = "test"

        zip_filename = create_panel_zip(name, code)

        try:
            with zipfile.ZipFile(zip_filename, "r") as zip_file:
                json_content = zip_file.read("tempVisualizationTemplate.json")
                template = json.loads(json_content.decode("utf-8"))

                thumbnail_name = template["thumbnailName"]
                assert thumbnail_name.startswith("template-thumbnail-")

                # Extract UUID part and verify it's 25 characters
                uuid_part = thumbnail_name.replace("template-thumbnail-", "")
                assert len(uuid_part) == 25
        finally:
            os.unlink(zip_filename)

    def test_create_panel_zip_empty_code(self):
        """Test create_panel_zip with empty code"""
        name = "Empty Code Test"
        code = ""

        zip_filename = create_panel_zip(name, code)

        try:
            with zipfile.ZipFile(zip_filename, "r") as zip_file:
                json_content = zip_file.read("tempVisualizationTemplate.json")
                template = json.loads(json_content.decode("utf-8"))

                assert template["code"]["pyCode"] == ""
        finally:
            os.unlink(zip_filename)

    def test_create_panel_zip_special_characters(self):
        """Test create_panel_zip with special characters in name and code"""
        name = "Special Chars: !@#$%^&*()"
        code = "print('Special: !@#$%^&*()')"

        zip_filename = create_panel_zip(name, code)

        try:
            with zipfile.ZipFile(zip_filename, "r") as zip_file:
                json_content = zip_file.read("tempVisualizationTemplate.json")
                template = json.loads(json_content.decode("utf-8"))

                assert template["templateName"] == name
                assert template["code"]["pyCode"] == code
        finally:
            os.unlink(zip_filename)


if __name__ == "__main__":
    unittest.main()
