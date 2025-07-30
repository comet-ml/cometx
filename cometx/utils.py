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

import base64
import os
import pathlib
import sys
import time

import six
from comet_ml.config import get_config
from comet_ml.utils import clean_string, get_root_url


class ProgressBar:
    """
    A simple ASCII progress bar, showing a box for each item.
    Uses no control characters.
    """

    def __init__(self, sequence, description=None):
        """
        The sequence to iterate over. For best results,
        don't print during the iteration.
        """
        self.sequence = sequence
        if description:
            self.description = "%s " % description
        else:
            self.description = None

    def set_description(self, description):
        self.description = "%s " % description

    def __iter__(self):
        if self.description:
            print(self.description, end="")
        print("[", end="")
        sys.stdout.flush()
        for item in self.sequence:
            print("█", end="")
            sys.stdout.flush()
            yield item
        print("]")


def _input_user(prompt):
    # type: (str) -> str
    """Independent function to apply clean_string to all responses + make mocking easier"""
    return clean_string(six.moves.input(prompt))


def _input_user_yn(prompt):
    # type: (str) -> bool
    while True:
        response = _input_user(prompt).lower()
        if response.startswith("y") or response.startswith("n"):
            break
    return response.startswith("y")


def get_file_extension(file_path):
    if file_path is None:
        return ""

    ext = os.path.splitext(file_path)[1]
    if not ext:
        return ""

    # Get rid of the leading "."
    if "." in ext:
        return ext[1::]
    else:
        return ext


def display_invalid_api_key(api_key=None, cloud_url=None):
    print(
        "Invalid Comet API Key %r for %r"
        % (
            api_key or get_config("comet.api_key"),
            cloud_url
            or get_root_url(
                get_config("comet.url_override"),
            ),
        )
    )


def get_query_experiments(api, query_string, workspace, project_name):
    from comet_ml.query import Environment, Metadata, Metric, Other, Parameter, Tag

    env = {
        "Environment": Environment,
        "Metadata": Metadata,
        "Metric": Metric,
        "Other": Other,
        "Parameter": Parameter,
        "Tag": Tag,
    }
    query = eval(query_string, env)
    return api.query(workspace, project_name, query)


def download_url(
    url, output_filename, width=None, height=None, timeout=5, headless=False
):
    """
    Args:
        url: (str) the URL to download
        output_filename: (str) should end in ".pdf" or ".html"
        width: (int or float) default None; if output_filename is a pdf, then
            units are in inches. Otherwise ignored
        height: (int or float) default None; if output_filename is a pdf, then
            units are in inches. Otherwise ignored
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.common.print_page_options import PrintOptions
    except Exception:
        print("Downloading urls requires selenium; pip install selenium")
        return

    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless")

    driver = webdriver.Chrome(options=options)
    driver.get(url)
    time.sleep(timeout)

    try:
        button = driver.find_element(
            by="xpath", value='//*[@id="onetrust-reject-all-handler"]'
        )
    except Exception:
        button = None

    if button:
        button.click()
        time.sleep(2)

    if output_filename.endswith(".html"):
        page_source = driver.page_source
        with open(output_filename, "w", encoding="utf-8") as fp:
            fp.write(page_source)

    elif output_filename.endswith(".pdf"):
        print_options = PrintOptions()
        # paper size should be in centimeters
        if width is not None:
            print_options.page_width = width * 2.54
        if height is not None:
            print_options.page_height = height * 2.54
        pdf = driver.print_page(print_options=print_options)
        pdf_bytes = base64.b64decode(pdf)
        with open(output_filename, "wb") as fp:
            fp.write(pdf_bytes)

    elif output_filename.endswith(".png"):
        driver.save_screenshot(output_filename)

    else:
        raise Exception("unknown output_filename type: should end with html or pdf")

    driver.quit()


def remove_extra_slashes(path):
    if path:
        # Handle both forward and backward slashes for cross-platform
        # compatibility
        if path.startswith("/") or path.startswith("\\"):
            path = path[1:]
        if path.endswith("/") or path.endswith("\\"):
            path = path[:-1]
        # Normalize path separators to forward slashes for consistency
        path = path.replace("\\", "/")
        return path
    else:
        return ""


def get_path_parts(path):
    """
    Get path parts using pathlib.Path.parts for cross-platform compatibility.

    Args:
        path: Path string or Path object

    Returns:
        List of path components, excluding empty parts
    """
    if path is None:
        return []
    return [part for part in pathlib.Path(path).parts if part]
