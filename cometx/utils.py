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
import sys
import time
from datetime import datetime, timedelta

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
    from datetime import datetime

    from comet_ml.query import Environment, Metadata, Metric, Other, Parameter, Tag

    env = {
        "Environment": Environment,
        "Metadata": Metadata,
        "Metric": Metric,
        "Other": Other,
        "Parameter": Parameter,
        "Tag": Tag,
        "datetime": datetime,
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


def format_time_key(dt, unit):
    """
    Format a datetime object as a time key based on the specified unit.

    Args:
        dt: datetime object
        unit: One of "month", "week", "day", "hour"

    Returns:
        str: Formatted time key
    """
    if unit == "month":
        return dt.strftime("%Y-%m")
    elif unit == "week":
        # ISO week format: YYYY-WW
        year, week, _ = dt.isocalendar()
        return f"{year}-W{week:02d}"
    elif unit == "day":
        return dt.strftime("%Y-%m-%d")
    elif unit == "hour":
        return dt.strftime("%Y-%m-%d-%H")
    else:
        raise ValueError(f"Unknown unit: {unit}")


def parse_time_key(time_key, unit):
    """
    Parse a time key string back to a datetime object.

    Args:
        time_key: Time key string (e.g., "2024-01", "2024-W01", "2024-01-01", "2024-01-01-12")
        unit: One of "month", "week", "day", "hour"

    Returns:
        datetime: Parsed datetime object
    """
    if unit == "month":
        return datetime.strptime(time_key, "%Y-%m")
    elif unit == "week":
        # Parse ISO week format: YYYY-WW
        year_str, week_str = time_key.split("-W")
        year = int(year_str)
        week = int(week_str)
        # Create datetime for January 4th of the year (which is always in week 1)
        jan4 = datetime(year, 1, 4)
        # Get the Monday of week 1
        days_since_monday = jan4.weekday()
        week1_monday = jan4 - timedelta(days=days_since_monday)
        # Add weeks to get to the target week
        target_monday = week1_monday + timedelta(weeks=(week - 1))
        return target_monday
    elif unit == "day":
        return datetime.strptime(time_key, "%Y-%m-%d")
    elif unit == "hour":
        return datetime.strptime(time_key, "%Y-%m-%d-%H")
    else:
        raise ValueError(f"Unknown unit: {unit}")


def get_next_time_key(time_key, unit):
    """
    Get the next time key after the given one.

    Args:
        time_key: Current time key string
        unit: One of "month", "week", "day", "hour"

    Returns:
        str: Next time key
    """
    dt = parse_time_key(time_key, unit)
    if unit == "month":
        if dt.month == 12:
            next_dt = dt.replace(year=dt.year + 1, month=1)
        else:
            next_dt = dt.replace(month=dt.month + 1)
    elif unit == "week":
        next_dt = dt + timedelta(weeks=1)
    elif unit == "day":
        next_dt = dt + timedelta(days=1)
    elif unit == "hour":
        next_dt = dt + timedelta(hours=1)
    else:
        raise ValueError(f"Unknown unit: {unit}")
    return format_time_key(next_dt, unit)


def get_unit_label(unit):
    """
    Get a human-readable label for a time unit.

    Args:
        unit: One of "month", "week", "day", "hour"

    Returns:
        str: Label (e.g., "Month", "Week", "Day", "Hour")
    """
    labels = {
        "month": "Month",
        "week": "Week",
        "day": "Day",
        "hour": "Hour",
    }
    return labels.get(unit, unit.capitalize())


def get_unit_label_plural(unit):
    """
    Get a human-readable plural label for a time unit.

    Args:
        unit: One of "month", "week", "day", "hour"

    Returns:
        str: Plural label (e.g., "Months", "Weeks", "Days", "Hours")
    """
    labels = {
        "month": "Months",
        "week": "Weeks",
        "day": "Days",
        "hour": "Hours",
    }
    return labels.get(unit, unit.capitalize() + "s")


def remove_extra_slashes(path):
    if path:
        if path.startswith("/"):
            path = path[1:]
        if path.endswith("/"):
            path = path[:-1]
        return path
    else:
        return ""
