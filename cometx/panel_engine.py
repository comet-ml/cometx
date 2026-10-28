# -*- coding: utf-8 -*-
####################################
# Copyright (c) 2024 Comet ML INC  #
# All rights reserved              #
####################################
#
# Vendored into cometx from comet-ml/code-panels
#   python-panel-engine/compute_engine_server/__main__.py
#   at commit b57eab54.
#
# This is the code Comet's Python panel server uses to turn a panel's source
# into a Streamlit page. `cometx panel-studio` uses it so a panel runs locally
# the same way it runs on comet.com. Changes from the original, all marked
# "cometx:" below:
#   * The Flask server is left out; only the page-building functions are here.
#   * Paths into the server's container (/app/app/pages, /home/stuser/...) are
#     arguments instead.
#   * The preamble drops the container's HOME and user site-packages setup.
#   * `%pip install` imports cometx.panel_pip_utils, and locks with filelock
#     only when it is installed.
"""
Build a Streamlit page from a Python panel's code, as Comet's panel server does.
"""

import hashlib
import os

import requests

# Following template should have "{", "\n", "}" escaped:
PIP_INSTALL_TEMPLATE = """
import cometx.panel_pip_utils as comet_pip_utils
try:
    from filelock import FileLock, Timeout
except ImportError:  # cometx: locking is optional when running locally
    FileLock = None

package_names, debug, flags = comet_pip_utils.process_args('''{command_line}''')


def _install_packages():
    installed_packages = comet_pip_utils.get_pip_packages()
    to_install = []
    for package_name in package_names:
        if not comet_pip_utils.is_pip_installed_package(package_name, installed_packages):
            to_install.append(package_name)
    if to_install:
        with st.spinner(text="Installing packages..."):
            stderr = comet_pip_utils.pip_install_packages(to_install, flags)
        show_full_error = False
        for line in stderr:
            if line.startswith("ERROR"):
                if "pip's dependency resolver does not currently take into account all the packages" in line:
                    st.warning("WARNING: Some pip dependencies don't match exactly", icon="⚠")
                else:
                    st.error("pip install error", icon="🚨")
                    show_full_error = True
                break
        if show_full_error or debug:
            st.code("\\n".join(stderr), language=None)
            st.stop()


if FileLock is None:
    _install_packages()
else:
    message = None
    count = 0
    while count < 10:
        lock = FileLock({lock_path!r})
        try:
            with lock.acquire(timeout=10):
                _install_packages()
                break
        except Timeout:
            if message:
                message.empty()
            count += 1
            message = st.warning(f"Unable to acquire lock for pip installer; trying again {{count}}...", icon="⚠")
            continue

    if message:
        message.empty()
    if count == 10:
        st.error("Unable to acquire lock for pip installer; try again later", icon="🚨")
        st.stop()
    del message, count
del comet_pip_utils, _install_packages
"""


def build_preamble(cfg):
    # Inject the os.environ variables for comet_ml
    config = "\n".join([f'    "{key}": {repr(value)},' for key, value in cfg.items()])
    # cometx: the server's HOME, user site-packages and cache setup is dropped.
    preamble = """
from comet_ml.exceptions import Unauthorized
import streamlit as st
import os
import sys
if "comet_config_override" not in st.session_state:
    st.session_state["comet_config_override"] = {{}}
key = os.path.basename(__file__)
config = {{
{config}
}}
st.session_state["comet_config_override"][key] = config
os.environ["COMET_PANEL_INSTANCE_ID"] = config.get("COMET_PANEL_INSTANCE_ID", "") or ""
print = st.write
del os, sys
# User code:
""".format(
        config=config,
    )
    return preamble


def get_or_download_code(url, cache_dir):
    """
    Get code from cache, or download.

    cometx: cache_dir was /home/stuser/.streamlit/code.
    """
    flag = None
    if " " in url:
        flag, url = [item.strip() for item in url.split(" ", 1)]
    cache_file = hashlib.md5(url.encode()).hexdigest() + ".py"
    cache_path = os.path.join(cache_dir, cache_file)
    if os.path.exists(cache_path) and flag != "-f":
        return (open(cache_path).read() + "\n").splitlines()
    else:
        os.makedirs(cache_dir, exist_ok=True)
        response = requests.get(url, allow_redirects=True)
        code = "# %%include %s\n" % url
        code += response.content.decode()
        with open(cache_path, "w") as fp:
            fp.write(code)
        return (code + "\n").splitlines()


def process_magic_commands(code, cache_dir, lock_path):
    """
    Process code for magic commands:
        * Extract '%pip install PACKAGE...' and return user code and pip code.
        * Replace '%include URL' user code with retrieved code

    cometx: cache_dir and lock_path were fixed paths in the server's container.
    """
    user_code = ""
    initial_code = ""
    lines = code.splitlines()
    expanded_lines = []
    command_line = None
    for line in lines:
        if line.startswith("%include "):
            # Replace line with code:
            _, url = line.split(" ", 1)
            try:
                expanded_lines.extend(get_or_download_code(url, cache_dir))
                continue
            except Exception as exc:
                line = """st.error("%s: %s")""" % (line, exc)

        expanded_lines.append(line)

    for line in expanded_lines:
        if line.startswith("%pip install"):
            command_line = line[12:].strip()
        elif line.startswith("st.set_page_config("):
            # TODO: assumes all on one line, and that it uses "st."
            initial_code += line + "\n"
        else:
            user_code += line + "\n"

    if command_line:
        pip_code = initial_code + PIP_INSTALL_TEMPLATE.format(
            command_line=command_line, lock_path=lock_path
        )
    else:
        pip_code = initial_code
    return user_code, pip_code


def contains_magic_commands(code):
    """
    Return lines that start with a magic command.
    """
    return any([line.startswith("%") for line in code.splitlines()])


def copy_user_code(code, filename, cfg, cache_dir, lock_path):
    """
    Write the panel's code as a Streamlit page.

    cometx: the page path is the `filename` argument (the server used
    /app/app/pages/{instance_id}.py), and cache_dir/lock_path are passed on.
    The page's file name must start with "page_", which is how comet_ml finds
    this page's config.
    """
    preamble = build_preamble(cfg)
    if contains_magic_commands(code):
        user_code, pip_code = process_magic_commands(code, cache_dir, lock_path)
    else:
        user_code, pip_code = code, ""

    if user_code.strip():
        # Indent user code:
        user_code = "\n".join(["    %s" % line for line in user_code.splitlines()])
        # Surround with try/except:
        user_code = """
try:
{user_code}
except Unauthorized:
    st.markdown(
        "### :warning: Your Panel API key has expired\\n" +
        "This happens to protect your data. " +
        "Please reload this page to refresh it."
    )
markdown_to_inject = '''
<style>
button[title="View fullscreen"]{{
  display: none;
}}
.element-container:has(div.stMarkdown style) {{
  display: none;
}}
</style>
'''
st.markdown(markdown_to_inject, unsafe_allow_html=True)
""".format(
            user_code=user_code
        )
    # Copy to Streamlit Page:
    with open(filename, "w") as fp:
        # write modified user code
        fp.write(preamble)
        if pip_code:
            fp.write(pip_code)
        fp.write(user_code)
