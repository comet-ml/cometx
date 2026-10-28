# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Copyright (C) 2015-2025 Comet ML INC
#  This file can not be copied and/or distributed without
#  the express permission of Comet ML Inc.
# *******************************************************

#
# Vendored into cometx from comet-ml/code-panels
#   python-panel-engine/comet-pip-utils/comet_pip_utils/pip_utils.py
#   at commit b57eab54. One change, marked "cometx:": --trusted-host takes a
#   value, so it is no longer treated as a flag without one.
# Used by the `%pip install` magic in `cometx panel-studio`.
# flake8: noqa -- kept as vendored.
import io
import shlex
import subprocess
import sys
from typing import Dict, List, Optional, Tuple, Union

from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import Version


def get_pip_packages() -> List[str]:
    """
    Get a list of installed pip packages in the current environment.

    Runs `pip freeze` command to retrieve all installed packages and their versions.
    Filters out packages that start with "-" or contain " @ " (which indicate
    editable installs or other special cases).

    Returns:
        List[str]: A list of package requirement strings in the format "package==version"

    Raises:
        subprocess.CalledProcessError: If the pip freeze command fails
    """
    cmd = [sys.executable, "-m", "pip", "freeze"]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    return [
        name
        for name in result.stdout.strip().split("\n")
        if not name.startswith("-") and " @ " not in name
    ]


def parse_frozen_requirements(frozen_reqs: List[str]) -> Dict[str, Version]:
    """
    Parse frozen requirements into a dictionary mapping package names to versions.

    Takes a list of frozen requirement strings (from pip freeze) and converts them
    into a dictionary where keys are package names (lowercase) and values are
    Version objects.

    Args:
        frozen_reqs: List of requirement strings in the format "package==version"

    Returns:
        Dict[str, Version]: Dictionary mapping package names to their versions

    Note:
        Only handles requirements with exact version specifiers (==). Other specifiers
        are skipped with a warning message.
    """
    installed = {}
    for line in frozen_reqs:
        try:
            req = Requirement(line)
        except InvalidRequirement:
            print(f"Skipping malformed requirement: {line}")
            continue
        # pip freeze only produces `==version`
        specifiers = list(req.specifier)
        if len(specifiers) != 1 or specifiers[0].operator != "==":
            print(f"Unexpected frozen requirement: {line}")
            continue
        installed[req.name.lower()] = Version(specifiers[0].version)
    return installed


def is_pip_installed_package(package_name: str, packages: List[str]) -> bool:
    """
    Check if a package is installed and satisfies the given requirement.

    Compares a package requirement against a list of installed packages to determine
    if the package is installed and meets the version constraints.

    Args:
        package_name: Package requirement string (e.g., "package>=1.0.0")
        packages: List of installed package strings from pip freeze

    Returns:
        bool: True if the package is installed and satisfies the requirement,
              False otherwise

    Raises:
        InvalidRequirement: if package_name is not valid
    """
    req = Requirement(package_name)
    installed = parse_frozen_requirements(packages)

    installed_version = installed.get(req.name.lower())
    if installed_version is None:
        return False

    if not req.specifier:
        return True

    # check if installed_version satisfies the requirement specifier
    return installed_version in req.specifier


def pip_install_packages(
    packages: Union[str, List[str]], flags: Optional[List[str]] = None
) -> List[str]:
    """
    Install packages using pip with optional flags.

    Executes pip install command for the specified packages with optional flags.
    Returns any error messages from the installation process.

    Args:
        packages: Package name(s) to install. Can be a single string or list of strings
        flags: Optional list of pip flags to pass to the install command

    Returns:
        List[str]: List of error messages from the pip install process

    Raises:
        subprocess.CalledProcessError: If the pip install command fails
    """
    if isinstance(packages, str):
        packages = [packages]
    if flags is None:
        flags = []

    cmd = [sys.executable, "-m", "pip", "install"] + packages + flags

    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    return result.stderr.split("\n")


def unload_package(qualified_name: str) -> None:
    """
    Unload a package and all its submodules from sys.modules.

    Removes the specified package and all modules that start with the package name
    from the sys.modules dictionary. This is useful for reloading packages or
    cleaning up module references.

    Args:
        qualified_name: Package requirement string (e.g., "package==3.0.3").
                       The package name is extracted from this string.
    """
    package_name = extract_package_name(qualified_name)
    if package_name:
        for key in list(sys.modules.keys()):
            if key.startswith(package_name + ".") or key == package_name:
                del sys.modules[key]


def extract_package_name(package_name: str) -> Optional[str]:
    """
    Extract the package name from a requirement string.

    Parses a requirement string (e.g., "package==3.0.3") and extracts just
    the package name without version constraints.

    Args:
        package_name: A requirement string to parse (e.g., "package>=1.0.0")

    Returns:
        Optional[str]: The package name if the requirement string is valid,
                      None if the requirement string is invalid

    Note:
        Invalid requirement strings are logged to stdout but don't raise exceptions
    """
    try:
        r = Requirement(package_name)
        return r.name
    except Exception as e:
        print(f"Invalid requirement string '{package_name}': {e}")
        return None


def process_args(args: str) -> Tuple[List[str], bool, List[str]]:
    """
    Parse pip command arguments to extract packages, debug flag, and other flags.

    Takes a string of pip command arguments and parses them to separate:
    - Package names (positional arguments)
    - Debug flag (--debug)
    - Other flags (everything else)

    Args:
        args: String containing pip command arguments (e.g., "xxx aitk --debug --extra-index-url http://localhost:8000")

    Returns:
        Tuple[List[str], bool, List[str]]: A tuple containing:
            - packages: List of package names
            - debug: Boolean indicating if --debug flag was present
            - flags: List of other flags suitable for pip install command

    Example:
        >>> args = "xxx aitk --debug --extra-index-url http://localhost:8000 --index-url=http://localhost:9000"
        >>> packages, debug, flags = process_args(args)
        >>> packages
        ['xxx', 'aitk']
        >>> debug
        True
        >>> flags
        ['--extra-index-url', 'http://localhost:8000', '--index-url=http://localhost:9000']
    """
    # Parse the arguments using shlex to handle quoted strings properly
    parsed_args = shlex.split(args)

    packages = []
    debug = False
    flags = []

    # Known pip flags that don't take values
    boolean_flags = {
        "--debug",
        "--verbose",
        "--quiet",
        "--no-deps",
        "--upgrade",
        "--force-reinstall",
        "--no-cache-dir",
        "--user",
        "--no-index",
        "--pre",
        # cometx: "--trusted-host" removed here; it takes a value, and is in
        # value_flags below.
        "--disable-pip-version-check",
        "--no-warn-script-location",
        "--no-warn-conflicts",
        "--no-warn-dependencies",
        "--no-warn-about-pip-version",
        "--no-warn-about-pip-version-check",
    }

    # Known pip flags that take values
    value_flags = {
        "--index-url",
        "--extra-index-url",
        "--find-links",
        "--constraint",
        "--requirement",
        "--build",
        "--src",
        "--prefix",
        "--root",
        "--target",
        "--upgrade-strategy",
        "--install-option",
        "--global-option",
        "--trusted-host",
        "--cert",
        "--client-cert",
        "--keyring-provider",
        "--retries",
        "--timeout",
        "--cache-dir",
        "--no-cache-dir",
        "--disable-pip-version-check",
        "--no-warn-script-location",
        "--no-warn-conflicts",
        "--no-warn-dependencies",
        "--no-warn-about-pip-version",
        "--no-warn-about-pip-version-check",
    }

    i = 0
    while i < len(parsed_args):
        arg = parsed_args[i]

        # Check for --debug flag
        if arg == "--debug":
            debug = True
        # Check for boolean flags
        elif arg in boolean_flags:
            flags.append(arg)
        # Check for flags that take values (like --extra-index-url)
        elif arg.startswith("--") and "=" not in arg:
            # This is a flag that might take a value
            if i + 1 < len(parsed_args) and not parsed_args[i + 1].startswith("-"):
                # Next argument is the value for this flag
                flags.extend([arg, parsed_args[i + 1]])
                i += 1  # Skip the value in next iteration
            else:
                # Flag without value
                flags.append(arg)
        # Check for flags with = (like --index-url=http://localhost:9000)
        elif arg.startswith("--") and "=" in arg:
            flags.append(arg)
        # Everything else is considered a package name
        else:
            packages.append(arg)

        i += 1

    return packages, debug, flags
