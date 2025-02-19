# -*- coding: utf-8 -*-
# ***************************************
#                              __
#   _________  ____ ___  ___  / /__  __
#  / ___/ __ \/ __ `__ \/ _ \/ __/ |/_/
# / /__/ /_/ / / / / / /  __/ /__>  <
# \___/\____/_/ /_/ /_/\___/\__/_/|_|
#
#
#  Copyright (c) 2022 Cometx Development
#      Team. All rights reserved.
# ***************************************
"""
cometx setup
"""
import io
import os

import setuptools

HERE = os.path.abspath(os.path.dirname(__file__))


def get_version(file, name="__version__"):
    """Get the version of the package from the given file by
    executing it and extracting the given `name`.
    """
    path = os.path.realpath(file)
    version_ns = {}
    with io.open(path, encoding="utf8") as f:
        exec(f.read(), {}, version_ns)
    return version_ns[name]


__version__ = get_version(os.path.join(HERE, "cometx/_version.py"))

with io.open(os.path.join(HERE, "README.md"), encoding="utf8") as fh:
    long_description = fh.read()

setup_args = dict(
    name="cometx",
    version=__version__,
    url="https://github.com/comet-ml/cometx/",
    author="cometx development team",
    description="Python tools for Comet",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "comet_ml>=3.49.1",
    ],
    extras_require={
        "all": [
            "pillow>=11.1.0",
            "opik",
            "comet_mpm",
        ],
    },
    packages=[
        "cometx.cli",
        "cometx.tools",
        "cometx.framework",
        "cometx.framework.comet",
        "cometx",
    ],
    entry_points={"console_scripts": ["cometx = cometx.cli:main"]},
    python_requires=">=3.6",
    license="MIT License",
    platforms="Linux, Mac OS X, Windows",
    keywords=["ai", "artificial intelligence", "python", "machine learning"],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Framework :: Jupyter",
    ],
)

if __name__ == "__main__":
    setuptools.setup(**setup_args)
