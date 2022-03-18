#!/usr/bin/env python
# coding=utf-8

"""The setup script."""

import ast
from typing import List

from setuptools import find_packages, setup  # type: ignore

with open("dlup_lightning_mil/__init__.py") as f:
    for line in f:
        if line.startswith("__version__"):
            version = ast.parse(line).body[0].value.s  # type: ignore
            break


# Get the long description from the README file
with open("README.md") as f:
    LONG_DESCRIPTION = f.read()


setup(
    author="Yoni Schirris",
    author_email="y.schirris@nki.nl",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    entry_points={
        "console_scripts": [
            "dlup_lightning_mil=dlup_lightning_mil.cli:main",
        ],
    },
    install_requires=[
    ],
    extras_require={
        "dev": [
        ],
    },
    license="Apache Software License 2.0",
    include_package_data=True,
    keywords="dlup_lightning_mil",
    name="dlup_lightning_mil",
    packages=find_packages(include=["dlup_lightning_mil", "dlup_lightning_mil.*"]),
    url="https://github.com/NKI-AI/dlup-lightning-mil",
    version=version,
    zip_safe=False,
)
