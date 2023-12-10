#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from setuptools import find_packages, setup

# Package meta-data.
NAME = 'financial_india_data'
DESCRIPTION = "Gathering Data for India"
EMAIL = "------"
AUTHOR = "----------"
REQUIRES_PYTHON = ">=3.9.0"


# The rest no need to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# Trove Classifiers: https://pypi.org/classifiers/

long_description = DESCRIPTION

# Load the package's VERSION file as a dictionary.
about = {}
ROOT_DIR = Path(__file__).resolve().parent
print(ROOT_DIR)
PACKAGE_DIR = ROOT_DIR / 'findata'
with open(PACKAGE_DIR / "VERSION" , 'r') as f:
    _version = f.read().strip()
    _verlist = _version.split(".")
    last = str(int(_verlist[-1])+1)
    _verlist[-1] = last
    _version = ".".join(_verlist)
    about["__version__"] = _version

with open(PACKAGE_DIR / "VERSION" , 'w') as f:
    f.write(about["__version__"])

# What packages are required for this module to be executed?
def list_reqs(fname="requirements.txt"):
    with open(ROOT_DIR / fname) as fd:
        return fd.read().splitlines()

# Where the magic happens:
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    packages=find_packages(),
    package_data={"findata": [
        "VERSION",
        "data/company_list.csv"
    ]},
    install_requires=list_reqs(),
    extras_require={},
    include_package_data=True,
    license="BSD-3",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
)
