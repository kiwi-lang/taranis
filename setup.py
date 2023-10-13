#!/usr/bin/env python
import os
from pathlib import Path

from setuptools import setup

with open("taranis/core/__init__.py") as file:
    for line in file.readlines():
        if "version" in line:
            version = line.split("=")[1].strip().replace('"', "")
            break

extra_requires = {"plugins": ["importlib_resources"]}
extra_requires["all"] = sorted(set(sum(extra_requires.values(), [])))

if __name__ == "__main__":
    setup(
        name="taranis",
        version=version,
        extras_require=extra_requires,
        description="Lightweight Machine Learning Framework",
        long_description=(Path(__file__).parent / "README.rst").read_text(),
        author="Pierre Delaunay",
        author_email="pierre@delaunay.io",
        license="BSD 3-Clause License",
        url="https://taranis.readthedocs.io",
        classifiers=[
            "License :: OSI Approved :: BSD License",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Operating System :: OS Independent",
        ],
        packages=[
            "taranis.core",
            "taranis.plugins.example",
        ],
        setup_requires=["setuptools"],
        install_requires=["importlib_resources"],
        package_data={
            "taranis.data": [
                "taranis/data",
            ],
        },
    )
