"""Top level module for taranis"""

import importlib
import json
import pkgutil

import importlib_resources

__descr__ = "Lightweight Machine Learning Framework"
__version__ = "0.0.1"
__license__ = "BSD 3-Clause License"
__author__ = "Pierre Delaunay"
__author_email__ = "pierre@delaunay.io"
__copyright__ = "2023 Pierre Delaunay"
__url__ = "https://github.com/kiwi-lang/taranis"


def discover_plugins(module):
    """Discover uetools plugins"""
    path = module.__path__
    name = module.__name__

    plugins = {}

    for _, name, _ in pkgutil.iter_modules(path, name + "."):
        plugins[name] = importlib.import_module(name)
        print(f" - Found plugin: {name}")

    return plugins


def _():
    data_path = importlib_resources.files("taranis.data")

    with open(data_path / "data.json", encoding="utf-8") as file:
        print(json.dumps(json.load(file), indent=2))
