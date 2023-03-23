import taranis.plugins
from taranis.core import discover_plugins


def test_plugins():
    plugins = discover_plugins(taranis.plugins)

    assert len(plugins) == 1
