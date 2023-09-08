import copy
import json
import os

_options = {}


def flatten(dictionary):
    """Turn all nested dict keys into a {key}.{subkey} format"""

    def _flatten(dictionary):
        if dictionary == {}:
            return dictionary

        key, value = dictionary.popitem()
        if not isinstance(value, dict) or not value:
            new_dictionary = {key: value}
            new_dictionary.update(_flatten(dictionary))
            return new_dictionary

        flat_sub_dictionary = _flatten(value)
        for flat_sub_key in list(flat_sub_dictionary.keys()):
            flat_key = key + "." + flat_sub_key
            flat_sub_dictionary[flat_key] = flat_sub_dictionary.pop(flat_sub_key)

        new_dictionary = flat_sub_dictionary
        new_dictionary.update(_flatten(dictionary))
        return new_dictionary

    return _flatten(copy.deepcopy(dictionary))


def load_configuration(file_name):
    global _options

    options = json.load(open(file_name))
    _options = flatten(options)


def set_option(name, value):
    global _options
    _options[name] = value


def options(name, default, type=str):
    """Look for an option locally and using the environment variables
    Environment variables are use as the ultimate overrides
    """

    env_name = name.upper().replace(".", "_")
    value = os.getenv(f"TARANIS_{env_name}", None)

    if not value:
        return type(_options.get(name, default))

    return type(value)


def option(name, default, type=str):
    return options(name, default, type=type)
