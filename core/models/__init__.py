import importlib


def get_model(alias):
    module = importlib.import_module('core.models.' + alias)
    return module.Model
