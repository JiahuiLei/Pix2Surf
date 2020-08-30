import importlib


def get_dataset(alias):
    module = importlib.import_module('dataset.' + alias.lower())
    return module.Dataset
