import importlib


def get_logger(alias):
    module = importlib.import_module('logger.' + alias.lower())
    return module.Logger
