from .default import get_default_cfg
from .cmdline import merge_from_cmdline
from .startup import startup
from os.path import join


def get_cfg(filename=None, interactive=True):
    """
    config priority: cmd>cfg-file>default
    :param filename: filename XXX in $project/config/config_files/XXX.yaml
    :return: a frozen configuration
    """
    # get default
    cfg = get_default_cfg()
    # get config file parameter, will do another time later to overwrite local file config
    cfg = merge_from_cmdline(cfg)
    # merge local file, cmd>file
    if cfg.CONFIG_FILE is not 'None':
        filepath = join(cfg.ROOT_DIR, 'config', 'config_files', cfg.CONFIG_FILE)
        cfg.merge_from_file(filepath)
    elif filename is not None:
        filepath = join(cfg.ROOT_DIR, 'config', 'config_files', filename)
        cfg.merge_from_file(filepath)
    # merge cmd line
    cfg = merge_from_cmdline(cfg)
    # startup
    startup(cfg, interactive)
    cfg.freeze()
    return cfg
