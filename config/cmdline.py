import argparse


def merge_from_cmdline(cfg):
    """
    Merge some usually changed settings from cmd
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='None', help="Choose config file")
    parser.add_argument('--gpu', type=str, default=None, help="Set GPU device, type=str, e.g: --gpu=0,1,2 ")
    parser.add_argument('--logdir', type=str, default=None, help='log dir name in $project/log/XXXX/.... e.g. exp')
    cmd = vars(parser.parse_args())
    if cmd['config'] is not 'None':
        cfg.CONFIG_FILE = cmd['config']
    if cmd['gpu'] is not None:
        cfg.GPU = [int(id) for id in cmd['gpu'].split(",")]
    if cmd['logdir'] is not None:
        cfg.LOG_DIR = cmd['logdir']
    return cfg
