"""
06/31/2019
Each passed in batch must contain info:
    1. method of model to save itself
    2. the way to parsing the batch keys
    3. meta information of each sample
    4. batch head (phase)
Logger can be selected from config file
"""
import os
from tensorboardX import SummaryWriter as writer
from .logger_meta import LOGGER_REGISTED


class Logger(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.phase = 'INIT'
        self.epoch = -1
        tb_path = os.path.join(cfg.ROOT_DIR, 'log', cfg.LOG_DIR, 'tensorboardx')
        self.tb_writer = writer(tb_path)
        self.lg_list = self.compose(self.cfg.LOGGER_SELECT)

    def compose(self, names):
        loggers_list = list()
        mapping = LOGGER_REGISTED
        for name in names:
            if name in mapping.keys():
                loggers_list.append(mapping[name](self.tb_writer, os.path.join(
                    os.path.join(self.cfg.ROOT_DIR, 'log', self.cfg.LOG_DIR, name)), self.cfg))
            else:
                raise Warning('Required logger ' + name + ' not found!')
        return loggers_list

    def log(self, batch):
        # if phase changes, first log phase then log batch
        phase = batch['phase']
        epoch = batch['epoch-id']
        if self.phase != 'INIT' and phase != self.phase:
            for lgr in self.lg_list:
                lgr.log_phase()
        elif self.epoch != -1 and epoch != self.epoch:
            for lgr in self.lg_list:
                lgr.log_phase()
        self.phase = phase
        self.epoch = epoch
        for lgr in self.lg_list:
            lgr.log_batch(batch)

    def end_log(self):
        for lgr in self.lg_list:
            lgr.log_phase()
