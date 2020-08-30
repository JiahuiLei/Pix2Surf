import os
from .base_logger import BaseLogger


class ModelLogger(BaseLogger):
    def __init__(self, tb_logger, log_path, cfg):
        super().__init__(tb_logger, log_path, cfg)
        self.NAME = 'model'
        os.makedirs(self.log_path, exist_ok=True)
        self.phase = 'train'
        self.current_epoch = -1
        self.save_interval = cfg.MODEL_SAVE_PER_N_EPOCH
        self.save_method = None

    def log_batch(self, batch):
        self.phase = batch['phase']
        self.current_epoch = batch['epoch-id']
        self.save_method = batch['save-method']

    def log_phase(self):
        if self.phase == 'train' and (self.current_epoch % self.save_interval == 0):
            self.save_method(os.path.join(self.log_path, "epoch_%d.model" % self.current_epoch))
        if self.phase == 'train':
            if any([True if fn.endswith('latest.model') else False for fn in os.listdir(self.log_path)]):
                os.system('rm ' + os.path.join(self.log_path, "epoch_*_latest.model"))
            self.save_method(os.path.join(self.log_path, "epoch_%d_latest.model" % self.current_epoch))
