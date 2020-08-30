class BaseLogger(object):
    def __init__(self, tb_logger, log_path, cfg):
        super().__init__()
        self.cfg = cfg
        self.NAME = 'base'
        self.tb = tb_logger
        self.log_path = log_path
        # make dir

    def log_phase(self):
        pass

    def log_batch(self, batch):
        pass
