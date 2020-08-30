"""
Main program
May the Force be with you.

This main file is used on slurm server without interactive check of config
"""
from torch.utils.data import DataLoader

from dataset import get_dataset
from logger import get_logger
from core.models import get_model
from core.trainer import Trainer
from config import get_cfg

# preparer configuration
cfg = get_cfg(interactive=False)

# prepare dataset
DatasetClass = get_dataset(cfg.DATASET)
dataloader_dict = dict()
for mode in cfg.MODES:
    phase_dataset = DatasetClass(cfg, mode=mode)
    dataloader_dict[mode] = DataLoader(phase_dataset, batch_size=cfg.BATCHSIZE,
                                       shuffle=True if mode in ['train'] else False,
                                       num_workers=cfg.DATALOADER_WORKERS, pin_memory=True,
                                       drop_last=True)

# prepare models
ModelClass = get_model(cfg.MODEL)
model = ModelClass(cfg)

# prepare logger
LoggerClass = get_logger(cfg.LOGGER)
logger = LoggerClass(cfg)

# register dataset, models, logger to trainer
trainer = Trainer(cfg, model, dataloader_dict, logger)

# start training
epoch_total = cfg.EPOCH_TOTAL + (cfg.RESUME_EPOCH_ID if cfg.RESUME else 0)
while trainer.do_epoch() <= cfg.EPOCH_TOTAL:
    pass
