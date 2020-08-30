"""
Default configuration
"""
from yacs.config import CfgNode as CN
import os


def get_default_cfg():
    cfg = CN()
    # dataset
    cfg.DATASET = ''
    cfg.DATASET_ROOT = ''
    cfg.DATASET_CATES = []
    cfg.DATASET_INDEX = []
    cfg.PROPORTION = 1.0  # Ratio of first K samples in the dataset list
    cfg.ADD_BACKGROUND_NOISE = False
    cfg.ROOT_DIR = os.getcwd()
    cfg.CONFIG_FILE = 'None'
    cfg.MODES = ['train', 'test']
    cfg.GPU = [0]

    cfg.BATCHSIZE = 2
    cfg.DATALOADER_WORKERS = 2

    cfg.MODEL = 'None'
    cfg.MODEL_INIT_PATH = ['None']
    cfg.LR = 0.001
    cfg.EPOCH_TOTAL = 1

    cfg.ADAM_BETA1 = 0.5
    cfg.ADAM_BETA2 = 0.9
    cfg.OPTI_DECAY_RATE = 1.0
    cfg.OPTI_DECAY_INTERVAL = 1000
    cfg.OPTI_DECAY_MIN = 0.00001

    cfg.LOG_DIR = 'debug'  # this param is the name under $ProjectRoot/log/

    cfg.RESUME = False  # If true, check whether the log dir exists, if not, just start a new one
    cfg.RESUME_EPOCH_ID = 0

    cfg.LOGGER = 'logger_s1'
    cfg.LOGGER_SELECT = ['metric']

    cfg.MODEL_SAVE_PER_N_EPOCH = 5

    cfg.VIS_PER_N_EPOCH = 1
    cfg.VIS_ONE_PER_BATCH = True
    # visualization config for non-training phase
    cfg.VIS_PER_N_BATCH = 1
    # visualization config for trianing phase
    cfg.VIS_TRAIN_PER_BATCH = 20

    cfg.BACKUP_FILES = []

    # for postprocessing (render)
    cfg.RENDER_MASK_TH = 0.02  # distance th for valid neighborhood point in 3D
    cfg.RENDER_IMAGE_MASK_NK = 4  # image mask optimization n nearest neighbor

    return cfg
