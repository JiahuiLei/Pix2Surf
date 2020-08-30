from .base_logger import BaseLogger
import matplotlib
import os
import numpy as np
from cv2 import imwrite
import cv2
import torch
from copy import deepcopy

matplotlib.use('Agg')


class ImageLogger(BaseLogger):
    def __init__(self, tb_logger, log_path, cfg):
        super().__init__(tb_logger, log_path, cfg)
        self.NAME = 'image'
        os.makedirs(self.log_path, exist_ok=True)
        self.visual_interval_epoch = cfg.VIS_PER_N_EPOCH
        self.visual_interval_batch = cfg.VIS_PER_N_BATCH
        self.visual_one = cfg.VIS_ONE_PER_BATCH
        self.visual_train_interval_batch = cfg.VIS_TRAIN_PER_BATCH

    def log_batch(self, batch):
        # get data
        if not self.NAME in batch['parser'].keys():
            return
        keys_list = batch['parser'][self.NAME]
        if len(keys_list) == 0:
            return
        data = batch['data']
        phase = batch['phase']
        current_batch = batch['batch-id']
        current_epoch = batch['epoch-id']
        meta_info = batch['meta-info']  # {k:[info0,info1,info2,....]}
        # check dir
        os.makedirs(os.path.join(self.log_path, 'epoch_%d' % current_epoch), exist_ok=True)
        # check whether need log
        if phase.lower() == 'train':
            if current_batch % self.visual_train_interval_batch != 0 or current_batch % self.visual_interval_batch != 0:
                return
        else:
            if current_epoch % self.visual_interval_epoch != 0 or current_batch % self.visual_interval_batch != 0:
                return
        # for each key
        for img_key in keys_list:  # for each key
            kdata = data[img_key]
            if isinstance(kdata, list):
                assert len(kdata[0].shape) == 4
                nbatch = kdata[0].shape[0]
                kdata = deepcopy(kdata)
            else:
                assert len(kdata.shape) == 4
                nbatch = kdata.shape[0]
                if isinstance(kdata, torch.Tensor):
                    kdata = [deepcopy(kdata.detach().cpu().numpy())]
                else:
                    kdata = [deepcopy(kdata)]
            # convert to ndarray
            if isinstance(kdata[0], torch.Tensor):
                for i, tensor in enumerate(kdata):
                    kdata[i] = tensor.detach().cpu().numpy()
            # for each sample in batch
            for batch_id in range(nbatch):
                # get meta postfix
                prefix = ""
                for k, v in meta_info.items():
                    prefix += k + "_" + str(v[batch_id]) + "_"
                # now all cases are converted to list of image
                nview = len(kdata)
                for view_id in range(nview):
                    img = kdata[view_id][batch_id]  # 3*W*H / 1*W*H
                    assert img.ndim == 3
                    # first process image
                    color_flag = False
                    if img.shape[0] == 1:
                        color_flag = True
                        cm = matplotlib.cm.get_cmap('viridis')
                        img = cm(img.squeeze(0))[..., :3]
                        img = img.transpose(2, 0, 1)
                        img *= 255
                    else:
                        img *= 255.0 if img.max() < 200 else 1
                    img = np.clip(img, a_min=0, a_max=255)
                    img = img.astype(np.uint8)
                    self.tb.add_image(phase + '/' + img_key, img if color_flag else img[[2, 1, 0], ...], current_epoch)
                    # save to file
                    img = img.transpose(1, 2, 0)
                    filename = os.path.join(
                        self.log_path, 'epoch_%d' % current_epoch,
                                       prefix + 'b%d_v%d_' % (batch_id, view_id) + img_key + '.png'
                    )
                    if color_flag:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    imwrite(filename, img)
                if self.visual_one:
                    break

    def log_phase(self):
        pass
