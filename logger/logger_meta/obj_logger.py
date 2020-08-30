from .base_logger import BaseLogger
import matplotlib
import os
import torch
from tk3dv.nocstools import datastructures as ds
import cv2 as cv
from copy import deepcopy

matplotlib.use('Agg')


class ObjectLogger(BaseLogger):
    def __init__(self, tb_logger, log_path, cfg):
        super().__init__(tb_logger, log_path, cfg)
        self.NAME = 'obj'
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
        for obj_key in keys_list:  # for each key
            nocs = data[obj_key[0]]
            color = data[obj_key[1]]
            if isinstance(nocs, list):
                assert isinstance(color, list)
                assert len(nocs[0].shape) == 4
                assert len(color[0].shape) == 4
                assert nocs[0].shape[0] == color[0].shape[0]
                nbatch = nocs[0].shape[0]
            else:
                assert len(nocs.shape) == 4
                assert len(color.shape) == 4
                assert nocs.shape[0] == color.shape[0]
                nbatch = nocs.shape[0]
                nocs = [nocs]
                color = [color]

            # convert to ndarray
            if isinstance(nocs[0], torch.Tensor):
                for i, tensor in enumerate(nocs):
                    nocs[i] = tensor.detach().cpu().numpy()
            if isinstance(color[0], torch.Tensor):
                for i, tensor in enumerate(color):
                    color[i] = tensor.detach().cpu().numpy()
            nocs = deepcopy(nocs)
            color = deepcopy(color)

            # for each sample in batch
            for batch_id in range(nbatch):
                # get meta postfix
                prefix = ""
                for k, v in meta_info.items():
                    prefix += k + "_" + str(v[batch_id]) + "_"
                # now all case convert to list of image
                nview = len(nocs)
                for view_id in range(nview):
                    nocs_map = nocs[view_id][batch_id]  # 3*H*W [0,1]
                    color_map = color[view_id][batch_id]  # 3*H*W [0,1]

                    assert nocs_map.ndim == 3
                    assert nocs_map.shape[0] == 3
                    assert color_map.ndim == 3
                    assert color_map.shape[0] == 3
                    nocs_map = nocs_map.transpose(1, 2, 0)  # H*W*3
                    color_map = color_map.transpose(1, 2, 0)  # H*W*3

                    # smaller meshes
                    nocs_map = cv.resize(nocs_map, dsize=(int(nocs_map.shape[1] / 4), int(nocs_map.shape[0] / 4)),
                                         interpolation=cv.INTER_NEAREST)
                    color_map = cv.resize(color_map, dsize=(int(color_map.shape[1] / 4), int(color_map.shape[0] / 4)),
                                          interpolation=cv.INTER_NEAREST)

                    # save to file
                    filename = os.path.join(
                        self.log_path, 'epoch_%d' % current_epoch,
                                       prefix + 'b%d_v%d_' % (batch_id, view_id) +
                                       obj_key[0] + '-' + obj_key[1] + '.obj'
                    )
                    # save here
                    # need to convert color, need to *255
                    if nocs_map.max() < 1 + 1e-4:
                        nocs_map = nocs_map[:, :, [2, 1, 0]] * 255.0
                    if color_map.max() < 1 + 1e-4:
                        color_map = color_map[:, :, [2, 1, 0]] * 255.0
                    tk3dv_nocs_mp = ds.NOCSMap(nocs_map, RGB=color_map)
                    tk3dv_nocs_mp.serialize(filename)
                if self.visual_one:
                    break

    def log_phase(self):
        pass
