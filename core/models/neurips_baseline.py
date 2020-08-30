"""
ECCV Pix2Surf
Initialization with Neurips NOX training
"""

from .modelbase_v2 import ModelBase
from .modelbase_v2 import Network as NetBase
from core.net_bank.xnocs_segnet import SegNet
from core.net_bank.loss import MaskL2Loss
import os
import torch
from torch import nn


class Model(ModelBase):
    def __init__(self, cfg):
        super(Model, self).__init__(cfg)
        self.name = 'neurips-nox'
        self.cfg = cfg
        self.network = Network()
        self.optimizer = torch.optim.Adam(params=self.network.parameters(), lr=self.lr,
                                          betas=(self.cfg.ADAM_BETA1, self.cfg.ADAM_BETA2))
        self.resume = cfg.RESUME
        if self.resume:
            self.resume_id = cfg.RESUME_EPOCH_ID
            load_path = os.path.join(cfg.ROOT_DIR, 'log', cfg.LOG_DIR, 'model',
                                     'epoch_%d' % cfg.RESUME_EPOCH_ID + '.model')
            self.load_model(loadpath=load_path, current_model_state='cpu')
        elif cfg.MODEL_INIT_PATH != ['None']:
            self.load_model(loadpath=cfg.MODEL_INIT_PATH)
        self.to_gpus()
        # config output meaning
        self.output_info_dict = {
            'metric': ['batch-loss', 'reg-v-loss', 'reg-x-loss', 'mask-v-loss', 'mask-x-loss'],
            'image': ['rgb-v', 'nox-v-gt', 'nox-x-gt', 'nox-v-pred', 'nox-x-pred', 'mask-v', 'mask-x'],
        }

    def _preprocess(self, in_batch):
        device = torch.device("cuda")
        nox_v = in_batch['nox-v'].float().permute(0, 3, 1, 2).to(device) / 255.0  # [0,1]
        nox_x = in_batch['nox-x'].float().permute(0, 3, 1, 2).to(device) / 255.0  # [0,1]
        rgb_v = in_batch['rgb-v'].float().permute(0, 3, 1, 2).to(device) / 255.0  # [0,1]
        rgb_x = in_batch['rgb-x'].float().permute(0, 3, 1, 2).to(device) / 255.0  # [0,1]
        mask_v = in_batch['mask-v'].float().permute(0, 3, 1, 2).to(device)  # 0,1
        mask_x = in_batch['mask-x'].float().permute(0, 3, 1, 2).to(device)  # 0,1
        pack = {'rgb-v': rgb_v, 'rgb-x': rgb_x, 'nox-v': nox_v, 'nox-x': nox_x,
                'mask-v': mask_v, 'mask-x': mask_x}
        return {'to-nn': pack, 'meta-info': in_batch['info']}


class Network(NetBase):
    def __init__(self):
        super(Network, self).__init__()
        net_dict = {
            'seg-net': SegNet(out_channels=10)
        }
        self.network_dict = nn.ModuleDict(net_dict)
        # loss
        self.cls_criterion = nn.CrossEntropyLoss()  # not masked, for all pixel
        self.ml2_criterion = MaskL2Loss()  # masked for nocs regression

    def forward(self, pack, is_train=True):
        batch = dict()

        # make cnn prediction
        pred = self.network_dict['seg-net'](pack['rgb-v'])
        pred_nox_v = pred[:, :3, :, :]
        pred_nox_x = pred[:, 3:6, :, :]
        pred_score_v = pred[:, 6:8, :, :]
        pred_score_x = pred[:, 8:10, :, :]

        mask1c_v = pack['mask-v'][:, 0, :, :].unsqueeze(1).detach()
        mask_v_loss = self.cls_criterion(pred_score_v, mask1c_v.squeeze(1).long().detach())
        pred_mask_v = torch.argmax(pred_score_v, dim=1, keepdim=True).float()

        mask1c_x = pack['mask-x'][:, 0, :, :].unsqueeze(1).detach()
        mask_x_loss = self.cls_criterion(pred_score_x, mask1c_x.squeeze(1).long().detach())
        pred_mask_x = torch.argmax(pred_score_x, dim=1, keepdim=True).float()

        reg_v_loss = self.ml2_criterion(pred_nox_v, pack['nox-v'], mask1c_v, True)
        reg_x_loss = self.ml2_criterion(pred_nox_x, pack['nox-x'], mask1c_x, True)

        # summary
        batch['batch-loss'] = ((reg_v_loss + reg_x_loss) * 0.3 + (mask_v_loss + mask_x_loss) * 0.7).unsqueeze(0)

        batch['reg-v-loss'] = reg_v_loss.detach().unsqueeze(0)
        batch['reg-x-loss'] = reg_x_loss.detach().unsqueeze(0)
        batch['mask-v-loss'] = mask_v_loss.detach().unsqueeze(0)
        batch['mask-x-loss'] = mask_x_loss.detach().unsqueeze(0)

        batch['nox-v-gt'] = pack['nox-v'] * pack['mask-v'] + (1.0 - pack['mask-v'])
        batch['nox-x-gt'] = pack['nox-x'] * pack['mask-x'] + (1.0 - pack['mask-x'])
        batch['nox-v-pred'] = pred_nox_v * pred_mask_v + (1.0 - pred_mask_v)
        batch['nox-x-pred'] = pred_nox_x * pred_mask_x + (1.0 - pred_mask_x)
        batch['mask-v'] = pred_mask_v
        batch['mask-x'] = pred_mask_x
        batch['rgb-v'] = pack['rgb-v']
        return batch
