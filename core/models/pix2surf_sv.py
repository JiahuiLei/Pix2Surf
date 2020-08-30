"""
Pix2Surf Single View Single Chart Version
"""

from .modelbase_v2 import ModelBase
from .modelbase_v2 import Network as NetBase
from core.models.utils import *
from core.net_bank.pix2surf_cnn import SegNet
from core.net_bank.mlp import NOCS_AMP_MLP
from core.net_bank.loss import MaskL2Loss
import os
import torch
from torch import nn


class Model(ModelBase):
    def __init__(self, cfg):
        super(Model, self).__init__(cfg)
        self.name = 'pix2surf-sv'
        self.cfg = cfg
        # register key component
        self.network = Network()
        self.optimizer = torch.optim.Adam(params=self.network.parameters(), lr=self.lr,
                                          betas=(self.cfg.ADAM_BETA1, self.cfg.ADAM_BETA2))
        # initialize models
        self.resume = cfg.RESUME
        if self.resume:
            self.resume_id = cfg.RESUME_EPOCH_ID
            load_path = os.path.join(cfg.ROOT_DIR, 'log', cfg.LOG_DIR, 'model',
                                     'epoch_%d' % cfg.RESUME_EPOCH_ID + '.model')
            self.load_model(loadpath=load_path, current_model_state='cpu')
        elif cfg.MODEL_INIT_PATH != ['None']:
            self.load_model(loadpath=cfg.MODEL_INIT_PATH, strict=False)
        self.to_gpus()
        # config output meaning
        self.output_info_dict = {
            'metric': ['batch-loss', 'reg-v-loss', 'reg-x-loss', 'mask-v-loss', 'mask-x-loss', 'mlp-v-loss'],
            'image': ['rgb-v', 'nox-v-gt', 'mask-v', 'tex', 'gim', 'learned-chart', 'sp-image'],
            # tex(ture) is unwrapped chart with color visualization
            # gim (Geometry Image) is unwrapped chart with NOCS XYZ visualization
            # learned-chart is the 2 channel output color coded visualization in image space
        }

    def _preprocess(self, in_batch):
        return load_singleview_batch(in_batch)


class Network(NetBase):
    def __init__(self):
        super(Network, self).__init__()
        net_dict = {
            'seg-net': SegNet(out_channels=10, additional=2),
            'global-code': nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=0, stride=1),
                nn.BatchNorm2d(512),
                nn.ELU(),
                nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=0, stride=1),
                nn.MaxPool2d(kernel_size=(3, 6))
            ),
            'mlp': NOCS_AMP_MLP(latent_dim=1024, amp_dim=256, p_in=2, c_out=3)
        }
        self.network_dict = nn.ModuleDict(net_dict)
        self.sgmd = nn.Sigmoid()
        # loss
        self.cls_criterion = nn.CrossEntropyLoss()  # not masked, for all pixel
        self.ml2_criterion = MaskL2Loss()  # masked for nocs regression
        # visualization
        self.vis_chart_res = 128
        self.vis_chart_container = torch.zeros(1, 3, self.vis_chart_res, self.vis_chart_res)

    def forward(self, pack, is_train=True):
        batch = dict()
        n_batch = pack['nox-v'].shape[0]

        # make cnn prediction
        pred, fm = self.network_dict['seg-net'](pack['rgb-v'], return_code=True)
        pred_nox_v = pred[:, :3, :, :]
        pred_nox_x = pred[:, 3:6, :, :]
        pred_score_v = pred[:, 6:8, :, :]
        pred_score_x = pred[:, 8:10, :, :]
        learned_uv = self.sgmd(pred[:, 10:12, :, :])

        # make NOCS-regression branch
        mask1c_v = pack['mask-v'][:, 0, :, :].unsqueeze(1).detach()
        mask_v_loss = self.cls_criterion(pred_score_v, mask1c_v.squeeze(1).long().detach())
        pred_mask_v = torch.argmax(pred_score_v, dim=1, keepdim=True).float()
        mask1c_x = pack['mask-x'][:, 0, :, :].unsqueeze(1).detach()
        mask_x_loss = self.cls_criterion(pred_score_x, mask1c_x.squeeze(1).long().detach())
        pred_mask_x = torch.argmax(pred_score_x, dim=1, keepdim=True).float()
        reg_v_loss = self.ml2_criterion(pred_nox_v, pack['nox-v'], mask1c_v, True)
        reg_x_loss = self.ml2_criterion(pred_nox_x, pack['nox-x'], mask1c_x, True)

        # make mlp prediction
        z = self.network_dict['global-code'](fm).reshape(n_batch, -1).contiguous()
        queried_uv_v = query_feature(learned_uv, pack['uv-v'])
        pred_xyz_v = self.network_dict['mlp'](z, queried_uv_v, unique_code=True)
        pred_xyz_v = self.sgmd(pred_xyz_v)
        sp_v_loss = self.ml2_criterion(pred_xyz_v, pack['uv-xyz-v'], pack['uv-mask-v'])

        # vis
        tex = self.vis_chart_container.repeat(n_batch, 1, 1, 1).cuda()
        tex = spread_feature(tex, learned_uv, pack['rgb-v'], pack['mask-v'])
        gim = self.vis_chart_container.repeat(n_batch, 1, 1, 1).cuda()
        gim = spread_feature(gim, queried_uv_v, pred_xyz_v, pack['uv-mask-v'])

        vis_sampled_xyz = torch.ones_like(pack['rgb-v']).float()
        uv = pack['uv-v']
        uv[:, 0, :, :] = uv[:, 0, :, :] * mask1c_v.shape[2]
        uv[:, 1, :, :] = uv[:, 1, :, :] * mask1c_v.shape[3]
        uv = uv.long()
        idx = uv[:, 0, :, :] * mask1c_v.shape[3] + uv[:, 1, :, :]  # B,N,1
        idx = idx.permute(0, 2, 1)  # B,1,N
        vis_sampled_xyz = vis_sampled_xyz.reshape(n_batch, 3, -1)  # B,3,R*R
        vis_sampled_xyz = vis_sampled_xyz.scatter(dim=2, index=idx.repeat(1, 3, 1), src=pred_xyz_v.squeeze(3))
        vis_sampled_xyz = vis_sampled_xyz.reshape(n_batch, 3, mask1c_v.shape[2], mask1c_v.shape[3])

        # summary
        batch['batch-loss'] = (((reg_v_loss + reg_x_loss) * 0.3 + (mask_v_loss + mask_x_loss) * 0.7) * 0.1 + \
                               sp_v_loss * 0.9).unsqueeze(0)
        batch['reg-v-loss'] = reg_v_loss.detach().unsqueeze(0)
        batch['reg-x-loss'] = reg_x_loss.detach().unsqueeze(0)
        batch['mask-v-loss'] = mask_v_loss.detach().unsqueeze(0)
        batch['mask-x-loss'] = mask_x_loss.detach().unsqueeze(0)
        batch['mlp-v-loss'] = sp_v_loss.detach().unsqueeze(0)

        batch['nox-v-gt'] = pack['nox-v'] * pack['mask-v'] + (1.0 - pack['mask-v'])
        batch['nox-x-gt'] = pack['nox-x'] * pack['mask-x'] + (1.0 - pack['mask-x'])
        batch['mask-v'] = pred_mask_v
        batch['mask-x'] = pred_mask_x
        batch['rgb-v'] = pack['rgb-v']

        batch['learned-chart'] = learned_uv.repeat(1, 2, 1, 1)[:, :3, :, :] * pred_mask_v + (1.0 - pred_mask_v)
        batch['tex'] = tex.detach()
        batch['gim'] = gim.detach()
        batch['sp-image'] = vis_sampled_xyz

        return batch
