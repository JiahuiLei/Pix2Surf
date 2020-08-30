"""
Pix2Surf Multi-View Version Evaluation in multi view protocol
"""

from .modelbase_v2 import ModelBase
from core.models.utils import *
from .pix2surf_mv import Network as MV_Net

import os
import torch
from core.evaluation import eval_warp


class Model(ModelBase):
    def __init__(self, cfg):
        super(Model, self).__init__(cfg)
        self.name = 'pix2surf-mv'
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
            self.load_model(loadpath=load_path, current_model_state='cpu', strict=False)
        elif cfg.MODEL_INIT_PATH != ['None']:
            self.load_model(loadpath=cfg.MODEL_INIT_PATH, strict=False)
        self.to_gpus()
        # config output meaning
        self.output_info_dict = {
            'metric': ['batch-loss', 'reg-v-loss', 'reg-x-loss', 'mask-v-loss', 'mask-x-loss',
                       'sp-loss', 'crr-xyz-loss'],
            'image': ['uni-rgb-v', 'nox-v-gt-uni', 'mask-v'] +
                     ['unwrapped-chart', 'unwrapped-chart-uni', 'learned-chart', 'sp-image-uni'],
            'xls': ['metric-report']
        }

    def _preprocess(self, in_batch):
        return load_multiview_batch(in_batch)

    def _postprocess(self, batch):
        # compute metric in multi thread
        batch = eval_warp(batch, method_name='pix2surf-mv', nox_gt_key='nox-v-gt', nox_pred_key='sp-image')
        # add crr_loss to xls report
        batch['metric-report']['consistency-error'] = [float(i) for i in
                                                       batch['crr-xyz-loss-xls'].detach().cpu().numpy()]
        return batch


class Network(MV_Net):
    def __init__(self):
        super(Network, self).__init__()
        # make eval config
        self.eval_image_res = (240, 320)
        self.eval_image_grid = make_grid(self.eval_image_res)

    def forward(self, pack, is_train=True):
        batch = dict()
        n_batch = pack['nox-v'][0].shape[0]
        n_view = len(pack['rgb-v'])

        code_list = list()
        pred_list, featuremap_list = self.network_dict['seg-net'](pack['rgb-v'], return_code=True)
        for fm in featuremap_list:  # do for each view
            code_list.append(self.network_dict['global-code'](fm).reshape(n_batch, -1, 1).contiguous())
        global_z = torch.max(torch.cat(code_list, 2), dim=2).values.contiguous()

        # prepare gather container
        pred_nox_v_list, pred_nox_x_list, pred_mask_v_list, pred_mask_x_list = [], [], [], []
        pred_xyz_list, pred_uv_list = [], []
        learned_chart_list, unwrapped_chart_list = [], []
        reg_v_loss, reg_x_loss, mask_v_loss, mask_x_loss, sp_loss = 0, 0, 0, 0, 0
        eval_rendered_list = []

        for ii in range(n_view):
            mask_v = pack['mask-v'][ii]
            mask_x = pack['mask-x'][ii]

            # make cnn prediction
            pred = pred_list[ii]
            pred_nox_v = pred[:, :3, :, :]
            pred_nox_x = pred[:, 3:6, :, :]
            pred_score_v = pred[:, 6:8, :, :]
            pred_score_x = pred[:, 8:10, :, :]
            learned_uv = self.sgmd(pred[:, 10:12, :, :])

            # make NOCS-regression branch
            mask1c_v = mask_v[:, 0, :, :].unsqueeze(1).detach()
            mask_v_loss = mask_v_loss + self.cls_criterion(pred_score_v, mask1c_v.squeeze(1).long().detach()) / n_view
            pred_mask_v = torch.argmax(pred_score_v, dim=1, keepdim=True).float()
            mask1c_x = mask_x[:, 0, :, :].unsqueeze(1).detach()
            mask_x_loss = mask_x_loss + self.cls_criterion(pred_score_x, mask1c_x.squeeze(1).long().detach()) / n_view
            pred_mask_x = torch.argmax(pred_score_x, dim=1, keepdim=True).float()
            reg_v_loss = reg_v_loss + self.ml2_criterion(pred_nox_v, pack['nox-v'][ii], mask1c_v, True) / n_view
            reg_x_loss = reg_x_loss + self.ml2_criterion(pred_nox_x, pack['nox-x'][ii], mask1c_x, True) / n_view

            # make mlp prediction
            eachview_z = code_list[ii].squeeze(2)
            latent_dim = eachview_z.shape[1]
            c = torch.cat((eachview_z[:, :latent_dim // 2], global_z[:, latent_dim // 2:]), dim=1)
            queried_uv = query_feature(learned_uv, pack['uv-v'][ii])
            pred_xyz = self.network_dict['mlp'](c, queried_uv, unique_code=True)
            pred_xyz = self.sgmd(pred_xyz)
            sp_loss = sp_loss + self.ml2_criterion(pred_xyz, pack['uv-xyz-v'][ii], pack['uv-mask-v'][ii]) / n_view

            # Do SP evaluation
            _eval_rendered_list = list()
            for bid in range(n_batch):
                # select mask
                _mask = pred_mask_v[bid, ...].reshape(-1)  # H*W
                _learned_uv = learned_uv[bid, ...].reshape(1, 2, -1)  # 1,2,H*W
                _learned_uv = _learned_uv[:, :, _mask > 0]  # 1,2,S
                uv = self.eval_image_grid.cuda().reshape(1, 2, -1)[:, :, _mask > 0].unsqueeze(3)  # 1,2,S,1
                # do Surface Parametrization
                eval_xyz_v = self.network_dict['mlp'](c[bid, ...].unsqueeze(0), _learned_uv.unsqueeze(3),
                                                      unique_code=True)
                eval_xyz_v = self.sgmd(eval_xyz_v)  # 1,3,S,1
                uv[:, 0, :, :] = uv[:, 0, :, :] * mask1c_v.shape[2]
                uv[:, 1, :, :] = uv[:, 1, :, :] * mask1c_v.shape[3]
                uv = uv.long()
                idx = uv[:, 0, :, :] * mask1c_v.shape[3] + uv[:, 1, :, :]  # B,N,1
                idx = idx.permute(0, 2, 1)  # B,1,N
                vis_eval = torch.ones_like(pack['rgb-v'][ii]).float()[bid, ...].unsqueeze(0)
                vis_eval = vis_eval.reshape(1, 3, -1)  # B,3,R*R
                vis_eval = vis_eval.scatter(dim=2, index=idx.repeat(1, 3, 1), src=eval_xyz_v.squeeze(3))
                vis_eval = vis_eval.reshape(1, 3, mask1c_v.shape[2], mask1c_v.shape[3])
                _eval_rendered_list.append(vis_eval)
            eval_rendered = torch.cat(_eval_rendered_list, 0)
            eval_rendered_list.append(eval_rendered)

            # vis unwrapped chart
            unwrapped_chart = self.vis_chart_container.repeat(n_batch, 1, 1, 1).cuda()
            unwrapped_chart = spread_feature(unwrapped_chart, learned_uv, pack['rgb-v'][ii], pack['mask-v'][ii])

            # gather
            pred_nox_v_list.append(pred_nox_v)
            pred_nox_x_list.append(pred_nox_x)
            pred_mask_v_list.append(pred_mask_v)
            pred_mask_x_list.append(pred_mask_x)

            pred_xyz_list.append(pred_xyz)
            pred_uv_list.append(queried_uv)
            unwrapped_chart_list.append(unwrapped_chart)
            learned_chart_list.append(learned_uv.repeat(1, 2, 1, 1)[:, :3, :, :] * pred_mask_x + (1.0 - pred_mask_v))

        # make naive multi-view constrain:
        _p1_list, _p2_list, _m_list = [], [], []
        _uv1_list, _uv2_list = [], []
        for base_view_id in range(len(pack['crr-idx-mtx'])):
            for query_view_id in range(len(pack['crr-idx-mtx'][base_view_id])):
                base_pc = pred_xyz_list[base_view_id]
                query_pc = pred_xyz_list[base_view_id + query_view_id + 1]
                base_uv = pred_uv_list[base_view_id]
                query_uv = pred_uv_list[base_view_id + query_view_id + 1]
                pair_idx = pack['crr-idx-mtx'][base_view_id][query_view_id].squeeze(3)
                paired_pc_from_base_to_query = torch.gather(base_pc.squeeze(3), dim=2,
                                                            index=pair_idx.repeat(1, 3, 1)).unsqueeze(3)
                paired_uv_from_base_to_query = torch.gather(base_uv.squeeze(3), dim=2,
                                                            index=pair_idx.repeat(1, 2, 1)).unsqueeze(3)
                _p1_list.append(paired_pc_from_base_to_query)
                _p2_list.append(query_pc)
                _uv1_list.append(paired_uv_from_base_to_query)
                _uv2_list.append(query_uv)
                _m_list.append(pack['crr-mask-mtx'][base_view_id][query_view_id])

        crr_xyz_loss_each = self.ml2_criterion(torch.cat(_p1_list, dim=2).contiguous(),
                                               torch.cat(_p2_list, dim=2).contiguous(),
                                               torch.cat(_m_list, dim=2).contiguous(),
                                               detach=False, reduce_batch=False)
        crr_xyz_loss = crr_xyz_loss_each.mean()

        crr_uv_loss_each = self.ml2_criterion(torch.cat(_uv1_list, dim=2).contiguous(),
                                              torch.cat(_uv2_list, dim=2).contiguous(),
                                              torch.cat(_m_list, dim=2).contiguous(),
                                              detach=False, reduce_batch=False)
        crr_uv_loss = crr_uv_loss_each.mean()

        # summary
        batch['batch-loss'] = (((reg_v_loss + reg_x_loss) * 0.1 + (mask_v_loss + mask_x_loss) * 0.1) * 0.1 + \
                               sp_loss * 0.9 + crr_xyz_loss * 0.9).unsqueeze(0)  # + crr_uv_loss * 0.1

        batch['reg-v-loss'] = reg_v_loss.detach().unsqueeze(0)
        batch['reg-x-loss'] = reg_x_loss.detach().unsqueeze(0)
        batch['mask-v-loss'] = mask_v_loss.detach().unsqueeze(0)
        batch['mask-x-loss'] = mask_x_loss.detach().unsqueeze(0)
        batch['sp-loss'] = sp_loss.detach().unsqueeze(0)
        batch['crr-xyz-loss'] = crr_xyz_loss.detach().unsqueeze(0)
        batch['crr-xyz-loss-xls'] = crr_xyz_loss_each.detach()

        batch['mask-v'] = torch.cat(pred_mask_v_list, 3)
        batch['mask-x'] = torch.cat(pred_mask_x_list, 3)
        batch['rgb-v'] = pack['rgb-v']
        batch['uni-rgb-v'] = torch.cat(pack['rgb-v'], 3)

        batch['nox-v-gt'] = [p * m + (1.0 - m) for p, m in zip(pack['nox-v'], pack['mask-v'])]
        batch['nox-x-gt'] = [p * m + (1.0 - m) for p, m in zip(pack['nox-x'], pack['mask-x'])]
        batch['nox-v-gt-uni'] = torch.cat([p * m + (1.0 - m) for p, m in zip(pack['nox-v'], pack['mask-v'])], 3)
        batch['nox-x-gt-uni'] = torch.cat([p * m + (1.0 - m) for p, m in zip(pack['nox-x'], pack['mask-x'])], 3)

        batch['sp-image'] = eval_rendered_list
        batch['sp-image-uni'] = torch.cat(eval_rendered_list, 3)

        batch['learned-chart'] = torch.cat(learned_chart_list, 3)
        batch['unwrapped-chart'] = torch.cat(unwrapped_chart_list, 3)
        vis_nsc_uni = unwrapped_chart_list[0]
        for new_scatter in unwrapped_chart_list:
            vis_nsc_uni = torch.max(new_scatter, vis_nsc_uni)
        batch['unwrapped-chart-uni'] = vis_nsc_uni

        return batch
