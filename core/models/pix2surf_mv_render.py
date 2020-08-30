"""
Pix2Surf Multi-View Version Post-processing (generate geometry image and texture map)
"""

from .modelbase_v2 import ModelBase
from .pix2surf_mv import Network as MV_Net
from core.models.utils import *

import os
import torch


class Model(ModelBase):
    def __init__(self, cfg):
        super(Model, self).__init__(cfg)
        self.name = 'pix2surf-mv'
        self.cfg = cfg
        # register key component
        self.network = Network(cfg)
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
            'metric': ['batch-loss'],
            'image': ['rgb-v', 'uni-rgb-v', 'nox-v-gt'] +
                     ['unwrapped-chart', 'learned-chart', 'sp-image', 'sp-image-uni'] +
                     ['GIM', 'GIM-uni', 'TEX', 'TEX-uni', ],
            'obj': [('GIM-uni', 'TEX-uni'), ('GIM', 'TEX')]
            # Here the final output is:
            # GIM (Geometry image) for each view, the value at each pixel is the XYZ and is in UV space
            # TEX (Texture map) align with the GIM
            # These can be easily visualized by the tk3dv toolbox: https://github.com/drsrinathsridhar/tk3dv
        }

    def _preprocess(self, in_batch):
        return load_multiview_batch(in_batch)


class Network(MV_Net):
    def __init__(self, cfg):
        super(Network, self).__init__()
        # make eval config
        self.eval_image_res = (240, 320)
        self.eval_image_grid = make_grid(self.eval_image_res)

        # for render (post-processing)
        self.image_resize_factor = 4
        self.final_gim_res = 512
        self.mask_container_res = 128

        self.mask_container_low_res = torch.zeros(1, 1, self.mask_container_res,
                                                  self.mask_container_res)  # use to crop the gim

        self.final_gim = torch.ones(1, 3, self.final_gim_res, self.final_gim_res)  # final gim
        self.final_texture = torch.ones(1, 3, self.final_gim_res, self.final_gim_res)  # final texture map

        self.full_gim_query_grid, self.full_gim_query_grid_np = make_grid(
            (self.final_gim_res, self.final_gim_res), return_np=True)  # used for query geometry image

        # render param
        self.xyz_texture_inter_nk = 4
        self.mask_opti_th = float(cfg.RENDER_MASK_TH)
        self.image_mask_opti_nK = int(cfg.RENDER_IMAGE_MASK_NK)

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
        learned_chart_list, unwrapped_chart_list = [], []
        reg_v_loss, reg_x_loss, mask_v_loss, mask_x_loss, sp_loss = 0, 0, 0, 0, 0
        eval_rendered_list = []
        GIM, TEX, MASK = [], [], []

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

            ####################################### process the final output ###########################################
            final_texture_list, final_mask_list = [], []  # used to collect for each sample
            full_gim = self.generate_full_gim(code=c, split=4096)  # generate full Geometry Image by querying the SP MLP
            for bid in range(n_batch):
                image_learned_uv = learned_uv[bid, ...].permute(1, 2, 0).detach().cpu().numpy()  # H,W,2
                image_mask = pred_mask_v[bid, ...].squeeze(0).detach().cpu().numpy()  # H,W
                image_rgb = pack['rgb-v'][ii][bid, ...].permute(1, 2, 0).detach().cpu().numpy()  # H,W,3
                sp_image_np = eval_rendered[bid, ...].permute(1, 2, 0).detach().cpu().numpy()  # H,W,3
                full_gim_np = full_gim[bid, ...].permute(1, 2, 0).detach().cpu().numpy()  # R,R,3

                image_mask = optimize_image_mask(image_mask, sp_image_np, nK=self.image_mask_opti_nK,
                                                 th=self.mask_opti_th)
                final_mask_np = generate_final_mask(image_learned_uv, image_mask,
                                                    self.image_resize_factor, self.mask_container_low_res,
                                                    self.final_gim)  # R,R
                # Generate Texture
                final_texture_xyz_np, opti_final_mask_np = generate_texture(sp_image_np, full_gim_np,
                                                                            image_rgb, image_mask,
                                                                            final_mask_np,
                                                                            nK=self.xyz_texture_inter_nk,
                                                                            th=self.mask_opti_th,
                                                                            final_res=self.final_gim_res)
                # gather
                final_mask = torch.from_numpy(opti_final_mask_np).unsqueeze(0).unsqueeze(0).cuda().float()
                final_mask_list.append(final_mask)
                final_texture = torch.from_numpy(final_texture_xyz_np).permute(2, 0, 1).unsqueeze(0).cuda().float()
                final_texture_list.append(final_texture)
            _final_mask = torch.cat(final_mask_list, 0)
            _final_texture = torch.cat(final_texture_list, 0)
            _final_gim = full_gim * _final_mask + (1.0 - _final_mask)
            GIM.append(_final_gim)
            TEX.append(_final_texture)
            MASK.append(_final_mask)
            ############################################# process end ##################################################

            # gather
            pred_nox_v_list.append(pred_nox_v)
            pred_nox_x_list.append(pred_nox_x)
            pred_mask_v_list.append(pred_mask_v)
            pred_mask_x_list.append(pred_mask_x)

            unwrapped_chart_list.append(unwrapped_chart)
            learned_chart_list.append(learned_uv.repeat(1, 2, 1, 1)[:, :3, :, :] * pred_mask_x + (1.0 - pred_mask_v))

        # summary
        batch['batch-loss'] = torch.zeros(1, requires_grad=True).unsqueeze(0)

        batch['nox-v-gt'] = torch.cat([p * m + (1.0 - m) for p, m in zip(pack['nox-v'], pack['mask-v'])], 3)
        batch['rgb-v'] = pack['rgb-v']
        batch['uni-rgb-v'] = torch.cat(pack['rgb-v'], 3)
        batch['unwrapped-chart'] = torch.cat(unwrapped_chart_list, 3)
        vis_nsc_uni = unwrapped_chart_list[0]
        for new_scatter in unwrapped_chart_list:
            vis_nsc_uni = torch.max(new_scatter, vis_nsc_uni)
        batch['unwrapped-chart-uni'] = vis_nsc_uni
        batch['learned-chart'] = torch.cat(learned_chart_list, 3)

        # eval content
        batch['sp-image'] = eval_rendered_list
        batch['sp-image-uni'] = torch.cat(eval_rendered_list, 3)

        batch['GIM-uni'] = torch.cat(GIM, 3)
        batch['GIM'] = GIM
        batch['TEX-uni'] = torch.cat(TEX, 3)
        batch['TEX'] = TEX
        batch['MASK-uni'] = torch.cat(MASK, 3)
        batch['MASK'] = MASK

        return batch

    def generate_full_gim(self, code, split=4096):
        with torch.no_grad():
            query_grid = self.full_gim_query_grid.reshape(2, -1).cuda()  # 2,R*R
            query_grid = query_grid.unsqueeze(0).unsqueeze(3).repeat(code.shape[0], 1, 1, 1)  # B,2,R*R,1
            while split > query_grid.shape[2]:
                split = split // 2
            assert query_grid.shape[3] == 1
            count = query_grid.shape[2] // split
            buffer = list()
            for idx in range(count):
                input_query_grid = query_grid[:, :, idx * split:(idx + 1) * split, :]
                buffer.append(self.network_dict['mlp'](code, input_query_grid, unique_code=True))
            if count * split != query_grid.shape[2]:
                input_query_grid = query_grid[:, :, count * split:, :]
                buffer.append(self.network_dict['mlp'](code, input_query_grid, unique_code=True))
            queried_xyz = torch.cat(buffer, dim=2)
            assert queried_xyz.shape[2] == query_grid.shape[2]
            queried_xyz = self.sgmd(queried_xyz)
            queried_xyz = queried_xyz.reshape(code.shape[0], 3, self.final_gim_res, self.final_gim_res)
        return queried_xyz.detach().float()
