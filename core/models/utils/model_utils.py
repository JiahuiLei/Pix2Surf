import torch
import numpy as np


def load_multiview_batch(in_batch):
    device = torch.device("cuda")
    nox_v = [item.float().permute(0, 3, 1, 2).to(device) for item in in_batch['nox-v']]  # [0,1]
    nox_x = [item.float().permute(0, 3, 1, 2).to(device) for item in in_batch['nox-x']]  # [0,1]
    rgb_v = [item.float().permute(0, 3, 1, 2).to(device) for item in in_batch['rgb-v']]  # [0,1]
    rgb_x = [item.float().permute(0, 3, 1, 2).to(device) for item in in_batch['rgb-x']]  # [0,1]
    mask_v = [item.float().permute(0, 3, 1, 2).to(device) for item in in_batch['mask-v']]  # 0,1
    mask_x = [item.float().permute(0, 3, 1, 2).to(device) for item in in_batch['mask-x']]  # 0,1

    uv_v = [item.float().permute(0, 2, 1).unsqueeze(3).to(device) for item in in_batch['uv-v']]
    uv_x = [item.float().permute(0, 2, 1).unsqueeze(3).to(device) for item in in_batch['uv-x']]
    uv_mask_v = [item.float().permute(0, 2, 1).unsqueeze(3).to(device) for item in in_batch['uv-mask-v']]
    uv_mask_x = [item.float().permute(0, 2, 1).unsqueeze(3).to(device) for item in in_batch['uv-mask-x']]
    uv_xyz_v = [item.float().permute(0, 2, 1).unsqueeze(3).to(device) for item in in_batch['uv-xyz-v']]
    uv_xyz_x = [item.float().permute(0, 2, 1).unsqueeze(3).to(device) for item in in_batch['uv-xyz-x']]

    crr_idx_mtx = [[ii.long().permute(0, 2, 1).unsqueeze(3).to(device) for ii in item]
                   for item in in_batch['crr-idx-mtx']]
    crr_mask_mtx = [[ii.float().permute(0, 2, 1).unsqueeze(3).to(device) for ii in item]
                    for item in in_batch['crr-mask-mtx']]

    pack = {'rgb-v': rgb_v, 'rgb-x': rgb_x, 'nox-v': nox_v, 'nox-x': nox_x,
            'mask-v': mask_v, 'mask-x': mask_x,
            'uv-v': uv_v, 'uv-x': uv_x,
            'uv-mask-v': uv_mask_v, 'uv-mask-x': uv_mask_x,
            'uv-xyz-v': uv_xyz_v, 'uv-xyz-x': uv_xyz_x,
            'crr-idx-mtx': crr_idx_mtx, 'crr-mask-mtx': crr_mask_mtx}
    return {'to-nn': pack, 'meta-info': in_batch['info']}


def load_singleview_batch(in_batch):
    device = torch.device("cuda")
    nox_v = in_batch['nox-v'].float().permute(0, 3, 1, 2).to(device) / 255.0  # [0,1]
    nox_x = in_batch['nox-x'].float().permute(0, 3, 1, 2).to(device) / 255.0  # [0,1]
    rgb_v = in_batch['rgb-v'].float().permute(0, 3, 1, 2).to(device) / 255.0  # [0,1]
    rgb_x = in_batch['rgb-x'].float().permute(0, 3, 1, 2).to(device) / 255.0  # [0,1]
    mask_v = in_batch['mask-v'].float().permute(0, 3, 1, 2).to(device)  # 0,1
    mask_x = in_batch['mask-x'].float().permute(0, 3, 1, 2).to(device)  # 0,1

    uv_v = in_batch['uv-v'].float().permute(0, 2, 1).unsqueeze(3).to(device)
    uv_x = in_batch['uv-x'].float().permute(0, 2, 1).unsqueeze(3).to(device)
    uv_mask_v = in_batch['uv-mask-v'].float().permute(0, 2, 1).unsqueeze(3).to(device)
    uv_mask_x = in_batch['uv-mask-x'].float().permute(0, 2, 1).unsqueeze(3).to(device)
    uv_xyz_v = in_batch['uv-xyz-v'].float().permute(0, 2, 1).unsqueeze(3).to(device) / 255
    uv_xyz_x = in_batch['uv-xyz-x'].float().permute(0, 2, 1).unsqueeze(3).to(device) / 255
    pack = {'rgb-v': rgb_v, 'rgb-x': rgb_x, 'nox-v': nox_v, 'nox-x': nox_x,
            'mask-v': mask_v, 'mask-x': mask_x,
            'uv-v': uv_v, 'uv-x': uv_x,
            'uv-mask-v': uv_mask_v, 'uv-mask-x': uv_mask_x,
            'uv-xyz-v': uv_xyz_v, 'uv-xyz-x': uv_xyz_x}
    return {'to-nn': pack, 'meta-info': in_batch['info']}


def spread_feature(container, learned_uv, feature, mask1c):
    """
    :param container: B,C,R,R
    :param learned_uv: B,2,H,W
    :param feature: B,C,H,W aligned with latent uv map
    :param mask1c: B,1,H,W used to mask latent uv and feature
    :return: container
    """
    assert float(mask1c.max()) < (1.0 + 1e-9)
    assert container.shape[1] == feature.shape[1]
    c = container.shape[1]
    res = container.shape[2]
    _learned_uv = learned_uv * mask1c.repeat(1, 2, 1, 1)
    _feature = feature * mask1c.repeat(1, c, 1, 1)
    learned_uv = torch.clamp((_learned_uv * res).long(), 0, res - 1)
    learned_uv = learned_uv.reshape(learned_uv.shape[0], 2, -1)
    learned_uv = learned_uv[:, 0, :] * res + learned_uv[:, 1, :]  # B, R*R
    learned_uv = learned_uv.unsqueeze(1).repeat(1, c, 1)  # B,C,R*R
    container = container.reshape(container.shape[0], container.shape[1], -1)
    container = container.scatter(2, learned_uv, _feature.reshape(feature.shape[0], c, -1))
    container = container.reshape(container.shape[0], container.shape[1], res, res)
    return container


def query_feature(feature_map, query_uv):
    """
    query features from feature map
    :param feature_map: B,C,res1,res2
    :param query_uv: B,2,K,1 in [0,1]
    :return B,C,K,1
    """
    assert float(query_uv.max()) < 1 + 1e-9
    assert query_uv.shape[1] == 2
    res1 = feature_map.shape[2]
    res2 = feature_map.shape[3]
    query_index = query_uv.clone()
    query_index[:, 0, ...] = torch.clamp((query_uv[:, 0, ...] * res1).long(), 0, res1 - 1)
    query_index[:, 1, ...] = torch.clamp((query_uv[:, 1, ...] * res2).long(), 0, res2 - 1)
    if query_index.ndimension() > 3:
        index = query_index.squeeze(3)  # B,2,K
    else:
        index = query_index  # B*2*K
    # cvt to 1D index
    index = index[:, 0, :] * feature_map.shape[3] + index[:, 1, :]  # B,K
    index = index.unsqueeze(2).repeat(1, 1, feature_map.shape[1])
    flatten_feature_map = feature_map.reshape(feature_map.shape[0],
                                              feature_map.shape[1], -1).permute(0, 2, 1)  # B,N,C
    query = torch.gather(flatten_feature_map, 1, index.long()).contiguous()  # B,K,C
    query = query.permute(0, 2, 1).unsqueeze(3)
    return query


def make_grid(res, return_np=False):
    dim0 = np.arange(0, res[0]) + 0.5
    dim0 = dim0 / len(dim0)
    dim1 = np.arange(0, res[1]) + 0.5
    dim1 = dim1 / len(dim1)
    col_uv, row_uv = np.meshgrid(dim1, dim0)
    super_uv = np.concatenate((row_uv[..., np.newaxis], col_uv[..., np.newaxis]), 2)  # R,R,2
    super_uv_tensor = torch.from_numpy(super_uv.astype(np.float32))
    if return_np:
        return super_uv_tensor.permute(2, 0, 1).unsqueeze(0), super_uv
    else:
        return super_uv_tensor.permute(2, 0, 1).unsqueeze(0)
