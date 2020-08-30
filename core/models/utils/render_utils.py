import torch
import cv2 as cv
import numpy as np
from sklearn.neighbors import NearestNeighbors
from .model_utils import spread_feature

def optimize_image_mask(image_mask, sp_image, nK=4, th=1e-2):
    mask_pts = image_mask.reshape(-1)
    xyz_pts = sp_image.reshape(-1, 3)
    xyz_pts = xyz_pts[mask_pts > 0.5, :]
    Neighbors = NearestNeighbors(n_neighbors=nK + 1, algorithm='kd_tree').fit(xyz_pts)
    nn_dist, nn_idx = Neighbors.kneighbors(xyz_pts)  # N,nK
    nn_dist = nn_dist[:, 1:]
    valid = (np.sum((nn_dist < th).astype(np.float), axis=1) == nK).astype(np.float)
    optimized_mask = image_mask.reshape(-1)
    optimized_mask[mask_pts > 0.5] = valid
    optimized_mask = optimized_mask.reshape(image_mask.shape[0], image_mask.shape[1])
    return optimized_mask


def generate_final_mask(image_learned_uv, image_mask,
                        image_resize_factor, mask_container_low_res, final_gim):
    """
    Post Process Algorithm to generate mask of the unwrapped chart
    Parameters
    ----------
    image_learned_uv: [H,W,2]
    image_mask: [H,W]
    image_resize_factor: float
    mask_container_low_res: a predefined tensor with intermediate low resolution
    final_gim: a predefined tensor with target high resolution
    """
    # resize (larger) rgb and uv with Bi-linear up-sampling
    resized_uv = cv.resize(image_learned_uv, dsize=(image_resize_factor * image_learned_uv.shape[0],
                                                    image_resize_factor * image_learned_uv.shape[1]),
                           interpolation=cv.INTER_LINEAR)
    resized_mask = cv.resize(image_mask, dsize=(image_resize_factor * image_learned_uv.shape[0],
                                                image_resize_factor * image_learned_uv.shape[1]),
                             interpolation=cv.INTER_LINEAR)
    resized_mask = (resized_mask > 0.5).astype(np.float)
    # use gradient to remove the edge
    discontinuous_mask_u = cv.Laplacian(image_learned_uv[..., 0], ddepth=cv.CV_32F)  # small gradient map
    discontinuous_mask_v = cv.Laplacian(image_learned_uv[..., 1], ddepth=cv.CV_32F)  # small gradient map
    # use the max and min in latent u and v to find the threshhold
    u_max = (image_learned_uv[..., 0] * image_mask).max()
    v_max = (image_learned_uv[..., 1] * image_mask).max()
    u_min = (image_learned_uv[..., 0] * image_mask + (1.0 - image_mask)).min()
    v_min = (image_learned_uv[..., 1] * image_mask + (1.0 - image_mask)).min()
    u_th = (u_max - u_min) / 30
    v_th = (v_max - v_min) / 30
    discontinuous_mask_u = (discontinuous_mask_u > u_th).astype(np.float) * image_mask
    discontinuous_mask_v = (discontinuous_mask_v > v_th).astype(np.float) * image_mask
    discontinuous_mask = ((discontinuous_mask_u + discontinuous_mask_v) > 0).astype(np.float)
    # use the mask to remove the boundary
    boundary_recovery_mask = (cv.Laplacian(image_mask, ddepth=cv.CV_32F) > 0.01).astype(np.float)
    discontinuous_mask = discontinuous_mask * (1.0 - boundary_recovery_mask)
    resized_discontinuous_mask = cv.resize(discontinuous_mask,
                                           dsize=(image_resize_factor * image_learned_uv.shape[0],
                                                  image_resize_factor * image_learned_uv.shape[1]),
                                           interpolation=cv.INTER_NEAREST)
    # make the small mask & texture
    high_res_mask = torch.from_numpy(resized_mask * (1.0 - resized_discontinuous_mask)) \
        .unsqueeze(0).unsqueeze(0).cuda().float()  # 1,1,R,R
    high_res_uv = torch.from_numpy(resized_uv).permute(2, 0, 1).unsqueeze(0).cuda().float()
    low_res_mask = mask_container_low_res.cuda()
    low_res_mask = spread_feature(low_res_mask, high_res_uv, high_res_mask, high_res_mask)
    # use close to remove the holes in small mask and then resize
    low_res_mask_closed = low_res_mask.detach().cpu().squeeze(0).squeeze(0).numpy()  # R,R
    close_k_size = int(final_gim.shape[2] / 100)
    close_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (close_k_size, close_k_size))
    final_mask_np = cv.resize(low_res_mask_closed, dsize=(final_gim.shape[2],
                                                          final_gim.shape[2]),
                              interpolation=cv.INTER_NEAREST)  # R,R,3
    final_mask_np = (final_mask_np > 0).astype(np.float)
    final_mask_np = cv.morphologyEx(final_mask_np, cv.MORPH_OPEN, close_kernel)
    return final_mask_np


def generate_texture(sp_image, full_gim, image_rgb, image_mask, final_mask_np, final_res, nK=4, th=1e-2):
    # prepare root and query points form the image and from the high-res chart
    root_xyz_np = sp_image.reshape(-1, 3)  # H*W,3
    root_rgb_np = image_rgb.reshape(-1, 3)  # H*W,3
    _image_mask = image_mask.reshape(-1)  # H*W
    root_xyz_np = root_xyz_np[_image_mask > 0.5, :]  # M,2 [0,1]
    root_rgb_np = root_rgb_np[_image_mask > 0.5, :]  # M,3 [0,1]
    query_xyz_np = full_gim.reshape(-1, 3)  # R*R,3
    _final_mask_np = final_mask_np.reshape(-1)  # R*R
    query_xyz_np = query_xyz_np[_final_mask_np > 0.5, :]  # N,3 [0,1]
    # finding nearest root pixel points
    Neighbors = NearestNeighbors(n_neighbors=nK, algorithm='kd_tree').fit(root_xyz_np)
    nn_dist, nn_idx = Neighbors.kneighbors(query_xyz_np)  # N,nK
    # optimize the gim mask
    valid = (nn_dist[:, 0] < th).astype(np.float)
    optimized_final_mask_np = final_mask_np.reshape(-1).copy()
    optimized_final_mask_np[_final_mask_np > 0.5] = valid
    optimized_final_mask_np = optimized_final_mask_np.reshape(final_mask_np.shape[0], final_mask_np.shape[1])
    # do interpolation based on chart distance
    interpolation_weight = nn_dist.copy()
    interpolation_weight = 1 - interpolation_weight / np.sum(interpolation_weight, 1, keepdims=True)
    interpolation_weight = interpolation_weight / np.sum(interpolation_weight, 1, keepdims=True)
    query_rgb_np = np.zeros((query_xyz_np.shape[0], 3))
    for kdx in range(nK):
        nn_color = root_rgb_np[nn_idx[:, kdx], :]
        query_rgb_np += nn_color * interpolation_weight[:, kdx][..., np.newaxis]
    final_texture_np = np.ones((final_res ** 2, 3))
    final_texture_np[_final_mask_np > 0.5, :] = query_rgb_np
    final_texture_np = final_texture_np.reshape(final_res, final_res, 3)
    return final_texture_np, optimized_final_mask_np