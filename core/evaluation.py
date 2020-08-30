import numpy as np
from sklearn.neighbors import NearestNeighbors
from multiprocessing.dummy import Pool as ThreadPool
import cv2 as cv
from copy import deepcopy


def eval_warp(batch, method_name, nox_gt_key, nox_pred_key):
    """
    Parameters
    ----------
    batch: the batch in post-processing, must be a multi-view like batch
    method_name: string of name e.g 'pix2surf-sv' that will be written to the xls
    nox_gt_key: the name of nox-gt; WARNING! the nocs-map should have white background
    nox_pred_key: see above, for prediction

    Returns: the batch that added the 'metric-report' the report xls
    -------
    """

    # multi thread eval (Initial motivation of multi-thread is for some slow computation like normal)
    n_batch = batch[nox_gt_key][0].shape[0]
    n_view = len(batch[nox_gt_key])

    # make the parameter tuple list for multi-thread
    TASKS, RESULTS = [], []
    id = 0
    for bdx in range(n_batch):
        for vdx in range(n_view):
            arg = [id]
            id += 1
            _nocs_v_gt = batch[nox_gt_key][vdx][bdx].detach().cpu().numpy().transpose(1, 2, 0)
            arg.append(_nocs_v_gt)
            _nox_v_pred = batch[nox_pred_key][vdx][bdx].detach().cpu().numpy().transpose(1, 2, 0)
            arg.append(_nox_v_pred)
            TASKS.append(tuple(arg))
    assert id == n_batch * n_view

    with ThreadPool(max(id, 16)) as pool:
        _results = [pool.apply_async(eval_thread, t) for t in TASKS]
        RESULTS = [r.get() for r in _results]

    ordered_results = []
    for idx in range(id):
        for r in RESULTS:
            if r[0] == idx:
                ordered_results.append(r)
    assert len(ordered_results) == len(RESULTS)

    accuracy_list, correspondence_error_list, discontinuity_score_list = [], [], []
    id = 0
    for bdx in range(n_batch):
        _cd, _corr_l2, _disconti_score = 0, 0, 0
        for vdx in range(n_view):
            r = ordered_results[id]
            id += 1
            # for different viewpoint of each object, average across views
            _cd += r[1] / n_view
            _corr_l2 += r[2] / n_view
            _disconti_score += r[3] / n_view
        accuracy_list.append(_cd)
        correspondence_error_list.append(_corr_l2)
        discontinuity_score_list.append(_disconti_score)

    # make xls
    report_xls = dict()
    if 'metric-report' in batch.keys():
        report_xls = batch['metric-report']
    report_xls[method_name + '-accuracy'] = accuracy_list
    report_xls[method_name + '-2D3D-correspondence'] = correspondence_error_list
    report_xls[method_name + '-discontinuity-score'] = discontinuity_score_list
    batch['metric-report'] = report_xls

    return batch


def eval_thread(bid, nocs_gt, nocs_pred):
    """
    Parameters
    ----------
    bid: batch idx, used to reconstruct a batch in right order
    nocs_gt: numpy,  [H,W,3]
    nocs_pred: numpy, [H,W,3]
    Returns
    -------
    a tuple of returned values elements
    """
    nocs_gt_pts = nocs_gt.reshape(-1, 3)
    mask_gt_pts = np.sum(nocs_gt_pts, axis=1) < 3.0
    nocs_gt_pts = nocs_gt_pts[mask_gt_pts, :]

    nocs_pred_pts = nocs_pred.reshape(-1, 3)
    mask_pred_pts = np.sum(nocs_pred_pts, axis=1) < 3.0
    nocs_pred_pts = nocs_pred_pts[mask_pred_pts, :]

    # based on xyz, find nearest neighbor in each other
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(nocs_pred_pts)
    gt_nn_distance, gt_nn_index_of_pred = neigh.kneighbors(nocs_gt_pts, return_distance=True)
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(nocs_gt_pts)
    pred_nn_distance, pred_nn_index_of_gt = neigh.kneighbors(nocs_pred_pts, return_distance=True)
    gt_nn_index_of_pred = gt_nn_index_of_pred.squeeze(1)
    pred_nn_index_of_gt = pred_nn_index_of_gt.squeeze(1)

    # Compute 2 way Chamfer distance
    cd_dist_gt2pred = np.sum((nocs_gt_pts - nocs_pred_pts[gt_nn_index_of_pred, :]) ** 2, axis=1)
    cd_dist_pred2gt = np.sum((nocs_pred_pts - nocs_gt_pts[pred_nn_index_of_gt, :]) ** 2, axis=1)
    visible_2way_chamfer_distance = cd_dist_gt2pred.mean() + cd_dist_pred2gt.mean()

    # Compute Correspondence error
    mask_gt = (nocs_gt.sum(2) < 3.0).astype(np.float)
    mask_pred = (nocs_pred.sum(2) < 3.0).astype(np.float)
    mask_intersection = mask_gt * mask_pred  # H,W,1
    xyz_dif = np.sum((deepcopy(nocs_gt) - deepcopy(nocs_pred)) ** 2, axis=2)
    xyz_correspondence_distance = (xyz_dif * mask_intersection).sum() / (mask_intersection.sum() + 1e-5)

    # Compute Discontinuity score
    pair_dist_gt = compute_pixel_neighbor_diff(deepcopy(nocs_gt))
    pair_dist_pred = compute_pixel_neighbor_diff(deepcopy(nocs_pred))
    k1, k2 = 30, 20
    th = 0.05
    gt_hist_normalized, gt_count = pair_dist2hist(pair_dist_gt, k1, k2, th)
    pred_hist_normalized, pred_count = pair_dist2hist(pair_dist_pred, k1, k2, th)
    large_dist_pairs_conv = np.sum((gt_count[k1:] / (gt_count[k1:].sum() + 1e-5)) * \
                                   (pred_count[k1:] / (pred_count[k1:].sum() + 1e-5)))

    # Cross-View Consistency Error is computed outside here, in the network
    return (
        bid,
        visible_2way_chamfer_distance,  # accuracy
        xyz_correspondence_distance,  # correspondence
        large_dist_pairs_conv,  # continuity
    )


def compute_pixel_neighbor_diff(nocs_map):
    mask = (np.sum(nocs_map, axis=2) < 3.0).astype(np.float)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    mask_smaller = cv.erode(mask, kernel, iterations=1)

    d_r = deepcopy(nocs_map)
    d_r[:, :-1, :] -= deepcopy(nocs_map)[:, 1:, :]
    d_r = np.sqrt(np.sum(d_r ** 2, axis=2)) * mask_smaller

    d_l = deepcopy(nocs_map)
    d_l[:, 1:, :] -= deepcopy(nocs_map)[:, :-1, :]
    d_l = np.sqrt(np.sum(d_l ** 2, axis=2)) * mask_smaller

    d_d = deepcopy(nocs_map)
    d_d[:-1, :, :] -= deepcopy(nocs_map)[1:, :, :]
    d_d = np.sqrt(np.sum(d_d ** 2, axis=2)) * mask_smaller

    d_u = deepcopy(nocs_map)
    d_u[1:, :, :] -= deepcopy(nocs_map)[:-1, :, :]
    d_u = np.sqrt(np.sum(d_u ** 2, axis=2)) * mask_smaller

    select_mask = mask_smaller.reshape(-1) > 0.5
    dr = d_r.reshape(-1)[select_mask]
    dl = d_l.reshape(-1)[select_mask]
    du = d_u.reshape(-1)[select_mask]
    dd = d_d.reshape(-1)[select_mask]
    distance = np.concatenate((dr, dl, du, dd), 0)

    return distance


def pair_dist2hist(pair_dist, k1=20, k2=20, th=0.04):
    bin1 = np.linspace(0, th, k1)
    bin2 = np.linspace(th, np.sqrt(3), k2)
    bin = np.concatenate((bin1, bin2, np.array([np.sqrt(3)])), 0)
    conunt_list = []
    for idx in range(k1 + k2):
        mask = (pair_dist >= bin[idx]).astype(np.float) * (pair_dist < bin[idx + 1]).astype(np.float)
        conunt_list.append(mask.sum())
    count = np.array(conunt_list)
    hist = count / (len(pair_dist) + 1e-5)
    return hist, count
