"""
multi-view dataset, load multiple NOCS maps of each object
find correspondence between different viewpoint of the same object
"""

import torch.utils.data as data
import os
import numpy as np
import cv2 as cv
import json
from random import sample
from sklearn.neighbors import NearestNeighbors


class Dataset(data.Dataset):

    def __init__(self, cfg, mode):

        super().__init__()
        self.mode = mode
        self.dataset_root = os.path.join(cfg.ROOT_DIR, 'resource', 'data', cfg.DATASET_ROOT)
        # each phase has separate dir ends with phase name: e.g. xxxx_train xxxx_test
        modes_name = os.listdir(self.dataset_root)
        self.split_dir_name = None
        for name in modes_name:
            if name.lower().endswith(mode):
                self.split_dir_name = name
        if self.split_dir_name is None:
            raise ValueError('No corresponding phase data')
        self.cates_list = cfg.DATASET_CATES

        # build or directly index
        index_fn = None
        for f_index in cfg.DATASET_INDEX:
            if f_index.endswith(self.mode + '.json'):
                index_fn = f_index
        if index_fn is None:
            self.index = self.build_index(check=True)
        else:
            with open(index_fn, 'r') as filehandle:
                self.index = json.load(filehandle)

        # custom config
        self.shape = (320, 240)
        self.n_view = 5
        self.supervision_cap = 4096  # how many pixel in nocs map foreground will be sampled to trian SP branch

        self.index = self.group_index(self.index)

        # use the proportion configuration to reduce the dataset for development
        if cfg.PROPORTION < 1.0:
            p = cfg.PROPORTION
            all_num = len(self.index)
            self.index = self.index[:int(all_num * p - 1)]

        # if use noise background augmentation
        self.add_noise_flag = cfg.ADD_BACKGROUND_NOISE

        return

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        obj_list = self.index[idx]
        if self.mode == 'train':
            sampled_list = sample(obj_list, self.n_view)
        else:
            delta = int(len(obj_list) / self.n_view)
            assert delta >= 1
            sampled_list = [obj_list[i * delta] for i in range(self.n_view)]
        batch = dict()
        data_list = list()
        views_id = ''
        for data in sampled_list:
            views_id += data['id']
            views_id += '-'
        meta_info_reduced = {
            'cate': obj_list[0]['cate'],
            'object': obj_list[0]['object'],
            'id': views_id,
            'phase': obj_list[0]['phase']
        }
        for meta_info in sampled_list:
            data = self.grab_atom_data(meta_info)
            data_list.append(data)
        for k in data.keys():
            batch[k] = [item[k] for item in data_list]
        batch['info'] = meta_info_reduced  # meta info

        # Find correspondence
        crr_idx_mtx, crr_mask_mtx, crr_min_d_mtx = list(), list(), list()
        pair_count = 0
        for view_id in range(self.n_view - 1):
            idx_list, mask_list, mind_list = self.find_correspondence_list(
                query_pc_list=batch['uv-xyz-v'][view_id + 1:], query_mask_list=batch['uv-mask-v'][view_id + 1:],
                base_pc=batch['uv-xyz-v'][view_id], base_mask=batch['uv-mask-v'][view_id]
            )
            crr_idx_mtx.append(idx_list)
            crr_mask_mtx.append(mask_list)
            crr_min_d_mtx.append(mind_list)
            for mask in mask_list:
                pair_count += mask.sum()
        ave_crr_per_view = int(pair_count / self.n_view)
        batch['crr-idx-mtx'] = crr_idx_mtx
        batch['crr-mask-mtx'] = crr_mask_mtx

        return batch

    def find_correspondence_list(self, query_pc_list, base_pc, query_mask_list, base_mask, th=1e-3):
        """
        For each pc in query_pc_list, find correspondence point in base pc,
        if no correspondent point in base, mask this position with 0 in mask
        """
        q_pc_list = [query_view[query_mask.squeeze() > 0, :] for
                     query_view, query_mask in zip(query_pc_list, query_mask_list)]
        b_pc = base_pc[base_mask.squeeze() > 0, :]
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(b_pc)
        index_list, mask_list, mind_list = list(), list(), list()
        for q_pc in q_pc_list:
            assert q_pc.shape[1] == b_pc.shape[1]
            distance, indices = neigh.kneighbors(q_pc, return_distance=True)
            _min_d = distance.ravel()
            idx = indices.ravel()
            _mask = (_min_d < th).astype(np.float)
            index = np.zeros((self.supervision_cap, 1))
            mask = np.zeros((self.supervision_cap, 1)).astype(np.float)
            min_d = np.zeros((self.supervision_cap, 1)).astype(np.float)
            index[:len(q_pc), 0] = idx
            mask[:len(q_pc), 0] = _mask
            min_d[:len(q_pc), 0] = _min_d
            index_list.append(index.astype(np.int))
            mask_list.append(mask.astype(np.float))
            mind_list.append(min_d.astype(np.float))
        return index_list, mask_list, mind_list

    def grab_atom_data(self, meta_info):
        # get direct rgb, nox, mask from NOCS dataset
        v_rgb_w_bg = cv.imread(os.path.join(self.dataset_root, meta_info['view0']))
        x_rgb_x_bg = cv.imread(os.path.join(self.dataset_root, meta_info['view1']))
        v_nocs_x_bg = cv.imread(os.path.join(self.dataset_root, meta_info['nox0']))
        x_nocs_x_bg = cv.imread(os.path.join(self.dataset_root, meta_info['nox1']))
        v_rgb_w_bg = cv.resize(v_rgb_w_bg, self.shape, interpolation=cv.INTER_NEAREST)
        x_rgb_x_bg = cv.resize(x_rgb_x_bg, self.shape, interpolation=cv.INTER_NEAREST)
        v_nocs_x_bg = cv.resize(v_nocs_x_bg, self.shape, interpolation=cv.INTER_NEAREST)
        x_nocs_x_bg = cv.resize(x_nocs_x_bg, self.shape, interpolation=cv.INTER_NEAREST)
        mask_v = (np.sum(v_nocs_x_bg.astype(float), axis=2, keepdims=True) < 255 * 3).astype(np.float)
        mask_x = (np.sum(x_nocs_x_bg.astype(float), axis=2, keepdims=True) < 255 * 3).astype(np.float)
        if self.add_noise_flag:
            # add noise to background
            if self.mode == 'train':
                # have some possibility to have the noise
                seed = np.random.rand()
                if seed > 0.5:
                    noise = np.random.rand(self.shape[1], self.shape[0], 3).astype(np.float32)
                    v_rgb_w_bg = v_rgb_w_bg * mask_v + noise * (1.0 - mask_v) * 255
        v_rgb_w_bg = v_rgb_w_bg.astype(np.float32)
        # sample query position inside mask
        sample_uv_v, sample_mask_v, sample_xyz_v = self.get_uv_mask_xyz(v_nocs_x_bg)
        sample_uv_x, sample_mask_x, sample_xyz_x = self.get_uv_mask_xyz(x_nocs_x_bg)
        # prepare meta info
        meta_info_reduced = {
            'cate': meta_info['cate'],
            'object': meta_info['object'],
            'id': meta_info['id'],
            'phase': meta_info['phase']
        }
        return {
            'rgb-v': v_rgb_w_bg / 255.0,
            'rgb-x': x_rgb_x_bg / 255.0,
            'nox-v': v_nocs_x_bg / 255.0,
            'nox-x': x_nocs_x_bg / 255.0,
            'mask-v': mask_v,
            'mask-x': mask_x,
            'uv-v': sample_uv_v,
            'uv-x': sample_uv_x,
            'uv-mask-v': sample_mask_v,
            'uv-mask-x': sample_mask_x,
            'uv-xyz-v': sample_xyz_v / 255.0,
            'uv-xyz-x': sample_xyz_x / 255.0,
            'pose': meta_info['pose'],
            'info': meta_info_reduced  # meta info
        }

    def get_uv_mask_xyz(self, nocs_map):
        """
        This uv is on the whole resized image
        """
        # get image space uv grid
        crop_param = ((0, nocs_map.shape[0]), (0, nocs_map.shape[1]))
        u = np.arange(0, crop_param[0][1] - crop_param[0][0]) + 0.5
        u = u / len(u)
        v = np.arange(0, crop_param[1][1] - crop_param[1][0]) + 0.5
        v = v / len(v)
        col_uv, row_uv = np.meshgrid(v, u)
        uv_map = np.concatenate((row_uv[..., np.newaxis], col_uv[..., np.newaxis]), 2)

        nocs_croped_v = nocs_map[crop_param[0][0]:crop_param[0][1], crop_param[1][0]: crop_param[1][1], :]  # not used
        uv_list = uv_map.reshape(-1, 2)  # [K,2]
        nocs_list_v = nocs_croped_v.reshape(-1, 3)  # [K,3]
        # mask the background
        mask = np.sum(nocs_list_v, axis=1) < 255 * 3
        uv_list = uv_list[mask, :]
        nocs_list_v = nocs_list_v[mask, :]
        #  sample
        if uv_list.shape[0] >= self.supervision_cap:
            sample_index = np.random.randint(low=0, high=max(uv_list.shape[0] - 1, 1), size=self.supervision_cap)
            uv_list = uv_list[sample_index, :]
            nocs_list_v = nocs_list_v[sample_index, :]
        # safely padding
        uv = np.zeros((self.supervision_cap, 2))
        mask = np.zeros((self.supervision_cap, 1))
        xyz_v = np.zeros((self.supervision_cap, 3))
        # fill value
        uv[0:uv_list.shape[0], :] = uv_list
        mask[0:uv_list.shape[0], :] = 1
        xyz_v[0:uv_list.shape[0], :] = nocs_list_v
        return uv.astype(np.float), mask.astype(np.float), xyz_v.astype(np.float)

    def build_index(self, check=False):
        """
        make index for each data, to make sure each data sample is usable
        """
        index_list = list()
        root = os.path.join(self.dataset_root, self.split_dir_name)
        for cate in self.cates_list:
            cate_root = os.path.join(root, cate)
            models_list = os.listdir(cate_root)
            for obj in models_list:
                obj_root = os.path.join(cate_root, obj)
                files_list = os.listdir(obj_root)
                id_set = set(id.split('_')[1] for id in files_list)
                for id in id_set:
                    relative_obj_root = os.path.join(self.split_dir_name, cate, obj)
                    sample = {
                        'cate': cate, 'object': obj, 'id': id, 'phase': self.mode,
                        'view0': os.path.join(relative_obj_root, 'frame_' + id + '_Color_00.png'),
                        'view1': os.path.join(relative_obj_root, 'frame_' + id + '_Color_01.png'),
                        'nox0': os.path.join(relative_obj_root, 'frame_' + id + '_NOXRayTL_00.png'),
                        'nox1': os.path.join(relative_obj_root, 'frame_' + id + '_NOXRayTL_01.png'),
                        'pose': os.path.join(relative_obj_root, 'frame_' + id + '_CameraPose.json'),
                    }
                    if check:  # check this sample
                        try:
                            self.grab_atom_data(sample)
                            index_list.append(sample)
                            print("\r Check %d meta data" % len(index_list), end='')
                        except:
                            print("Cate " + cate + " Objcet " + obj + " ID " + id + " get error during check")
        if check:
            # save
            with open(self.mode + '.json', 'w') as filehandle:
                json.dump(index_list, filehandle, indent=1)
        return index_list

    def group_index(self, index):
        """
        the index file is a list of each view sampe, group different view of an object
        """
        print("group index file...")
        grouped = dict()
        grouped_list = list()
        for info in index:
            if not info['object'] in grouped.keys():
                grouped[info['object']] = list()
            grouped[info['object']].append(info)
        for obj_list in grouped.values():
            if len(obj_list) >= self.n_view:
                grouped_list.append(obj_list)
        print("%d object have enough views" % len(grouped_list))
        return grouped_list


if __name__ == '__main__':
    # test
    from yacs.config import CfgNode as CN

    cfg = CN()
    ROOT_DIR = os.getcwd().split('/')
    ROOT_DIR.pop(-1)
    ROOT_DIR = ['/'] + ROOT_DIR
    cfg.ROOT_DIR = os.path.join(*ROOT_DIR)
    cfg.DATASET_ROOT = "shapenet_plain"
    cfg.DATASET_CATES = ["02958343"]
    cfg.DATASET_INDEX = ['../resource/index/shapenet_plain_car_train.json']
    cfg.PROPORTION = 0.5
    # cfg.DATASET_INDEX = []
    dataset = Dataset(cfg, 'train')

    print(len(dataset))
    for data in dataset:
        a = data
