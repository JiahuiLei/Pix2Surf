"""
NOX dataset for neurips nocs dataset
"""

import torch.utils.data as data
import os
import numpy as np
import cv2 as cv
import json


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

        self.shape = (320, 240)
        self.supervision_cap = 4096  # how many pixel in nocs map foreground will be sampled to trian SP branch

        # if use noise background augmentation
        self.add_noise_flag = cfg.ADD_BACKGROUND_NOISE

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
        # self.index = self.build_index(check=True)

        # use the proportion configuration to reduce the dataset for development
        if cfg.PROPORTION < 1.0:
            p = cfg.PROPORTION
            all_num = len(self.index)
            self.index = self.index[:int(all_num * p - 1)]
        return

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        meta_info = self.index[idx]
        return self.grab(meta_info)

    def grab(self, meta_info):
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

        sample_uv_v, sample_mask_v, sample_xyz_v = self.get_uv_mask_xyz(v_nocs_x_bg)
        sample_uv_x, sample_mask_x, sample_xyz_x = self.get_uv_mask_xyz(x_nocs_x_bg)

        meta_info_reduced = {
            'cate': meta_info['cate'],
            'object': meta_info['object'],
            'id': meta_info['id'],
            'phase': meta_info['phase']
        }
        return {
            'rgb-v': v_rgb_w_bg,
            'rgb-x': x_rgb_x_bg,
            'nox-v': v_nocs_x_bg,
            'nox-x': x_nocs_x_bg,
            'mask-v': mask_v,
            'mask-x': mask_x,
            'uv-v': sample_uv_v,
            'uv-x': sample_uv_x,
            'uv-mask-v': sample_mask_v,
            'uv-mask-x': sample_mask_x,
            'uv-xyz-v': sample_xyz_v,
            'uv-xyz-x': sample_xyz_x,
            'info': meta_info_reduced,  # meta info
            'pose': meta_info['pose'],
        }

    def get_uv_mask_xyz(self, nocs_map):
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
        # randomly shuffle
        arr = np.array(range(0, uv_list.shape[0]))
        np.random.shuffle(arr)
        uv_list = uv_list[arr, :]
        nocs_list_v = nocs_list_v[arr, :]
        # now two lists are both shorter/equal to the supervision size
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
        make inner index for each data, to make sure each data sample is usable
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
                    # index_list.append(sample) # used to debug
                    if check:  # check this sample
                        try:
                            self.grab(sample)
                            index_list.append(sample)
                            print("\r Check %d meta data" % len(index_list), end='')
                        except:
                            print("Cate " + cate + " Objcet " + obj + " ID " + id + " get error during check")
        if check:
            # save
            with open(self.mode + '.json', 'w') as filehandle:
                json.dump(index_list, filehandle)
        return index_list


if __name__ == '__main__':
    from yacs.config import CfgNode as CN
    from matplotlib import pyplot as plt

    plt.ioff()
    cfg = CN()

    ROOT_DIR = os.getcwd().split('/')
    ROOT_DIR.pop(-1)
    ROOT_DIR = ['/'] + ROOT_DIR
    cfg.ROOT_DIR = os.path.join(*ROOT_DIR)
    cfg.DATASET_ROOT = "pix2surf_viz"
    cfg.DATASET_CATES = ['airplanes']
    cfg.PROPORTION = 1.0
    cfg.ADD_BACKGROUND_NOISE = False
    # cfg.DATASET_INDEX = ['../resource/index/shapenet_plain_car_train.json',
    #                      '../resource/index/shapenet_plain_car_test.json']
    cfg.DATASET_INDEX = []
    # dataset = Dataset(cfg, 'train')
    dataset = Dataset(cfg, 'test')
    for dbg in dataset:
        plt.subplot(2, 2, 1)
        plt.imshow(dbg['rgb-v'].squeeze()[..., [2, 1, 0]] / 255)
        plt.subplot(2, 2, 2)
        plt.imshow(dbg['nox-v'].squeeze())
        plt.subplot(2, 2, 3)
        plt.imshow(dbg['nox-x'].squeeze())
        plt.subplot(2, 2, 4)
        plt.imshow(dbg['mask-v'].squeeze())
        plt.show()
        plt.pause(1)
