import os
import numpy as np

# modify new version of shapenet plain to old version
if os.path.exists('../data/shapenet_plain'):
    if os.path.exists('../data/shapenet_plain/val'):
        os.system('mv ../data/shapenet_plain/val ../data/shapenet_plain/test')

# modify new version of shapenet coco to old version
# split a small validation set from training set
if os.path.exists('../data/shapenet_coco'):
    if os.path.exists('../data/shapenet_coco/val'):
        os.system('mv ../data/shapenet_coco/val ../data/shapenet_coco/test')
    source_root = '../data/shapenet_coco/train'
    target_root = '../data/shapenet_coco/vali'
    os.makedirs(target_root, exist_ok=False)

    for cate, vali_list_fn in zip(['03001627', '02958343', '02691156'],
                                  ['../utils/chair_vali.npy',
                                   '../utils/car_vali.npy',
                                   '../utils/plane_vali.npy']):
        fn_list = np.load(vali_list_fn)
        os.makedirs(os.path.join(target_root, cate))
        for fn in fn_list:
            src = os.path.join(source_root, cate, fn)
            tgt = os.path.join(target_root, cate, fn)
            cmd = 'mv ' + src + ' ' + tgt
            print(cmd)
            os.system(cmd)
