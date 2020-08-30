# Pix2Surf
One implementation of our ECCV2020 paper **[Pix2Surf: Learning Parametric 3D Surface Models of Objects from Images](https://geometry.stanford.edu/projects/pix2surf/)**

If you use the code please cite our paper.

```latex
@inproceedings{pix2surf_2020,
 author = {Lei, Jiahui and Sridhar, Srinath and Guerrero, Paul and Sung, Minhyuk and Mitra, Niloy and Guibas, Leonidas J.},
 title = {Pix2Surf: Learning Parametric 3D Surface Models of Objects from Images},
 booktitle = {Proceedings of European Conference on Computer Vision ({ECCV})},
 url = {https://geometry.stanford.edu/projects/pix2surf},
 month = August,
 year = {2020}
}
```



## Installation

We use [pytorch 1.1](https://pytorch.org/get-started/previous-versions/) and currently only support linux with GPUs. We also depend on [tk3dv](https://github.com/drsrinathsridhar/tk3dv). To install the code:
```shell script
# create virtue env
conda create -n pix2surf python=3.6
source activate pix2surf

# install pytorch with corresponding cuda version, for example we use cuda 10.0 here
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch

# install other requirements
pip install -r ./requirements.txt

# install tk3dv
pip install git+https://github.com/drsrinathsridhar/tk3dv.git
```



## Dataset

We use the dataset from X-NOCS project. Current code supports both ShapeNet-COCO (~172GB) and ShapeNet-Plain (~5GB) Dataset. Please download any or both of them from [here](https://github.com/drsrinathsridhar/xnocs/blob/master/dataset/README.md) and unzip. Then, link the dataset to the code and pre-process the files by:

```shell
# prepare dirs
cd [...]/ProjectRoot
mkdir ./log # or link to somewhere
mkdir ./resource/weight
mkdir ./resource/data

# link COCO or Plain
ln -s [...]/PathToShapeNetCOCO ./resource/data/shapenet_coco
ln -s [...]/PathToShapeNetPlain ./resource/data/shapenet_plain

# do some minor modification to the dataset structure
cd ./resource/utils
python make_coco_validation.py
```

Now, you will see the `./resource/data` folder structure like this:

```shell
├── shapenet_coco
│   ├── test
│   ├── train
│   └── vali
└── shapenet_plain
    ├── test
    └── train
```



## Run the code

Please check the `ProjectRoot/scripts` and run the `.sh` scripts under ProjectRoot dir. Here we give some examples.

### Evaluation

Here is an evaluation example for pre-trained model on ShapeNet-Plain Dataset. 

First, download pre-trained models on ShapeNet-Plain Dataset from **[here](http://download.cs.stanford.edu/orion/pix2surf/plain-weight.zip)**. Unzip and move all `*.model` to `ProjectRoot/resource/weight` and then just run:

```
cd [...]/ProjectRoot
bash ./scripts/evaluate_pix2surf.sh
```

The log will be stored at `./log/eval-pix2surf-XXXXX` and you can find the metric reports under the `xls` sub-folder. This will reproduce the numbers of both single view and multi view version of Pix2Surf in table 1 in our paper. 

Pre-trained weights on ShapeNet-COCO dataset can be found [here](http://download.cs.stanford.edu/orion/pix2surf/coco-weight.zip).

### Train your own model

Here is a training example for car category on ShapeNet-COCO Dataset.

```shell
cd [...]/ProjectRoot
bash ./scripts/pix2surf_car_coco.sh
```

All the trainings currently are configured with 2 GPUs, but 4 or more GPUs with larger batch size will probably lead to higher performance.

### Visualization

We provide a naive post-processing code to generate meshes and high resolution Geometry Images with Texture Maps. Here is an example for generating the visualization in the paper.

Please download a small dataset with consistent lighting condition across multiple views **[here](http://download.cs.stanford.edu/orion/pix2surf/pix2surf_viz_dataset.zip)**. As our method has unsupervised learned component, to reproduce the same learned chart pattern as shown in the paper, please download our weight **[here](http://download.cs.stanford.edu/orion/pix2surf/viz-weight.zip)**. (It's interesting to see that different patterns are discovered in different trainings)

```shell
# link visualziation dataset
cd [...]/ProjectRoot
ln -s [...]/pix2surf_viz ./resource/data

# move weights to resource/weight
mv [...]/PathToDownloadedWeight/*.model ./resource/weight

# run postprocessing
bash ./scripts/render_pix2surf_viz.sh
```

Under the `log\render-pix2surf-XXXXXX`, you will find `obj` sub-folder that contains the mesh results. In the  `image` sub-folder, you will find Geometry Image: `*GIM.png`or `*GIM-uni.png` and Texture Map `*TEX-uni.png` or  `*TEX.png` . Geometry Images along with Texture Maps have similar data structure as NOCS Maps along with RGB colors (different arrangement), so you can use [tk3dv tools](https://github.com/drsrinathsridhar/tk3dv/blob/master/examples/visualizeNOCSMap.py) to visualize it.



## Check more about NOCS series

**[[NOCS]](https://geometry.stanford.edu/projects/NOCS_CVPR2019/)** NOCS for Pose Estimation [CVPR2019]

**[[X-NOCS]](https://geometry.stanford.edu/projects/xnocs/)** Two-intersection NOCS for shape reconstruction [NeurIPS2019]

**[[ANCSH]](https://articulated-pose.github.io/)** Articulated Pose Estimation [CVPR2020]

**[[S-NOCS]](https://geometry.stanford.edu/projects/pix2surf/)** Shape reconstruction in NOCS with Surfaces (This work) [ECCV2020]

**[[T-NOCS]](https://geometry.stanford.edu/projects/caspr/)** NOCS along Time Axis [arXiv2020 Pre-print]