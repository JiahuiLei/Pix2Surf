source activate pix2surf

python run_batch.py --config=coco/xnocsinit_sv_plane.yaml
python run_batch.py --config=coco/pix2surf_sv_plane.yaml
python run_batch.py --config=coco/pix2surf_mv_plane.yaml