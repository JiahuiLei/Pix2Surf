source activate pix2surf

python run_batch.py --config=coco/xnocsinit_sv_car.yaml
python run_batch.py --config=coco/pix2surf_sv_car.yaml
python run_batch.py --config=coco/pix2surf_mv_car.yaml