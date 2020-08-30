source activate pix2surf

python run_batch.py --config=coco/eval_pix2surf_sv_car.yaml
python run_batch.py --config=coco/eval_pix2surf_mv_car.yaml

python run_batch.py --config=coco/eval_pix2surf_sv_chair.yaml
python run_batch.py --config=coco/eval_pix2surf_mv_chair.yaml

python run_batch.py --config=coco/eval_pix2surf_sv_plane.yaml
python run_batch.py --config=coco/eval_pix2surf_mv_plane.yaml