dataset="coco"
candidates_json="exp_eval/pretrain/coco_candidate_coco_temperature_0.7_vit32_train.json"
device="cuda:3"
dataset_mode="train"

python pacscore/compute_metrics.py --dataset ${dataset} --candidates_json ${candidates_json} --device ${device} --dataset_mode ${dataset_mode} --compute_refpac