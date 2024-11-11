python evaluate_captioning.py --exp_name mhcg_vit_32_16_probvlm_lora_traindata_1 --dataset_name cc3m --num_workers 8 --device cuda:3 || True    
python evaluate_captioning.py --exp_name mhcg_vit_32_16_probvlm_lora_traindata_1 --dataset_name coco --num_workers 8 --device cuda:3 || True
python evaluate_captioning.py --exp_name mhcg_vit_32_16_probvlm_lora_traindata_1 --dataset_name cc3m --num_workers 8 --device cuda:3 --mode train || True
python evaluate_captioning.py --exp_name mhcg_vit_32_16_probvlm_lora_traindata_1 --dataset_name coco --num_workers 8 --device cuda:3 --mode train || True
bash run_eval_metrics.sh