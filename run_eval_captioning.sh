#!/bin/bash

# デフォルト値の設定
exp_name="all_acceptance_person_only_0"
datasets=("coco_a" "coco_b" "coco_all")
dataset_mode=("eval")
epochs=(29)
temperature=0.7
device="cuda:1"
num_workers=8
dataset_prefix="coco_2017_common_person_only"

# 引数解析（--exp_name）
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --exp_name) exp_name="$2"; shift ;;
        --datasets) datasets=($2); shift ;;
        --dataset_mode) dataset_mode=($2); shift ;;
        --epochs) epochs=($2); shift ;;
        --temperature) temperature="$2"; shift ;;
        --device) device="$2"; shift ;;
        --num_workers) num_workers="$2"; shift ;;
        --dataset_prefix) dataset_prefix="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$exp_name" ]; then
    echo "Error: --exp_name is required."
    exit 1
fi

# --- Captioning 実行部分 ---
# exp_name に "ensemble" または "packllm_sim" が含まれていれば ensemble_sampling.py を、
# それ以外の場合は evaluate_captioning.py を実行する
if [[ "$exp_name" == *"ensemble"* ]] || [[ "$exp_name" == *"packllm_sim"* ]]; then
    caption_script="ensemble_sampling.py"
    if [[ "$exp_name" == *"packllm_sim"* ]]; then
        caption_extra_args=(--ensemble_method packllm_sim)
    else
        caption_extra_args=(--ensemble_method ensemble)
    fi
else
    caption_script="evaluate_captioning.py"
    if [[ "$exp_name" == *"pretrain"* ]]; then
        caption_additional_args=(--use_pretrain_model)
    else
        caption_additional_args=(--exp_name "${exp_name}" --epochs "${epochs[@]}")
    fi
    caption_extra_args=("${caption_additional_args[@]}")
fi

# 各 dataset, mode に対して Captioning 実行部を呼び出す
for dataset in "${datasets[@]}"; do
    for mode in "${dataset_mode[@]}"; do        
        python "$caption_script" \
            --dataset_name "${dataset}" \
            --mode "${mode}" \
            --device "${device}" \
            --num_workers "${num_workers}" \
            --temperature "${temperature}" \
            --dataset_prefix "${dataset_prefix}" \
            "${caption_extra_args[@]}" || true
    done
done

# --- Category_scores の実行部分 ---
# 識別子は、exp_name によって以下のように決定
# ・"ensemble" または "packllm_sim" が含まれていれば "AB"
# ・"pretrain" が含まれていれば ("A" "B" "ALL")
# ・"linear_merged" が含まれていれば ("A")
# ・それ以外（distillation など）は ("A" "B")
if [[ "$exp_name" == *"ensemble"* ]] || [[ "$exp_name" == *"packllm_sim"* ]]; then
    identifiers=("AB")
elif [[ "$exp_name" == *"pretrain"* ]]; then
    identifiers=("A" "B" "ALL")
elif [[ "$exp_name" == *"linear_merged"* ]]; then
    identifiers=("A")
else
    identifiers=("A" "B")
fi

for dataset in "${datasets[@]}"; do
    for mode in "${dataset_mode[@]}"; do
        for id in "${identifiers[@]}"; do
            candidates_json="exp_eval/${exp_name}/${dataset}/${dataset_prefix}_candidate_${id}_temperature_${temperature}_${mode}.json"
            echo "Computing metrics for candidates_json: ${candidates_json}"
            python pacscore/compute_metrics_coco_divide.py \
                --dataset "${dataset}" \
                --candidates_json "${candidates_json}" \
                --device "${device}" \
                --dataset_mode "${mode}" \
                --dataset_prefix "${dataset_prefix}" \
                --compute_refpac || true
        done
    done
done
