#!/bin/bash

# デフォルト値の設定
dataset="coco_all"
dataset_mode=("eval")  # 複数指定する場合は引用符で囲んだ空白区切り文字列を渡してください
epochs=("29")
num_samples=10
num_workers=8
exp_name="mhcg_person_only_0"
temperature=1.0
device="cuda:3"
dataset_prefix="coco_2017_common_person_only"

# 引数解析（--exp_name, --dataset, --dataset_mode, --epochs, --num_samples, --num_workers, --temperature, --device, --dataset_prefix）
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --exp_name) exp_name="$2"; shift ;;
        --dataset) dataset="$2"; shift ;;
        --dataset_mode) dataset_mode=($2); shift ;;  # 例: "eval train"
        --epochs) epochs=($2); shift ;;             # 例: "299 300"
        --num_samples) num_samples="$2"; shift ;;
        --num_workers) num_workers="$2"; shift ;;
        --temperature) temperature="$2"; shift ;;
        --device) device="$2"; shift ;;
        --dataset_prefix) dataset_prefix="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$exp_name" ]; then
    echo "Error: --exp_name is required."
    exit 1
fi

# Captioning 実行部分
# exp_name に "ensemble" または "packllm_sim" が含まれていれば ensemble_sampling.py を、
# それ以外の場合は evaluate_captioning.py を実行する
if [[ "$exp_name" == *"ensemble"* ]] || [[ "$exp_name" == *"packllm_sim"* ]]; then
    script="ensemble_sampling.py"
    if [[ "$exp_name" == *"packllm_sim"* ]]; then
        extra_args=(--ensemble_method "packllm_sim")
    else
        extra_args=(--ensemble_method "ensemble")
    fi
else
    script="evaluate_captioning.py"
    extra_args=(--epochs "${epochs[@]}" --exp_name "${exp_name}")
fi

for mode in "${dataset_mode[@]}"; do
    python "$script" \
        --dataset_name "${dataset}" \
        --mode "${mode}" \
        --device "${device}" \
        --num_workers "${num_workers}" \
        --temperature "${temperature}" \
        --num_samples "${num_samples}" \
        --dataset_prefix "${dataset_prefix}" \
        "${extra_args[@]}" || true
done

# Category_scores の実行部分

if [[ "$exp_name" == *"ensemble"* ]] || [[ "$exp_name" == *"packllm_sim"* ]]; then
    identifiers=("AB")
elif [[ "$exp_name" == *"pretrain"* ]]; then
    identifiers=("A" "B" "ALL")
elif [[ "$exp_name" == *"linear_merged"* ]]; then
    identifiers=("A")
else
    identifiers=("A" "B")
fi

for mode in "${dataset_mode[@]}"; do
    for id in "${identifiers[@]}"; do
        python category_scores/calc_category_scores.py \
            --exp_dir "${exp_name}" \
            --base_generated_captions_path "exp_eval/${exp_name}/${dataset}/${dataset_prefix}_candidate_${id}_temperature_${temperature}_${mode}" \
            --output_file_prefix "${dataset_prefix}_${dataset}_candidate_${id}_epoch_temperature_${temperature}_${mode}" \
            --mode "${mode}" \
            --num_samples "${num_samples}" || true
    done
done