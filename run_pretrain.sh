#!/bin/bash
set -e

# 指定のパラメータで pretrain_clipcap.py と pretrain_probvlm.py を実行する関数
run_pretrain() {
    local clip_model_type=$1
    local dataset=$2
    local num_workers=$3
    local device=$4
    local prefix=$5

    python pretrain_clipcap.py \
        --clip_model_type "$clip_model_type" \
        --dataset "$dataset" \
        --num_workers "$num_workers" \
        --device "$device" \
        --prefix "$prefix" || true

    python pretrain_probvlm.py \
        --clip_model_type "$clip_model_type" \
        --dataset "$dataset" \
        --num_workers "$num_workers" \
        --device "$device" \
        --prefix "$prefix" || true
}

# 設定 A の実行
CLIP_MODEL_TYPE_A="ViT-B/16"
DATASET_A="COCO_A"
NUM_WORKERS=8
DEVICE_A="cuda:0"
PREFIX="coco_2017_common_person_only"
run_pretrain "$CLIP_MODEL_TYPE_A" "$DATASET_A" "$NUM_WORKERS" "$DEVICE_A" "$PREFIX"

# 設定 B の実行
CLIP_MODEL_TYPE_B="ViT-B/32"
DATASET_B="COCO_B"
NUM_WORKERS=8
DEVICE_B="cuda:1"
run_pretrain "$CLIP_MODEL_TYPE_B" "$DATASET_B" "$NUM_WORKERS" "$DEVICE_B" "$PREFIX"
