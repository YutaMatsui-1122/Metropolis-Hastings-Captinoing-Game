#!/bin/bash

# List of datasets and temperature values
datasets=("coco" "cc3m" "nocaps")
temperatures=(0.7 0.65 0.75)
em_iters=(20 15 10 5 0)

# Loop through the temperature values
for temp in "${temperatures[@]}"; do
    echo "Running evaluation with temperature: $temp"

    # 1st part: Evaluate with --use_official_model for each dataset
    for dataset in "${datasets[@]}"; do
        echo "Running evaluation with official model on dataset: $dataset with temperature: $temp"
        python evaluate_captioning.py --batch_size 64 --device cuda:0 --dataset $dataset --num_workers 8 --temperature $temp --use_official_model || true
    done

    # 2nd part: Evaluate with datasets and varying --em_iter
    for em_iter in "${em_iters[@]}"; do
        echo "Running evaluation with em_iter: $em_iter and temperature: $temp"

        for dataset in "${datasets[@]}"; do
            echo "Running on dataset: $dataset with em_iter: $em_iter and temperature: $temp"
            python evaluate_captioning.py --batch_size 64 --device cuda:0 --dataset $dataset --num_workers 8 --temperature $temp --em_iter $em_iter || true
        done
    done

done
