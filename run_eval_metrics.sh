#!/bin/bash

# List of datasets and epochs
datasets=("coco" "cc3m" "nocaps")
epochs=(20 15 10 5)
names=("A" "B")

# Loop through the epochs
for epoch in "${epochs[@]}"; do
    echo "Running for epoch: $epoch"

    # Loop through the datasets
    for dataset in "${datasets[@]}"; do
        echo "Running for dataset: $dataset"

        # Loop through the names A and B
        for name in "${names[@]}"; do
            echo "Running for name: $name"
            # Construct the candidates_json path dynamically
            candidates_json="exp_eval/mhcg_derpp_0.05_1/${dataset}/${dataset}_candidate_${name}_epoch_${epoch}_temperature_0.7.json"
            
            # Run the Python script
            python pacscore/compute_metrics.py --dataset ${dataset} --candidates_json ${candidates_json} || true
        done
    done
done
