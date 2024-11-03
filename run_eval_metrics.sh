#!/bin/bash

# List of datasets and epochs
datasets=("cc3m")
epochs=(0 18)
names=("B")
exp_name="mhcg_vit_32_16_anneal_1"
temperature=0.7
device="cuda:0"

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
            # candidates_json="exp_eval/mhcg_derpp_0.05_1/${dataset}/${dataset}_candidate_${name}_epoch_${epoch}_temperature_0.7.json"
            # candidates_json="exp_eval/pretrain/${dataset}_candidate_cc3m_temperature_0.7_vit16_epoch_${epoch}.json"
            candidates_json="exp_eval/${exp_name}/${dataset}/${dataset}_candidate_${name}_epoch_${epoch}_temperature_${temperature}.json"
            
            # Run the Python script
            python pacscore/compute_metrics.py --dataset ${dataset} --candidates_json ${candidates_json} --device ${device} || True
        done
    done
done
