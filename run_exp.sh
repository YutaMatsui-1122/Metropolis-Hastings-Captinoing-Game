#!/bin/bash

# Define the directory where the script is located
SCRIPT_DIR="/workspace/Inter-CLIP/Inter-ProbVLM"

# List of Python scripts with arguments
scripts=(
    # "python3 $SCRIPT_DIR/derpp.py --dataset fine_tune --num_workers 8 --save_dir lora_mlp_adapter --device cuda:3 --cl_mode None"
    "python3 $SCRIPT_DIR/derpp.py --dataset fine_tune --num_workers 8 --save_dir lora_mlp_adapter_DER --device cuda:3 --cl_mode DER"
    "python3 $SCRIPT_DIR/derpp.py --dataset fine_tune --num_workers 8 --save_dir lora_mlp_adapter_DERPP --device cuda:3 --cl_mode DERPP"
)

# Execute each Python file in sequence
for script in "${scripts[@]}"
do
    echo "Running $script"
    eval $script
    if [ $? -ne 0 ]; then
        echo "Error: $script failed"
        exit 1
    fi
done

echo "All scripts executed successfully"
