#!/bin/bash

# alpha_betaとsave_dirの値のリスト
alpha_beta_values=(0.05)

# バッチサイズ、デバイス、エージェント、ワーカー数の設定
batch_size=64
device="cuda:3"
speaker_agent="coco"
num_workers=8

# ループしてコマンドを実行
for alpha_beta in "${alpha_beta_values[@]}"
do
    save_dir="derpp_test_coco_${alpha_beta}"
    python derpp.py --batch_size $batch_size --device $device --speaker_agent $speaker_agent --save_dir $save_dir --num_workers $num_workers --alpha_beta $alpha_beta
done