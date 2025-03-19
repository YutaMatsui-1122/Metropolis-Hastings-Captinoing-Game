#!/bin/bash

# 引数を処理
dataset=$1  # データセット名（nocaps または coco）
json_dir=$2  # 実験ディレクトリ

# データセットに応じてディレクトリとファイルを設定
if [ "$dataset" == "coco" ]; then
    image_dir="../dataset/coco/val2014"
    references_json="../exp_eval/refs/coco_refs.json"
    json_subdir="coco"
elif [ "$dataset" == "nocaps" ]; then
    image_dir="../dataset/nocaps/validation"
    references_json="../exp_eval/refs/nocaps_refs.json"
    json_subdir="nocaps"
elif [ "$dataset" == "cc3m" ]; then
    image_dir="../DownloadConceptualCaptions/validation_copy"
    references_json="../exp_eval/refs/cc3m_refs.json"
    json_subdir="cc3m"
else
    echo "Unsupported dataset: $dataset"
    exit 1
fi

# ディレクトリ内のすべてのjsonファイルを順番に処理
for json_file in "$json_dir/$json_subdir"/*.json; do
    # JSONファイル名の確認
    echo "Processing $json_file"
    echo "  Image directory: $image_dir"
    echo "  References JSON: $references_json"

    # compute_metrics.pyを実行
    python compute_metrics.py --candidates_json "$json_file" --image_dir "$image_dir" --references_json "$references_json" --compute_refpac
done

echo "All files processed!"
