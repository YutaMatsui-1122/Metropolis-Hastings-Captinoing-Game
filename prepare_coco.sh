#!/bin/bash

# 作業ディレクトリを設定
COCO_DIR="dataset/coco"
mkdir -p $COCO_DIR

# 移動
cd $COCO_DIR

# ダウンロードするファイルのURL
TRAIN_IMAGES="http://images.cocodataset.org/zips/train2017.zip"
VAL_IMAGES="http://images.cocodataset.org/zips/val2017.zip"
ANNOTATIONS="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

echo "Starting download of COCO 2017 dataset..."

# 訓練画像をダウンロード
if [ ! -f "train2017.zip" ]; then
    echo "Downloading train2017.zip..."
    wget -q --show-progress $TRAIN_IMAGES
else
    echo "train2017.zip already exists. Skipping download."
fi

# 検証画像をダウンロード
if [ ! -f "val2017.zip" ]; then
    echo "Downloading val2017.zip..."
    wget -q --show-progress $VAL_IMAGES
else
    echo "val2017.zip already exists. Skipping download."
fi

# アノテーションをダウンロード
if [ ! -f "annotations_trainval2017.zip" ]; then
    echo "Downloading annotations_trainval2017.zip..."
    wget -q --show-progress $ANNOTATIONS
else
    echo "annotations_trainval2017.zip already exists. Skipping download."
fi

# 解凍
echo "Extracting files..."

if [ ! -d "train2017" ]; then
    echo "Extracting train2017.zip..."
    unzip -q train2017.zip
else
    echo "train2017 directory already exists. Skipping extraction."
fi

if [ ! -d "val2017" ]; then
    echo "Extracting val2017.zip..."
    unzip -q val2017.zip
else
    echo "val2017 directory already exists. Skipping extraction."
fi

if [ ! -d "annotations" ]; then
    echo "Extracting annotations_trainval2017.zip..."
    unzip -q annotations_trainval2017.zip
else
    echo "annotations directory already exists. Skipping extraction."
fi

echo "COCO 2017 dataset is ready in $COCO_DIR."
