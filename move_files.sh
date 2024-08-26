#!/bin/bash

# trainingディレクトリからtraindataディレクトリへのファイル移動用スクリプト

# 移動元ディレクトリ
SOURCE_DIR="./training"

# 移動先ディレクトリ
DEST_DIR="./traindata"

# 1から99までのファイルを順番に移動
for i in {1..99}
do
   mv "${SOURCE_DIR}/${i}_*" "${DEST_DIR}/"
done
