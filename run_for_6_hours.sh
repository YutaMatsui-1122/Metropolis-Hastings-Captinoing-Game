#!/bin/bash

end=$((SECONDS+21600))  # 現在の秒数に6時間分の秒数を加算

while [ $SECONDS -lt $end ]; do
    ./move_files.sh # 実行したいスクリプトへのパス
    sleep 20                 # 20秒待機
done
