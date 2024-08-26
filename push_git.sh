#!/bin/bash

# 変更をステージングに追加
git add .

# コミットを作成
commit_message="Auto commit on $(date)"
git commit -m "$commit_message"

# リモートリポジトリにプッシュ
git push origin main

# 終了メッセージ
echo "Changes have been pushed to the remote repository."
