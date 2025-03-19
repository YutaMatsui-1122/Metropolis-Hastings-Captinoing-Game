from huggingface_hub import create_repo, upload_folder

# Hugging Face Hub リポジトリ名（存在しない場合、自動で作成される）
repo_id = "ymatsui1122/clipcap_person_only_linear_merged"  

# リポジトリを自動作成（存在している場合はそのまま）
create_repo(repo_id, exist_ok=True)

# HTTP 経由でモデルをアップロード
upload_folder(
    repo_id=repo_id,
    folder_path="models/coco_2017_common_person_only/merged_models",
    commit_message="Initial upload of merged model"
)

print("モデルが正常にアップロードされました！")
