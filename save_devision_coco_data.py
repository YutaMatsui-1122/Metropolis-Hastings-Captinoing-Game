import json
from collections import defaultdict, Counter

# カテゴリーCとカテゴリーAの手動設定
categoryC = [0, 56, 2, 60, 41, 39, 45, 26, 7, 24, 13, 73, 67, 71, 62, 57, 74, 43, 58, 16, 32, 9, 15, 5, 25, 27, 59, 42, 75, 72]  # 共通カテゴリー
categoryA = [12, 38, 61, 4, 30, 65, 47, 53, 36, 40, 48, 33, 20, 64, 19, 22, 49, 10, 31, 18, 54, 79, 6, 70, 78]  # データセットAの独自カテゴリー
prefix = "coco_dataset_split"  # ファイル名の共通プレフィックス

# 重複チェック機能の追加
if len(categoryC) != len(set(categoryC)) or len(categoryA) != len(set(categoryA)):
    print("エラー: カテゴリCまたはカテゴリAに重複したIDがあります。")
    exit()

# カテゴリBは0〜79のうち、カテゴリCとカテゴリAに含まれない残りのカテゴリで設定
all_categories = set(range(80))
categoryC_set, categoryA_set = set(categoryC), set(categoryA)
categoryB_set = all_categories - categoryC_set - categoryA_set
categoryB = sorted(list(categoryB_set))

# データセットファイルのパス
instances_file_path = 'dataset/coco/annotations/annotations/instances_train2014.json'

# JSONデータの読み込み
with open(instances_file_path, 'r') as f:
    instances_data = json.load(f)

# カテゴリーIDとスーパーカテゴリのマッピング
category_id_to_supercategory = {cat['id']: cat['supercategory'] for cat in instances_data['categories']}
category_id_to_name = {cat['id']: cat['name'] for cat in instances_data['categories']}

# オリジナルIDからマッピングIDへの変換
used_category_ids = sorted({cat['id'] for cat in instances_data['categories']})
category_id_to_mapped_index = {category_id: i for i, category_id in enumerate(used_category_ids)}
mapped_index_to_category_id = {i: category_id for i, category_id in enumerate(used_category_ids)}

# 画像ごとのカテゴリー情報を収集
image_categories = defaultdict(set)
for annotation in instances_data['annotations']:
    image_id = annotation['image_id']
    category_id = annotation['category_id']
    image_categories[image_id].add(category_id)

# データセット分割
dataset_a_images = {image_id: list(categories) for image_id, categories in image_categories.items() if categories & categoryA_set or categories & categoryC_set}
dataset_b_images = {image_id: list(categories) for image_id, categories in image_categories.items() if categories & categoryB_set or categories & categoryC_set}

# データセットA情報をファイルに保存
dataset_a_file = f"{prefix}_dataset_a.json"
with open(dataset_a_file, 'w') as f:
    json.dump(dataset_a_images, f)
print(f"データセットAの情報を '{dataset_a_file}' に保存しました。")

# データセットB情報をファイルに保存
dataset_b_file = f"{prefix}_dataset_b.json"
with open(dataset_b_file, 'w') as f:
    json.dump(dataset_b_images, f)
print(f"データセットBの情報を '{dataset_b_file}' に保存しました。")

# 分割情報ファイル
split_info = {
    "Dataset_A": {
        "categories": list(categoryA_set | categoryC_set),
        "num_images": len(dataset_a_images),
        "image_ids": list(dataset_a_images.keys())
    },
    "Dataset_B": {
        "categories": list(categoryB_set | categoryC_set),
        "num_images": len(dataset_b_images),
        "image_ids": list(dataset_b_images.keys())
    },
    "Common_Categories": list(categoryC_set),
    "CategoryA_only": list(categoryA_set),
    "CategoryB_only": list(categoryB_set),
    "Total_images": len(image_categories)
}
split_info_file = f"{prefix}_split_info.json"
with open(split_info_file, 'w') as f:
    json.dump(split_info, f, indent=4)
print(f"分割情報を '{split_info_file}' に保存しました。")

# カテゴリIDと名前のマッピングを保存
category_mapping_file = f"{prefix}_category_mapping.json"
category_mapping = {str(id_): name for id_, name in category_id_to_name.items()}
with open(category_mapping_file, 'w') as f:
    json.dump(category_mapping, f, indent=4)
print(f"カテゴリIDと名前のマッピングを '{category_mapping_file}' に保存しました。")

# データセットA、Bのカテゴリ名一覧を保存（マッピングIDをオリジナルIDに変換）
dataset_a_categories_names = {mapped_index_to_category_id[id_]: category_id_to_name[mapped_index_to_category_id[id_]] for id_ in categoryA_set | categoryC_set}
dataset_b_categories_names = {mapped_index_to_category_id[id_]: category_id_to_name[mapped_index_to_category_id[id_]] for id_ in categoryB_set | categoryC_set}

dataset_a_names_file = f"{prefix}_dataset_a_category_names.txt"
with open(dataset_a_names_file, 'w') as f:
    for id_, name in sorted(dataset_a_categories_names.items()):
        f.write(f"カテゴリID: {id_}, 名前: {name}\n")
print(f"データセットAのカテゴリ名を '{dataset_a_names_file}' に保存しました。")

dataset_b_names_file = f"{prefix}_dataset_b_category_names.txt"
with open(dataset_b_names_file, 'w') as f:
    for id_, name in sorted(dataset_b_categories_names.items()):
        f.write(f"カテゴリID: {id_}, 名前: {name}\n")
print(f"データセットBのカテゴリ名を '{dataset_b_names_file}' に保存しました。")

# データセットAおよびデータセットBに関する詳細情報をテキスト形式で出力
detailed_info_file = f"{prefix}_detailed_info.txt"
with open(detailed_info_file, 'w') as f:
    f.write("データセットA + 共通カテゴリの画像情報:\n")
    f.write(f"カテゴリ数: {len(categoryA_set | categoryC_set)}, 画像数: {len(dataset_a_images)}\n")
    f.write("画像IDとカテゴリ:\n")
    for img_id, cats in dataset_a_images.items():
        f.write(f"画像ID: {img_id}, カテゴリ: {cats}\n")
    f.write("\nデータセットB + 共通カテゴリの画像情報:\n")
    f.write(f"カテゴリ数: {len(categoryB_set | categoryC_set)}, 画像数: {len(dataset_b_images)}\n")
    f.write("画像IDとカテゴリ:\n")
    for img_id, cats in dataset_b_images.items():
        f.write(f"画像ID: {img_id}, カテゴリ: {cats}\n")
print(f"詳細情報を '{detailed_info_file}' に保存しました。")
