import json
from collections import defaultdict

# カテゴリーリストの設定（手動設定しやすい形式に変更）
categoryA = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
categoryB = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
categoryC = [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]

# カテゴリーIDのセットとして格納（高速な検査のため）
categoryA_set = set(categoryA)
categoryB_set = set(categoryB)
categoryC_set = set(categoryC)

# カテゴリの過不足チェック
all_categories = categoryA_set | categoryB_set | categoryC_set
expected_categories = set(range(80))

missing_categories = expected_categories - all_categories
duplicate_categories = all_categories - expected_categories

if missing_categories or duplicate_categories:
    print("カテゴリの過不足エラーがあります。")
    if missing_categories:
        print("不足しているカテゴリ:", sorted(missing_categories))
    if duplicate_categories:
        print("重複しているカテゴリ:", sorted(duplicate_categories))
    exit()  # エラーがある場合は処理を終了

# データセットファイルのパス
instances_file_path = 'dataset/coco/annotations/annotations/instances_train2014.json'

# JSONデータの読み込み
with open(instances_file_path, 'r') as f:
    instances_data = json.load(f)

# データ内で使用されているカテゴリID一覧を取得
used_category_ids = set()
for annotation in instances_data['annotations']:
    used_category_ids.add(annotation['category_id'])

# 使用されているカテゴリIDを昇順にソートして連続インデックスにマッピング
used_category_ids = sorted(used_category_ids)
category_id_to_mapped_index = {category_id: i for i, category_id in enumerate(used_category_ids)}
mapped_index_to_category_id = {i: category_id for i, category_id in enumerate(used_category_ids)}

# カテゴリーIDと名前のマッピングを作成
category_id_to_name = {cat['id']: cat['name'] for cat in instances_data['categories']}

# 画像ごとのカテゴリー情報を収集し、マッピングインデックスを適用
image_categories = defaultdict(set)
for annotation in instances_data['annotations']:
    image_id = annotation['image_id']
    original_category_id = annotation['category_id']
    mapped_index = category_id_to_mapped_index[original_category_id]  # マッピングインデックスを取得
    image_categories[image_id].add(mapped_index)

# カテゴリIDごとの画像枚数を集計
category_image_count = defaultdict(int)
for categories in image_categories.values():
    for category_id in categories:
        category_image_count[category_id] += 1

# 各パターンの分類と集計
conflict_images = []
common_only_images = 0
common_and_a_images = 0
common_and_b_images = 0
common_a_b_images = 0
category_a_only_images = 0
category_b_only_images = 0
no_category_images = 0
total_images = len(image_categories)
unclassified_images = 0
unclassified_image_ids = []

# データセットAとBの集計用変数
dataset_a_images = set()
dataset_b_images = set()
dataset_a_categories = set()
dataset_b_categories = set()

for image_id, categories in image_categories.items():
    contains_common = bool(categories & categoryC_set)
    contains_a = bool(categories & categoryA_set)
    contains_b = bool(categories & categoryB_set)
    
    if not categories:
        no_category_images += 1
    elif contains_a and contains_b and contains_common:
        common_a_b_images += 1
    elif contains_a and contains_b:
        conflict_images.append(image_id)
    elif contains_common:
        if contains_a:
            common_and_a_images += 1
            dataset_a_images.add(image_id)
            dataset_a_categories.update(categories & (categoryA_set | categoryC_set))
        elif contains_b:
            common_and_b_images += 1
            dataset_b_images.add(image_id)
            dataset_b_categories.update(categories & (categoryB_set | categoryC_set))
        else:
            common_only_images += 1
            dataset_a_images.add(image_id)
            dataset_b_images.add(image_id)
            dataset_a_categories.update(categories & categoryC_set)
            dataset_b_categories.update(categories & categoryC_set)
    elif contains_a:
        category_a_only_images += 1
        dataset_a_images.add(image_id)
        dataset_a_categories.update(categories & categoryA_set)
    elif contains_b:
        category_b_only_images += 1
        dataset_b_images.add(image_id)
        dataset_b_categories.update(categories & categoryB_set)
    else:
        unclassified_images += 1
        unclassified_image_ids.append((image_id, categories))

# 結果の出力
print("画像の総数:", total_images)
print("カテゴリを含まない画像数:", no_category_images)
print("共通カテゴリーのみを含む画像数:", common_only_images)
print("共通カテゴリーとデータセットAの独自カテゴリーを含む画像数:", common_and_a_images)
print("共通カテゴリーとデータセットBの独自カテゴリーを含む画像数:", common_and_b_images)
print("カテゴリAのみを含む画像数:", category_a_only_images)
print("カテゴリBのみを含む画像数:", category_b_only_images)
print("上記のパターンに属さない画像数:", unclassified_images)

# データセットAとBの独自カテゴリーが混在している画像数と共通カテゴリーとデータセットAとBの独自カテゴリーがすべて含まれる画像数を改行後に出力
print("\nデータセットAとBの独自カテゴリーが混在している画像数:", len(conflict_images))
print("共通カテゴリーとデータセットAとBの独自カテゴリーがすべて含まれる画像数:", common_a_b_images)

# データセットAとデータセットBのカテゴリ数と画像数を出力
print("\nデータセットAの集計:")
print("カテゴリ数:", len(dataset_a_categories))
print("画像数:", len(dataset_a_images))

print("\nデータセットBの集計:")
print("カテゴリ数:", len(dataset_b_categories))
print("画像数:", len(dataset_b_images))

# カテゴリIDごとの画像枚数とカテゴリ名の出力（画像枚数が多い順）
sorted_category_image_count = sorted(category_image_count.items(), key=lambda x: x[1], reverse=True)

print("\n画像枚数が多い順のカテゴリID一覧:")
for category_id, count in sorted_category_image_count:
    category_name = category_id_to_name.get(mapped_index_to_category_id[category_id], "不明なカテゴリ")
    print(f"MI {category_id}, カテゴリ名: {category_name}, 画像枚数: {count} 枚")
