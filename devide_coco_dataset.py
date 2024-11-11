import json
from collections import defaultdict, Counter
import os

# 保存用ディレクトリとファイルprefixの設定
save_dir = "dataset_split_info"
file_prefix = "custom_split"

# 保存用ディレクトリの作成
os.makedirs(save_dir, exist_ok=True)

# カテゴリーCとカテゴリーAの手動設定
categoryC = [0, 56, 2, 60, 41, 39, 45, 26, 7, 24, 13, 73, 67, 71, 62, 57, 74, 43, 58, 16, 32, 9, 15, 5, 25, 27, 59, 42, 75, 72]  # 共通カテゴリー
categoryA = [12, 38, 61, 4, 30, 65, 47, 53, 36, 40, 48, 33, 20, 64, 19, 22, 49, 10, 31, 18, 54, 79, 6, 70, 78]  # データセットAの独自カテゴリー

print(len(categoryC), len(categoryA))

# 重複チェック機能の追加
if len(categoryC) != len(set(categoryC)):
    print("エラー: カテゴリCに重複したIDがあります。")
    exit()

if len(categoryA) != len(set(categoryA)):
    print("エラー: カテゴリAに重複したIDがあります。")
    exit()

# カテゴリBは0〜79のうち、カテゴリCとカテゴリAに含まれない残りのカテゴリで設定
all_categories = set(range(80))
categoryC_set = set(categoryC)
categoryA_set = set(categoryA)
categoryB_set = all_categories - categoryC_set - categoryA_set
categoryB = sorted(list(categoryB_set))  # リストとしてソート

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

# COCOのデフォルトIDに戻す関数
def convert_to_coco_ids(mapped_ids):
    return [mapped_index_to_category_id[mapped_id] for mapped_id in mapped_ids if mapped_id in mapped_index_to_category_id]

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
common_a_b_images = 0
common_only_images = 0
common_and_a_images = 0
common_and_b_images = 0
category_a_only_images = 0
category_b_only_images = 0
no_category_images = 0
total_images = len(image_categories)

# データセットAとBの集計用変数
dataset_a_images = dict()
dataset_b_images = dict()
dataset_a_categories = set()
dataset_b_categories = set()

# カテゴリの組み合わせを集計するためのカウンター
conflict_category_pairs = Counter()
common_a_b_category_pairs = Counter()

for image_id, categories in image_categories.items():
    contains_common = bool(categories & categoryC_set)
    contains_a = bool(categories & categoryA_set)
    contains_b = bool(categories & categoryB_set)
    
    if not categories:
        no_category_images += 1
    elif contains_a and contains_b and contains_common:
        common_a_b_images += 1
        # カテゴリAとカテゴリBに跨るカテゴリの組み合わせを集計
        a_categories = categories & categoryA_set
        b_categories = categories & categoryB_set
        for a_cat in a_categories:
            for b_cat in b_categories:
                common_a_b_category_pairs[(a_cat, b_cat)] += 1
    elif contains_a and contains_b:
        conflict_images.append(image_id)
        # カテゴリAとカテゴリBに跨るカテゴリの組み合わせを集計
        a_categories = categories & categoryA_set
        b_categories = categories & categoryB_set
        for a_cat in a_categories:
            for b_cat in b_categories:
                conflict_category_pairs[(a_cat, b_cat)] += 1
    elif contains_common:
        if contains_a:
            common_and_a_images += 1
            dataset_a_images[image_id] = convert_to_coco_ids(list(categories))
            dataset_a_categories.update(categories & (categoryA_set | categoryC_set))
        elif contains_b:
            common_and_b_images += 1
            dataset_b_images[image_id] = convert_to_coco_ids(list(categories))
            dataset_b_categories.update(categories & (categoryB_set | categoryC_set))
        else:
            common_only_images += 1
            dataset_a_images[image_id] = convert_to_coco_ids(list(categories))
            dataset_b_images[image_id] = convert_to_coco_ids(list(categories))
            dataset_a_categories.update(categories & categoryC_set)
            dataset_b_categories.update(categories & categoryC_set)
    elif contains_a:
        category_a_only_images += 1
        dataset_a_images[image_id] = convert_to_coco_ids(list(categories))
        dataset_a_categories.update(categories & categoryA_set)
    elif contains_b:
        category_b_only_images += 1
        dataset_b_images[image_id] = convert_to_coco_ids(list(categories))
        dataset_b_categories.update(categories & categoryB_set)

# 結果の出力
print("画像の総数:", total_images)
print("カテゴリを含まない画像数:", no_category_images)
print("共通カテゴリーのみを含む画像数:", common_only_images)
print("共通カテゴリーとデータセットAの独自カテゴリーを含む画像数:", common_and_a_images)
print("共通カテゴリーとデータセットBの独自カテゴリーを含む画像数:", common_and_b_images)
print("カテゴリAのみを含む画像数:", category_a_only_images)
print("カテゴリBのみを含む画像数:", category_b_only_images)

# データセットAとBの独自カテゴリーが混在している画像数と共通カテゴリーとデータセットAとBの独自カテゴリーがすべて含まれる画像数を改行後に出力
print("\nデータセットAとBの独自カテゴリーが混在している画像数:", len(conflict_images))
print("共通カテゴリーとデータセットAとBの独自カテゴリーがすべて含まれる画像数:", common_a_b_images)

# 合計値の計算と出力
sum_of_images = (
    no_category_images + common_only_images + common_and_a_images +
    common_and_b_images + category_a_only_images + category_b_only_images +
    len(conflict_images) + common_a_b_images
)
print("\n上記の数値の合計:", sum_of_images)

# データセットAとデータセットBのカテゴリ数と画像数を出力
print("\nデータセットAの集計:")
print("カテゴリ数:", len(dataset_a_categories))
print("画像数:", len(dataset_a_images))

print("\nデータセットBの集計:")
print("カテゴリ数:", len(dataset_b_categories))
print("画像数:", len(dataset_b_images))

# もとのカテゴリIDに戻して出力
dataset_a_categories = convert_to_coco_ids(dataset_a_categories)
dataset_b_categories = convert_to_coco_ids(dataset_b_categories)

# 分割に関する詳細情報を保存
split_info = {
    "total_images": total_images - len(conflict_images) - common_a_b_images,    
    "not_used_images": len(conflict_images) + common_a_b_images,
    "dataset_a": {
        "total_images": len(dataset_a_images),
        "categories": {cat_id:category_id_to_name[cat_id] for cat_id in dataset_a_categories},
    },
    "dataset_b": {
        "total_images": len(dataset_b_images),
        "categories": {cat_id:category_id_to_name[cat_id] for cat_id in dataset_b_categories},
    },
}

with open(os.path.join(save_dir, f"{file_prefix}_split_info.json"), "w") as f:
    json.dump(split_info, f, indent=4)
print(f"データセット分割情報を '{file_prefix}_split_info.json' に保存しました。")

# データセットAとBの情報を保存
dataset_a_info = {
    "categories": {cat_id: category_id_to_name[cat_id] for cat_id in dataset_a_categories},
    "image_ids": dataset_a_images,
}
dataset_b_info = {
    "categories": {cat_id: category_id_to_name[cat_id] for cat_id in dataset_b_categories},
    "image_ids": dataset_b_images,
}

with open(os.path.join(save_dir, f"{file_prefix}_dataset_a_info.json"), "w") as f:
    json.dump(dataset_a_info, f, indent=4)
print(f"データセットAの情報を '{file_prefix}_dataset_a_info.json' に保存しました。")

with open(os.path.join(save_dir, f"{file_prefix}_dataset_b_info.json"), "w") as f:
    json.dump(dataset_b_info, f, indent=4)
print(f"データセットBの情報を '{file_prefix}_dataset_b_info.json' に保存しました。")

num = 10
print("\n共通カテゴリーとデータセットAとBの独自カテゴリーがすべて含まれるカテゴリの組み合わせ（多い順）:")

for i, ((a_cat, b_cat), count) in enumerate(common_a_b_category_pairs.most_common()):
    print(f"カテゴリA: {a_cat}, カテゴリB: {b_cat}, 画像数: {count}")
    if i >= num - 1:
        break