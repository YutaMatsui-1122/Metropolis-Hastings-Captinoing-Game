import json
from collections import defaultdict

# カテゴリーリストの設定
categoryA = list(range(0, 40))  # データセットAの独自カテゴリー
categoryB = list(range(40, 60))  # データセットBの独自カテゴリー
categoryC = list(range(60, 80))  # 共通カテゴリー

# カテゴリーIDをセットとして格納（高速な検査のため）
categoryA_set = set(categoryA)
categoryB_set = set(categoryB)
categoryC_set = set(categoryC)

# データセットファイルのパス
instances_file_path = 'dataset/coco/annotations/annotations/instances_train2014.json'

# JSONデータの読み込み
with open(instances_file_path, 'r') as f:
    instances_data = json.load(f)

# 画像ごとのカテゴリー情報を収集
image_categories = defaultdict(set)
for annotation in instances_data['annotations']:
    image_id = annotation['image_id']
    category_id = annotation['category_id']
    image_categories[image_id].add(category_id)

# 検査
conflict_images = []  # データセットAとBの独自カテゴリーが混在する画像
common_only_images = 0  # 共通カテゴリーのみを含む画像数
common_and_a_images = 0  # 共通カテゴリーとデータセットAの独自カテゴリーを含む画像数
common_and_b_images = 0  # 共通カテゴリーとデータセットBの独自カテゴリーを含む画像数

for image_id, categories in image_categories.items():
    # カテゴリーをセットで保持
    categories = set(categories)
    
    # 各カテゴリーセットの有無を確認
    contains_common = bool(categories & categoryC_set)
    contains_a = bool(categories & categoryA_set)
    contains_b = bool(categories & categoryB_set)
    
    # 検査条件
    if contains_a and contains_b:
        # データセットAとBの独自カテゴリーが混在している画像
        conflict_images.append(image_id)
    elif contains_common:
        # 共通カテゴリーを含む画像の場合の分岐
        if contains_a:
            common_and_a_images += 1  # 共通カテゴリーとデータセットAの独自カテゴリー
        elif contains_b:
            common_and_b_images += 1  # 共通カテゴリーとデータセットBの独自カテゴリー
        else:
            common_only_images += 1  # 共通カテゴリーのみ

# 結果の出力
print("共通カテゴリーのみを含む画像数:", common_only_images)
print("共通カテゴリーとデータセットAの独自カテゴリーを含む画像数:", common_and_a_images)
print("共通カテゴリーとデータセットBの独自カテゴリーを含む画像数:", common_and_b_images)
print("データセットAとBの独自カテゴリーが混在している画像数:", len(conflict_images))

# コンフリクト画像のサンプル出力（最初の5件）
if conflict_images:
    print("\nデータセットAとBの独自カテゴリーが混在している画像IDのサンプル:")
    print(conflict_images[:5])
