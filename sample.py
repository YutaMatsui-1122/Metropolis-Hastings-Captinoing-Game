from pycocotools.coco import COCO
import random
from collections import defaultdict

# COCOデータセットのパス
path = "dataset/coco/annotations/instances_train2017.json"

# COCOデータセットの読み込み
coco = COCO(path)

# 1. カテゴリ情報の取得と分割
categories = coco.loadCats(coco.getCatIds())
category_frequency = {cat['id']: len(coco.getImgIds(catIds=cat['id'])) for cat in categories}

# 頻度順にソートして上位カテゴリを共通知識カテゴリとして選択
common_categories = {cat['id'] for cat in sorted(categories, key=lambda x: category_frequency[x['id']], reverse=True)[:30]}
unique_categories = [cat['id'] for cat in categories if cat['id'] not in common_categories]

# 固有カテゴリをVLM1用とVLM2用に分割
random.shuffle(unique_categories)
unique_categories_vlm1 = set(unique_categories[:len(unique_categories) // 2])
unique_categories_vlm2 = set(unique_categories[len(unique_categories) // 2:])

# 2. 画像の分配
vlm1_images, vlm2_images, both_images, mixed_images = set(), set(), set(), set()

for img_id in coco.getImgIds():
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    img_categories = set([ann['category_id'] for ann in anns])

    # 共通知識カテゴリのみを含む画像
    if img_categories.issubset(common_categories):
        both_images.add(img_id)
    # 固有知識カテゴリ（VLM1専用）を含む画像
    elif img_categories.issubset(common_categories.union(unique_categories_vlm1)):
        vlm1_images.add(img_id)
    # 固有知識カテゴリ（VLM2専用）を含む画像
    elif img_categories.issubset(common_categories.union(unique_categories_vlm2)):
        vlm2_images.add(img_id)
    # 両方の固有カテゴリが含まれる画像
    else:
        mixed_images.add(img_id)  # 両方の固有カテゴリが含まれるため、別管理

# 3. 統計情報の出力関数
def dataset_statistics(dataset_images, name):
    stats = {cat['name']: 0 for cat in categories}
    for img_id in dataset_images:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        unique_categories_in_image = set([ann['category_id'] for ann in anns])  # 重複を削除
        for cat_id in unique_categories_in_image:
            cat_name = coco.loadCats([cat_id])[0]['name']
            stats[cat_name] += 1
    print(f"\n{name} データセット統計:")
    print(f"画像数: {len(dataset_images)}")
    for cat_name, count in stats.items():
        if count > 0:
            print(f"{cat_name}: {count}")

# データセット統計の表示
dataset_statistics(vlm1_images, "VLM1")
dataset_statistics(vlm2_images, "VLM2")
dataset_statistics(both_images, "共通知識")
dataset_statistics(mixed_images, "混在画像")

# 4. スーパーカテゴリの偏りを分析する関数
def analyze_supercategory_bias(dataset_images, name):
    supercategory_counts = defaultdict(int)
    for img_id in dataset_images:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        unique_categories_in_image = set([ann['category_id'] for ann in anns])  # 重複を削除
        for cat_id in unique_categories_in_image:
            supercategory = coco.loadCats([cat_id])[0]['supercategory']
            supercategory_counts[supercategory] += 1

    print(f"\n{name} データセット スーパーカテゴリ統計:")
    for supercat, count in supercategory_counts.items():
        print(f"{supercat}: {count}")

# 各データセットごとのスーパーカテゴリ偏りの表示
analyze_supercategory_bias(vlm1_images, "VLM1")
analyze_supercategory_bias(vlm2_images, "VLM2")
analyze_supercategory_bias(both_images, "共通知識")
analyze_supercategory_bias(mixed_images, "混在画像")

# 5. カテゴリ一覧をスーパーカテゴリ別に整理して出力
def list_categories_by_supercategory(category_ids, name):
    categories_by_supercategory = defaultdict(list)
    for cat_id in category_ids:
        cat_info = coco.loadCats([cat_id])[0]
        categories_by_supercategory[cat_info['supercategory']].append(cat_info['name'])
    
    print(f"\n{name} データセットに含まれるカテゴリ（スーパーカテゴリごと）:")
    for supercat, cat_list in categories_by_supercategory.items():
        print(f"{supercat}: {', '.join(cat_list)}")

# 各データセットに含まれるカテゴリ一覧（スーパーカテゴリ別）の出力
list_categories_by_supercategory(common_categories.union(unique_categories_vlm1), "VLM1")
list_categories_by_supercategory(common_categories.union(unique_categories_vlm2), "VLM2")
list_categories_by_supercategory(common_categories, "共通知識")
