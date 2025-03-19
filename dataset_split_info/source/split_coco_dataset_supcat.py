import json
from collections import defaultdict, Counter
import os
from pycocotools.coco import COCO
import argparse

def split_coco_categories(instances_file_path, included_supercategories=None):
    # 1. COCOデータの読み込み
    coco = COCO(instances_file_path)
    
    # 各カテゴリの情報を取得
    categories = coco.loadCats(coco.getCatIds())
    category_info = {cat['id']: {'name': cat['name'], 'supercategory': cat['supercategory']} for cat in categories}
    
    # 画像ごとにカテゴリをカウント
    category_image_counts = defaultdict(int)
    for cat_id in category_info.keys():
        image_ids = coco.getImgIds(catIds=[cat_id])
        category_image_counts[cat_id] = len(image_ids)
    
    # 2. スーパーカテゴリごとに最も多いカテゴリを共通カテゴリ (categoryC) とする
    supercategory_to_categoryC = {}
    supercategory_to_categories = defaultdict(list)
    for cat_id, info in category_info.items():
        supercategory = info['supercategory']
        supercategory_to_categories[supercategory].append(cat_id)
        
        # 最も多いカテゴリを選択
        if supercategory not in supercategory_to_categoryC:
            supercategory_to_categoryC[supercategory] = (cat_id, category_image_counts[cat_id])
        else:
            current_max_cat_id, current_max_count = supercategory_to_categoryC[supercategory]
            if category_image_counts[cat_id] > current_max_count:
                supercategory_to_categoryC[supercategory] = (cat_id, category_image_counts[cat_id])
    
    # 共通カテゴリに含めるスーパーカテゴリの設定
    if included_supercategories is None:
        included_supercategories = list(supercategory_to_categories.keys())  # デフォルトで全て含める
    
    print("Included supercategories:", included_supercategories)
    print("Supercategory to categoryC", supercategory_to_categoryC)
    
    categoryC_ids = {
        cat_id for supercategory, (cat_id, _) in supercategory_to_categoryC.items()
        if supercategory in included_supercategories
    }
    
    # 3. 残りのカテゴリを A と B に分割 (スーパーカテゴリ内で分割)
    categoryA_ids, categoryB_ids = [], []

    def balanced_split_supercategory(cat_ids, current_a_count, current_b_count):
        """
        スーパーカテゴリ内でカテゴリを分割し、AとBのカテゴリ数が均等になるよう調整。
        """
        cat_ids = sorted(cat_ids, key=lambda x: category_image_counts[x], reverse=True)
        A, B = [], []
        
        # 分割アルゴリズム: グローバルなカテゴリ数を考慮しながら分配
        for cat_id in cat_ids:
            if current_a_count + len(A) < current_b_count + len(B):
                A.append(cat_id)
            elif current_b_count + len(B) < current_a_count + len(A):
                B.append(cat_id)
            else:
                # カテゴリ数が同じ場合、画像数が少ない方に割り当てる
                if sum(category_image_counts[x] for x in A) <= sum(category_image_counts[x] for x in B):
                    A.append(cat_id)
                else:
                    B.append(cat_id)
        return A, B

    for supercategory, cat_ids in supercategory_to_categories.items():
        remaining_ids = [cat_id for cat_id in cat_ids if cat_id not in categoryC_ids]
        if not remaining_ids:
            continue
        
        # スーパーカテゴリ内でカテゴリをバランスよく分割
        A_ids, B_ids = balanced_split_supercategory(
            remaining_ids, len(categoryA_ids), len(categoryB_ids)
        )
        categoryA_ids.extend(A_ids)
        categoryB_ids.extend(B_ids)
    
    # カテゴリ名に変換
    categoryC = [category_info[cat_id]['name'] for cat_id in categoryC_ids]
    categoryA = [category_info[cat_id]['name'] for cat_id in categoryA_ids]
    categoryB = [category_info[cat_id]['name'] for cat_id in categoryB_ids]
    
    # 4. 統計情報の出力
    def print_category_statistics(categories, ids, title):
        print(f"\n{title}:")
        print(f"{'Category':<20} {'Supercategory':<20} {'Image Count':<15}")
        print("-" * 55)
        total_images = 0
        for cat_id in ids:
            cat_name = category_info[cat_id]['name']
            supercategory = category_info[cat_id]['supercategory']
            image_count = category_image_counts[cat_id]
            total_images += image_count
            print(f"{cat_name:<20} {supercategory:<20} {image_count:<15}")
        print("-" * 55)
        print(f"Total Categories: {len(ids)}")
        print(f"Total Images:    {total_images}")

    print_category_statistics(categoryC, categoryC_ids, "Category C (共通カテゴリ)")
    print_category_statistics(categoryA, categoryA_ids, "Category A")
    print_category_statistics(categoryB, categoryB_ids, "Category B")

    # sort by category_id
    categoryC_ids = sorted(list(categoryC_ids))
    categoryA_ids = sorted(list(categoryA_ids))
    categoryB_ids = sorted(list(categoryB_ids))

    return categoryC_ids, categoryA_ids, categoryB_ids




# 保存用ディレクトリとファイルprefixの設定

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset_prefix', default="coco_2017_common_person_only", type=str)
args = argparser.parse_args()

file_prefix_base = args.dataset_prefix

save_dir = f"dataset_split_info/{file_prefix_base}"

# 保存用ディレクトリの作成
os.makedirs(save_dir, exist_ok=True)

# カテゴリーCとカテゴリーAの手動設定

if "person_only" in file_prefix_base:
    included_supercategories = ["person"]
else:
    included_supercategories = None

categoryC, categoryA, categoryB = split_coco_categories("dataset/coco/annotations/instances_train2017.json", included_supercategories=included_supercategories)
categoryC_set, categoryA_set, categoryB_set = set(categoryC), set(categoryA), set(categoryB)

# 全カテゴリーのID一覧を取得
all_categories = categoryC + categoryA + categoryB
#sort by category_id
all_categories = sorted(all_categories)

# JSONデータの読み込み
for mode in ["train", "val"]:
    instances_file_path = f'dataset/coco/annotations/instances_{mode}2017.json'
    file_prefix = f"{file_prefix_base}_{mode}"

    with open(instances_file_path, 'r') as f:
        instances_data = json.load(f)

    # データ内で使用されているカテゴリID一覧を取得
    used_category_ids = set()
    for annotation in instances_data['annotations']:
        used_category_ids.add(annotation['category_id'])

    # カテゴリーIDと名前のマッピングを作成
    category_id_to_name = {cat['id']: cat['name'] for cat in instances_data['categories']}

    # 画像ごとのカテゴリー情報を収集し、マッピングインデックスを適用
    image_categories = defaultdict(set)
    for annotation in instances_data['annotations']:
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        image_categories[image_id].add(category_id)

    # カテゴリIDごとの画像枚数を集計
    category_image_count = defaultdict(int)
    for categories in image_categories.values():
        for category_id in categories:
            category_image_count[category_id] += 1

            

    # 各パターンの分類と集計
    conflict_images = []
    common_a_b_images = 0
    common_and_a_images = 0
    common_and_b_images = 0
    no_category_images = 0
    total_images = len(image_categories)

    # データセットAとBの集計用変数
    dataset_a_images = dict()
    dataset_b_images = dict()
    common_only_images = dict()
    category_a_only_images = dict()
    category_b_only_images = dict()
    whole_category_images = dict()
    dataset_a_categories = set()
    dataset_b_categories = set()


    # カテゴリの組み合わせを集計するためのカウンター
    conflict_category_pairs = Counter()
    common_a_b_category_pairs = Counter()

    for image_id, categories in image_categories.items():
        contains_common = bool(categories & categoryC_set)
        contains_a = bool(categories & categoryA_set)
        contains_b = bool(categories & categoryB_set)
        
        whole_category_images[image_id] = list(categories)

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
                dataset_a_images[image_id] = list(categories)
                dataset_a_categories.update(categories & (categoryA_set | categoryC_set))
            elif contains_b:
                common_and_b_images += 1
                dataset_b_images[image_id] = list(categories)
                dataset_b_categories.update(categories & (categoryB_set | categoryC_set))
            else:
                common_only_images[image_id] = list(categories)
                dataset_a_images[image_id] = list(categories)
                dataset_b_images[image_id] = list(categories)
                dataset_a_categories.update(categories & categoryC_set)
                dataset_b_categories.update(categories & categoryC_set)
        elif contains_a:
            category_a_only_images[image_id] = list(categories)
            dataset_a_images[image_id] = list(categories)
            dataset_a_categories.update(categories & categoryA_set)
        elif contains_b:
            category_b_only_images[image_id] = list(categories)
            dataset_b_images[image_id] = list(categories)
            dataset_b_categories.update(categories & categoryB_set)

    # 結果の出力
    print("画像の総数:", total_images)
    print("すべてのカテゴリの画像数:", len(whole_category_images))
    print("カテゴリを含まない画像数:", no_category_images)
    print("共通カテゴリーのみを含む画像数:", len(common_only_images))
    print("共通カテゴリーとデータセットAの独自カテゴリーを含む画像数:", common_and_a_images)
    print("共通カテゴリーとデータセットBの独自カテゴリーを含む画像数:", common_and_b_images)
    print("カテゴリAのみを含む画像数:", len(category_a_only_images))
    print("カテゴリBのみを含む画像数:", len(category_b_only_images))

    # データセットAとBの独自カテゴリーが混在している画像数と共通カテゴリーとデータセットAとBの独自カテゴリーがすべて含まれる画像数を改行後に出力
    print("\nデータセットAとBの独自カテゴリーが混在している画像数:", len(conflict_images))
    print("共通カテゴリーとデータセットAとBの独自カテゴリーがすべて含まれる画像数:", common_a_b_images)

    # 合計値の計算と出力
    sum_of_images = (
        no_category_images + len(common_only_images) + common_and_a_images +
        common_and_b_images + len(category_a_only_images) + len(category_b_only_images) +
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

    with open(os.path.join(save_dir, f"{mode}_split_info.json"), "w") as f:
        json.dump(split_info, f, indent=4)
    print(f"データセット分割情報を '{mode}_split_info.json' に保存しました。")

    # データセットAとBの情報を保存
    dataset_a_info = {
        "categories": {cat_id: category_id_to_name[cat_id] for cat_id in dataset_a_categories},
        "image_ids": dataset_a_images,
    }
    dataset_b_info = {
        "categories": {cat_id: category_id_to_name[cat_id] for cat_id in dataset_b_categories},
        "image_ids": dataset_b_images,
    }

    category_a_only_images_info = {
        "categories": {cat_id: category_id_to_name[cat_id] for cat_id in categoryA_set},
        "image_ids": category_a_only_images,
    }

    category_b_only_images_info = {
        "categories": {cat_id: category_id_to_name[cat_id] for cat_id in categoryB_set},
        "image_ids": category_b_only_images,
    }

    common_only_images_info = {
        "categories": {cat_id: category_id_to_name[cat_id] for cat_id in categoryC_set},
        "image_ids": common_only_images,
    }

    whole_category_images_info = {
        "categories": {cat_id: category_id_to_name[cat_id] for cat_id in all_categories},
        "image_ids": whole_category_images,
    }

    print(list(common_only_images_info["image_ids"].values())[:10])


    with open(os.path.join(save_dir, f"{mode}_dataset_a_info.json"), "w") as f:
        json.dump(dataset_a_info, f, indent=4)
    print(f"データセットAの情報を '{mode}_dataset_a_info.json' に保存しました。")

    with open(os.path.join(save_dir, f"{mode}_dataset_b_info.json"), "w") as f:
        json.dump(dataset_b_info, f, indent=4)
    print(f"データセットBの情報を '{mode}_dataset_b_info.json' に保存しました。")

    with open(os.path.join(save_dir, f"{mode}_category_a_only_images_info.json"), "w") as f:
        json.dump(category_a_only_images_info, f, indent=4)
    print(f"データセットAの独自カテゴリーの画像情報を '{mode}_category_a_only_images_info.json' に保存しました。")

    with open(os.path.join(save_dir, f"{mode}_category_b_only_images_info.json"), "w") as f:
        json.dump(category_b_only_images_info, f, indent=4)
    print(f"データセットBの独自カテゴリーの画像情報を '{mode}_category_b_only_images_info.json' に保存しました。")

    with open(os.path.join(save_dir, f"{mode}_common_only_images_info.json"), "w") as f:
        json.dump(common_only_images_info, f, indent=4)
    print(f"共通カテゴリーのみを含む画像情報を '{mode}_common_only_images_info.json' に保存しました。")

    with open(os.path.join(save_dir, f"{mode}_whole_category_images_info.json"), "w") as f:
        json.dump(whole_category_images_info, f, indent=4)
    print(f"すべてのカテゴリーの画像情報を '{mode}_whole_category_images_info.json' に保存しました。")

    num = 10
    print("\n共通カテゴリーとデータセットAとBの独自カテゴリーがすべて含まれるカテゴリの組み合わせ（多い順）:")

    for i, ((a_cat, b_cat), count) in enumerate(common_a_b_category_pairs.most_common()):
        print(f"カテゴリA: {a_cat}, カテゴリB: {b_cat}, 画像数: {count}")
        if i >= num - 1:
            break