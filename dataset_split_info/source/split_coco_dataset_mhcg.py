import json
from pycocotools.coco import COCO
from collections import defaultdict
import argparse


def collect_another_dataset(json_file1, json_file2, annotation_file, output_file, image_num=10000):
    try:
        # 1. JSONファイルを読み込む
        with open(json_file1, 'r') as f1, open(json_file2, 'r') as f2:
            data1 = json.load(f1)
            data2 = json.load(f2)

        # 2. 2つのJSONファイルのimage_idsを統合
        image_ids_1 = set(map(int, data1.get("image_ids", {}).keys()))
        image_ids_2 = set(map(int, data2.get("image_ids", {}).keys()))
        combined_image_ids = image_ids_1.union(image_ids_2)

        # 3. COCOアノテーションファイルからすべてのimage_idsとカテゴリ情報を取得
        coco = COCO(annotation_file)
        all_image_ids = set(coco.getImgIds())
        categories = {cat["id"]: cat["name"] for cat in coco.loadCats(coco.getCatIds())}

        # 4. 差分を計算（another dataset）
        another_image_ids = all_image_ids.difference(combined_image_ids)

        # 5. カテゴリごとの画像数を先に集計
        category_image_count = defaultdict(int)
        image_to_categories = {}  # 画像IDごとのカテゴリ情報
        for image_id in another_image_ids:
            # 各画像のアノテーションIDを取得
            ann_ids = coco.getAnnIds(imgIds=image_id)
            annotations = coco.loadAnns(ann_ids)

            # 画像に含まれるカテゴリIDを取得
            category_ids = list({ann["category_id"] for ann in annotations if "category_id" in ann})
            image_to_categories[image_id] = category_ids

            # 各カテゴリの画像数をカウント
            for category_id in category_ids:
                category_image_count[category_id] += 1

        # カテゴリを画像数が少ない順にソート
        sorted_categories = sorted(category_image_count.items(), key=lambda x: x[1])

        # 6. another datasetのカテゴリと画像ごとのカテゴリ割り当てを作成
        another_data = {
            "categories": categories,
            "image_ids": {}
        }
        used_images = set()

        # 画像数が少ないカテゴリから順に処理
        for category_id, _ in sorted_categories:
            for image_id, category_ids in image_to_categories.items():
                if image_id in used_images:
                    continue  # すでに割り当てられた画像はスキップ
                if category_id in category_ids:
                    another_data["image_ids"][str(image_id)] = category_ids
                    used_images.add(image_id)
                    if len(another_data["image_ids"]) >= image_num:
                        break
            if len(another_data["image_ids"]) >= image_num:
                break

        # 7. JSONファイルとして保存
        with open(output_file, 'w') as out_file:
            json.dump(another_data, out_file, indent=4)

        # 8. 画像の総数を出力
        print(f"another datasetに含まれる画像の総数: {len(another_data['image_ids'])}") 

        # 9. カテゴリ統計情報の出力
        category_image_count = defaultdict(int)
        for image_id, category_ids in another_data["image_ids"].items():
            for category_id in category_ids:
                category_image_count[category_id] += 1

        print("\nanother datasetのカテゴリ統計情報:")
        print(f"{'Category ID':<12} {'Category Name':<20} {'Image Count':<15}")
        print("-" * 50)
        for category_id, count in sorted(category_image_count.items()):
            category_name = categories[category_id]
            print(f"{category_id:<12} {category_name:<20} {count:<15}")

    except FileNotFoundError as e:
        print(f"ファイルが見つかりません: {e}")
    except json.JSONDecodeError:
        print("JSONファイルの形式が正しくありません。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")


# 実行
if __name__ == "__main__":
    # 各ファイルのパスを指定
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset_prefix", type=str, default="coco_2017_common_person_only")
    argparser.add_argument("--image_num", type=int, default=15000)
    args = argparser.parse_args()

    prefix = args.dataset_prefix
    image_num = args.image_num
    
    for mode in ["train", "val"]:        
        json_file1 = f"dataset_split_info/{prefix}/{mode}_dataset_a_info.json"
        json_file2 = f"dataset_split_info/{prefix}/{mode}_dataset_b_info.json"
        annotation_file = f"dataset/coco/annotations/instances_{mode}2017.json"
        output_file = f"dataset_split_info/{prefix}/{mode}_split_dataset_mhcg_info.json"
        
        collect_another_dataset(json_file1, json_file2, annotation_file, output_file, image_num=image_num)
