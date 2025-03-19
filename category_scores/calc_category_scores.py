import os
import json
from collections import defaultdict
from pycocotools.coco import COCO
from tqdm import tqdm
import argparse

def load_annotations(categories_path, captions_path):
    """COCOのカテゴリ情報とキャプション情報を読み込む"""
    print("Loading COCO annotations...")
    coco_instances = COCO(categories_path)
    coco_captions = COCO(captions_path)
    return coco_instances, coco_captions

def extract_categories(json_file_path):
    """
    Extract unique and common categories from dataset_a and dataset_b in a JSON file.

    Args:
        json_file_path (str): Path to the JSON file.

    Returns:
        dict: A dictionary with keys "unique_a", "unique_b", and "common", containing the respective category IDs.
    """
    with open(json_file_path, "r") as f:
        data = json.load(f)

    # Extract category IDs from dataset_a and dataset_b
    categories_a = set(map(int, data["dataset_a"]["categories"].keys()))
    categories_b = set(map(int, data["dataset_b"]["categories"].keys()))

    # Compute unique and common categories
    unique_a = sorted(categories_a - categories_b)  # Categories unique to dataset_a
    unique_b = sorted(categories_b - categories_a)  # Categories unique to dataset_b
    common = sorted(categories_a & categories_b)    # Categories common to both

    # Construct result dictionary
    result = {
        "dataset_a": unique_a,
        "dataset_b": unique_b,
        "common": common
    }

    return result

def load_multiple_generated_captions(base_path, num_samples, coco_instances):
    """
    複数のgenerated_captions JSONファイルを読み込む。
    キーを画像ファイル名（.jpgを除いた形）から画像IDに変換する。
    """
    print("Loading multiple generated captions...")
    combined_captions = defaultdict(list)

    # ファイル名から画像IDへの対応を作成
    print("Creating filename-to-image-ID mapping...")
    filename_to_image_id = {}
    all_images = coco_instances.loadImgs(coco_instances.getImgIds())
    for img in all_images:
        file_name_no_ext = os.path.splitext(img["file_name"])[0]
        filename_to_image_id[file_name_no_ext] = img["id"]

    # 生成キャプションの読み込みとキーの変換
    for i in range(num_samples):
        file_path = f"{base_path}_{i}.json"
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                captions = json.load(f)
                for file_name_no_ext, caption in captions.items():
                    if file_name_no_ext in filename_to_image_id:
                        image_id = filename_to_image_id[file_name_no_ext]
                        combined_captions[image_id].append(caption)
                    else:
                        print(f"Warning: {file_name_no_ext} not found in COCO dataset.")
        else:
            print(f"File not found: {file_path}")

    return combined_captions

def create_image_category_mapping(coco_instances):
    """画像IDとそれに含まれるカテゴリの対応辞書を作成"""
    print("Creating image-category mapping...")
    image_category_mapping = defaultdict(list)

    # すべての画像IDを取得
    all_image_ids = coco_instances.getImgIds()
    for image_id in tqdm(all_image_ids):
        # 画像に関連するカテゴリIDを取得
        annotation_ids = coco_instances.getAnnIds(imgIds=[image_id])
        annotations = coco_instances.loadAnns(annotation_ids)
        category_ids = {annotation["category_id"] for annotation in annotations}
        image_category_mapping[image_id].extend(category_ids)

    return image_category_mapping

def load_synonyms(synonyms_path):
    """類義語ファイルを読み込む"""
    print("Loading synonyms...")
    with open(synonyms_path, "r") as f:
        synonyms_data = json.load(f)
    return synonyms_data

def create_category_synonym_mapping(synonyms_data, num_synonyms=5):
    """カテゴリIDと類義語リストの対応辞書を作成"""
    print("Creating category-synonym mapping...")
    category_synonym_mapping = {}

    for category_id, synonyms_info in synonyms_data.items():
        # 上位5つの一語および二語の類義語を取得
        one_gram_synonyms = [synonym[0] for synonym in synonyms_info["one_gram_synonyms"][:num_synonyms]]
        two_gram_synonyms = [synonym[0] for synonym in synonyms_info["two_gram_synonyms"][:num_synonyms]]
        # 類義語リストを結合
        all_synonyms = one_gram_synonyms + two_gram_synonyms
        category_synonym_mapping[int(category_id)] = all_synonyms

    return category_synonym_mapping

def count_synonyms_in_captions(combined_captions, category_synonym_mapping, image_category_mapping):
    """
    各カテゴリの類義語がキャプションに含まれる回数をカウント。
    """
    print("Counting synonyms in captions...")
    synonym_counts = defaultdict(int)
    correct_synonym_counts = defaultdict(int)
    ground_truth_counts = defaultdict(int)

    for image_id, captions in tqdm(combined_captions.items()):
        
        # 画像に含まれるカテゴリIDを取得
        correct_category_ids = image_category_mapping[image_id]

        for category_id in correct_category_ids:
            ground_truth_counts[category_id] += len(captions)

        for caption in captions:
            # キャプションを単語に分割（スペース区切り）
            words = caption.lower().replace(".", "").split()

            for category_id, synonyms in category_synonym_mapping.items():
                synonym_found = False
                for synonym in synonyms:
                    # 類義語が1-gramまたは2-gramであるかを確認
                    synonym_words = synonym.lower().split()
                    n = len(synonym_words)
                    for i in range(len(words) - n + 1):
                        if words[i:i + n] == synonym_words:
                            synonym_counts[category_id] += 1
                            synonym_found = True
                            break  # 同じキャプション内で1度カウントしたら終了
                    if synonym_found:
                        break
            
                # 正解カテゴリの類義語がキャプションに含まれる回数をカウント
                if category_id in correct_category_ids and synonym_found:
                    correct_synonym_counts[category_id] += 1                

    return synonym_counts, correct_synonym_counts, ground_truth_counts

def calculate_metrics(M_p, M_c, M_g, dataset_split_info, ignored_categories=[1]):
    """
    Calculate overall and per-category precision, recall, and F1-measures, including dataset-specific metrics.

    Args:
        M_p (dict): Predicted synonym counts per category.
        M_c (dict): Correct synonym counts per category.
        M_g (dict): Ground truth synonym counts per category.
        dataset_split_info (dict): Dictionary with "dataset_a", "dataset_b", and "common" categories.
        ignored_categories (list): List of category IDs to ignore during the calculation.

    Returns:
        dict: A dictionary containing overall and dataset-specific metrics.
    """

    def filter_categories(categories, ignored_categories):
        """Filter categories by excluding ignored categories."""
        return [cat for cat in categories if cat not in ignored_categories]

    # Filter categories for dataset splits
    categories_a = filter_categories(dataset_split_info["dataset_a"], ignored_categories)
    categories_b = filter_categories(dataset_split_info["dataset_b"], ignored_categories)
    categories_common = filter_categories(dataset_split_info["common"], ignored_categories)

    def compute_metrics(filtered_categories):
        """Compute metrics for a given set of categories."""
        CP_dict = {}
        CR_dict = {}
        CF1_dict = {}

        for category_id in sorted(filtered_categories):
            precision = M_c[category_id] / M_p[category_id] if M_p.get(category_id, 0) > 0 else 0
            recall = M_c[category_id] / M_g[category_id] if M_g.get(category_id, 0) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

            CP_dict[category_id] = precision
            CR_dict[category_id] = recall
            CF1_dict[category_id] = f1_score

        category_count = len(filtered_categories)
        CP = sum(CP_dict.values()) / category_count if category_count > 0 else 0
        CR = sum(CR_dict.values()) / category_count if category_count > 0 else 0
        CF1 = 2 * CP * CR / (CP + CR) if CP + CR > 0 else 0

        total_M_c = sum(M_c[cat] for cat in filtered_categories)
        total_M_p = sum(M_p[cat] for cat in filtered_categories)
        total_M_g = sum(M_g[cat] for cat in filtered_categories)

        OP = total_M_c / total_M_p if total_M_p > 0 else 0
        OR = total_M_c / total_M_g if total_M_g > 0 else 0
        OF1 = 2 * OP * OR / (OP + OR) if OP + OR > 0 else 0

        return {
            "OP": OP,
            "OR": OR,
            "OF1": OF1,
            "CP": CP,
            "CR": CR,
            "CF1": CF1,
            "CP_per_category": CP_dict,
            "CR_per_category": CR_dict,
            "CF1_per_category": CF1_dict
        }

    # Calculate overall metrics
    overall_categories = filter_categories(M_g.keys(), ignored_categories)
    overall_metrics = compute_metrics(overall_categories)

    # Calculate dataset-specific metrics
    metrics_a = compute_metrics(categories_a)
    metrics_b = compute_metrics(categories_b)
    metrics_common = compute_metrics(categories_common)

    # Return all metrics
    return {
        "overall": overall_metrics,
        "dataset_a": metrics_a,
        "dataset_b": metrics_b,
        "common": metrics_common
    }


# 以下はテスト用のメイン関数
def main():
    parser = argparse.ArgumentParser(description="Evaluate image captions using synonyms.")
    parser.add_argument("--exp_dir", type=str, default="mhcg_person_only_0", help="Experiment directory.")
    parser.add_argument("--base_generated_captions_path", type=str, default="exp_eval/mhcg_person_only_0/coco_all/coco_all_candidate_A_epoch_29_temperature_1.0_eval", help="Base path for generated captions.")
    parser.add_argument("--output_file_prefix", type=str, default="synonym_analysis_v2", help="Output file name.")
    parser.add_argument("--data_dir", type=str, default="dataset/coco", help="Directory containing the dataset.")
    parser.add_argument("--mode", type=str, choices=["eval", "train"], default="eval", help="Mode (eval or train).")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of generated caption samples.")
    parser.add_argument("--synonyms_path", type=str, default="category_scores/all_categories_synonyms_top100.json", help="Path to the synonyms file.")
    parser.add_argument("--dataset_prefix", type=str, default="coco_2017_common_person_only", help="Prefix for the dataset.")

    args = parser.parse_args()

    # パスの設定
    if args.mode == "eval":
        images_dir = os.path.join(args.data_dir, "val2017")
        captions_annotation_path = os.path.join(args.data_dir, "annotations/captions_val2017.json")
        categories_annotation_path = os.path.join(args.data_dir, "annotations/instances_val2017.json")
    else:
        images_dir = os.path.join(args.data_dir, "train2017")
        captions_annotation_path = os.path.join(args.data_dir, "annotations/captions_train2017.json")
        categories_annotation_path = os.path.join(args.data_dir, "annotations/instances_train2017.json")

    # データの読み込み
    coco_instances, coco_captions = load_annotations(categories_annotation_path, captions_annotation_path)

    # 生成キャプションの収集
    combined_captions = load_multiple_generated_captions(args.base_generated_captions_path, args.num_samples, coco_instances)

    # 画像IDとカテゴリの対応辞書作成
    image_category_mapping = create_image_category_mapping(coco_instances)

    # データセットのカテゴリ分割情報を読み込む
    dataset_split_path = f"dataset_split_info/{args.dataset_prefix}/train_split_info.json"
    dataset_split_info = extract_categories(dataset_split_path)

    # 類義語ファイルの読み込み
    synonyms_data = load_synonyms(args.synonyms_path)

    # カテゴリIDと類義語リストの対応辞書作成
    category_synonym_mapping = create_category_synonym_mapping(synonyms_data)

    # 類義語がキャプションに含まれる回数をカウント
    M_p, M_c, M_g = count_synonyms_in_captions(combined_captions, category_synonym_mapping, image_category_mapping)

    # メトリクスの計算
    metrics = calculate_metrics(M_p, M_c, M_g, dataset_split_info)

    # 結果の保存
    output_dir = f"exp_eval/{args.exp_dir}/synonym_analysis"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, args.output_file_prefix + ".json")
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
