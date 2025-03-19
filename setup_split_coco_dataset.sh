
dataset_prefix="coco_2017_common_person_only"

# データセットの分割
python dataset_split_info/source/split_coco_dataset_supcat.py --dataset_prefix "${dataset_prefix}"

# MHCG 用のデータセットの作成
python dataset_split_info/source/split_coco_dataset_mhcg.py --dataset_prefix "${dataset_prefix}"

# データセット.pklの保存
python dataset_split_info/source/split_coco_dataset_save.py --dataset_prefix "${dataset_prefix}"

# データセットごとの評価用の参照データの作成
python category_scores/generate_refs.py --dataset_prefix "${dataset_prefix}"