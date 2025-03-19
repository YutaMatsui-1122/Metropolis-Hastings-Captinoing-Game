import pickle
import torch
import clip
import os
import sys
import argparse

two_levels_up = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(two_levels_up)
from utils import SplitCocoDataset


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
clip_model, preprocess = clip.load("ViT-B/32", device=device)

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset_prefix', default="coco_2017_common_person_only", type=str)
args = argparser.parse_args()

file_prefix = args.dataset_prefix

save_dir = f"dataset/dataset_cache/{file_prefix}"
os.makedirs(save_dir, exist_ok=True)

for mode in ["train", "val"]:
    # Divide the COCO dataset into two parts
    dataset_files = [f"dataset_split_info/{file_prefix}/{mode}_dataset_a_info.json", f"dataset_split_info/{file_prefix}/{mode}_dataset_b_info.json",  f"dataset_split_info/{file_prefix}/{mode}_split_dataset_mhcg_info.json", ] 
    for dataset_name, dataset_file in zip(["a", "b", "mhcg",], dataset_files):
        captions_file = f"dataset/coco/annotations/captions_{mode}2017.json"  # COCOキャプションファイル
        image_dir = f"dataset/coco/{mode}2017"  # 画像ディレクトリ

        dataset = SplitCocoDataset(dataset_file=dataset_file, captions_file=captions_file, image_dir=image_dir, transform=preprocess, dataset_name=dataset_name)

        # save the dataset
        with open(f"dataset/dataset_cache/{file_prefix}/{mode}_split_dataset_{dataset_name}.pkl", "wb") as f:
            pickle.dump(dataset, f)

        print("length of the dataset:", len(dataset))
        print("Dataset saved" , f"dataset/dataset_cache/{file_prefix}/{mode}_split_dataset_{dataset_name}.pkl")