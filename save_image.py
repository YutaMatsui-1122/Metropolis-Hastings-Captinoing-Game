import pickle, clip, argparse
from utils import *
import os

argparser = argparse.ArgumentParser()
argparser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
argparser.add_argument('--dataset', default="coco", choices=('coco', 'conceptual'))
argparser.add_argument('--exp_name', default="debug")
argparser.add_argument('--initial_epoch', type=int, default=0)
argparser.add_argument('--num_workers', type=int, default=1)
argparser.add_argument('--batch_size', type=int, default=128)

args = argparser.parse_args()

dataset_file_path = "coco_train_dataset_10000"

with open(f"dataset/dataset_cache/{dataset_file_path}.pkl", "rb") as f:
    dataset = pickle.load(f)

# データセットにある画像をdataset_imagesに保存
# 保存する際は，dataset_file_pathを名前とするフォルダを作成し，その中に保存する

dataset_images_path = f"dataset_images/{dataset_file_path}"
os.makedirs(dataset_images_path, exist_ok=True)
for i in range(len(dataset)):
    image = dataset.get_img(i)
    # 画像を保存
    image.save(f"{dataset_images_path}/{i}.jpg")

