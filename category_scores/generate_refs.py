import json
import os
from torch.utils.data import Dataset
from tqdm import tqdm
import pickle
import sys
import argparse

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from utils import *

# COCOデータセット用のカスタムデータセットクラス

class CocoDataset(Dataset):
    def __init__(self, annotation_file, img_dir, transform=None):
        self.coco = COCO(annotation_file)
        self.img_dir = img_dir
        self.transform = transform
        self.img_ids = self.coco.getImgIds()

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])

        # Get all captions for this image ID
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        captions = [ann['caption'] for ann in anns]

        return img_info['file_name'], captions

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset_prefix', default="coco_2017_common_person_only", type=str)
args = argparser.parse_args()

dataset_prefix = args.dataset_prefix
coco_annotation_file = 'dataset/coco/annotations/captions_val2017.json'
coco_images_dir = 'dataset/coco/val2017'

save_dir = f"exp_eval/refs/{dataset_prefix}"
os.makedirs(save_dir, exist_ok=True)

modes = ["val"] 
dataset_names = ["all", "a", "b"] # "a" or "b"

for mode in modes:
    for dataset_name in dataset_names:
        file_name = f"{mode}_split_dataset_{dataset_name}"
        save_file = f"{mode}_split_dataset_{dataset_name}_refs"

        if dataset_name == "all":
            train_dataset = CocoDataset(coco_annotation_file, coco_images_dir)
            dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1)
        else:
            with open(f"dataset/dataset_cache/{dataset_prefix}/{file_name}.pkl", "rb") as f:
                train_dataset = pickle.load(f)
                train_dataset.prefix_length = 10

        references = {}
        if dataset_name == "all":
            for data in tqdm(dataloader):
                img_filenames, captions = data
                img_name = os.path.splitext(img_filenames[0])[0].split("/")[-1]
                references[img_name] = [caption[0] for caption in captions]

        else:
            for i in range(len(train_dataset)):
                image_path = train_dataset.dataset[i]["image"]
                caption = [train_dataset.dataset[i]["caption"]]
                img_name = os.path.splitext(image_path)[0].split("/")[-1]

                if img_name in references:
                    references[img_name].extend(caption)
                else:
                    references[img_name] = caption

        keys = list(references.keys())[:2]

        num_captions = [len(captions) for captions in references.values()]

        with open(f"{save_dir}/{save_file}.json", "w") as f:
            json.dump(references, f, indent=4)
        
        print(f"References saved to {save_dir}/{save_file}.json")