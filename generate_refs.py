import json
import os
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import pickle
from utils import *


class ConceptualCaptionsDataset(Dataset):
    def __init__(self, gpt2_type: str = "gpt2", data_mode: str = "train", img_dir=None, transform=None, return_captions=True, prefix_length=10, ids=None, datasize="full"):
        """
        Initializes the Conceptual Captions dataset class.
        
        Parameters:
        - gpt2_type: Type of GPT-2 tokenizer to use.
        - data_mode: 'train' or 'test' mode.
        - img_dir: Directory where images are stored (if provided).
        - transform: Optional transformations to apply to the images.
        - return_captions: Boolean indicating whether to return captions (default: True).
        - prefix_length: Length of the prefix for padding tokens.
        - ids: Specific indices to use for the dataset.
        - datasize: 'full' or subset of the dataset size.
        """
        self.root = "../../DownloadConceptualCaptions"
        self.data_mode = data_mode
        self.datasize = datasize
        self.return_captions = return_captions
        self.transform = transform
        self.img_dir = img_dir
        self.prefix_length = prefix_length
        
        self.set_conceptual_dataset(ids)
    
    def set_conceptual_dataset(self, ids):
        if self.data_mode == "train":
            self.image_root = os.path.join(self.root, 'training')
            self.data_file = os.path.join(self.root, 'training_imgtxt.tsv')
        elif self.data_mode == "test":
            self.image_root = os.path.join(self.root, 'validation')
            self.data_file = os.path.join(self.root, 'validation_imgtxt.tsv')
        
        # Read tsv file and subset the data if ids is provided
        if ids is not None:
            self.data = pd.read_csv(self.data_file, sep='\t').iloc[ids].reset_index(drop=True)
        else:
            self.data = pd.read_csv(self.data_file, delimiter='\t', header=0)
        
        self.ids = list(range(len(self.data)))

        # Manage the dataset size
        if self.datasize == "full":
            self.ids = self.ids
        elif "full" in self.datasize:
            datasize = int(self.datasize.split("_")[1])
            self.ids = self.ids[:datasize]
        elif self.datasize is not None:
            self.ids = self.ids[::5][:int(self.datasize)]
        else:
            self.ids = self.ids[::5]
        
        self.captions = self.data['caption'].tolist()
        self.images = self.data['image'].tolist()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Get image path
        img_path = os.path.join(self.image_root, self.images[idx])
        img_filename = self.images[idx]

        # Load image
        img = Image.open(img_path).convert('RGB')

        # Apply transformations if any
        if self.transform is not None:
            img = self.transform(img)

        # If return_captions is False, return only the image and file name
        if not self.return_captions:
            return img, img_filename

        # Retrieve caption
        caption = self.captions[idx]

        # Return the image, caption, and image file name
        return img, caption, img_filename

class NoCapsDataset(Dataset):
    def __init__(self, annotation_file, img_dir, return_captions=True, transform=None):
        """
        Initializes the NoCaps dataset class.

        Parameters:
        - annotation_file: Path to the annotation file.
        - img_dir: Directory where images are stored.
        - return_captions: Boolean indicating whether to return captions (default: True).
        - transform: Optional transformations to apply to the images.
        """
        # Load the annotation file
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        self.img_dir = img_dir
        self.transform = transform
        self.return_captions = return_captions
        
        # Store image information (mapping between file name and ID)
        self.images = {img['id']: img['file_name'] for img in self.annotations['images']}
        
        # Map image IDs to their corresponding captions
        self.id_to_captions = {}
        for ann in self.annotations['annotations']:
            if ann['image_id'] not in self.id_to_captions:
                self.id_to_captions[ann['image_id']] = []
            self.id_to_captions[ann['image_id']].append(ann['caption'])

    def __len__(self):
        # Return the total number of images
        return len(self.images)

    def __getitem__(self, idx):
        # Get the image ID for the given index
        img_id = list(self.images.keys())[idx]
        
        # Retrieve the file name for the image
        img_filename = self.images[img_id]
        img_path = os.path.join(self.img_dir, img_filename)
        
        # Load the image
        image = Image.open(img_path).convert('RGB')

        # Apply any provided transformations to the image
        if self.transform:
            image = self.transform(image)
        
        # If return_captions is False, return only the image and file name
        if not self.return_captions:
            return image, img_filename

        # Retrieve the captions corresponding to the image ID
        captions = self.id_to_captions[img_id]

        # Return the image, captions, and file name
        return image, captions, img_filename
    

train_file = "coco_2017_val_split_dataset_b"
save_file = "coco_b_val_refs"

with open(f"dataset/dataset_cache/{train_file}.pkl", "rb") as f:
    train_dataset = pickle.load(f)
    train_dataset.prefix_length = 10

import matplotlib.pyplot as plt

references = {}
print(len(train_dataset))
for i in range(len(train_dataset)):
    image_path = train_dataset.images[i]
    caption = [train_dataset.captions[i]]
    img_name = os.path.splitext(image_path)[0].split("/")[-1]
    # if i <10:
    #     print(img_name, caption, image_path)
    #     img_name_ = img_name.split("/")[-1]
    #     print(img_name_)
    # image_path = train_dataset.dataset[i]["image"]
    # caption = train_dataset.dataset[i]["caption"]
    if img_name in references:
        references[img_name].extend(caption)
    else:
        references[img_name] = caption
    # references[img_name] = caption

keys = list(references.keys())[:2]
print(keys)

print([references[key] for key in keys])
print("Number of images:", len(references))
num_captions = [len(captions) for captions in references.values()]
print("Number of captions per image:")
print("Min:", min(num_captions))
print("Max:", max(num_captions))

with open(f"exp_eval/refs/{save_file}.json", "w") as f:
    json.dump(references, f, indent=4)

# dataset_name = "cc3m" # "nocaps" or "cc3m"

# # NoCapsアノテーションファイルと画像ディレクトリのパス
# nocaps_annotation_file = '../../dataset/nocaps/nocaps_val_4500_captions.json'
# nocaps_images_dir = '../../dataset/nocaps/validation/'

# # NoCapsデータセットの初期化
# if dataset_name == "nocaps":
#     dataset = NoCapsDataset(annotation_file=nocaps_annotation_file, img_dir=nocaps_images_dir, return_captions=True)
# else:
#     dataset = ConceptualCaptionsDataset(data_mode = "test", return_captions=True, datasize="full")

# print("dataset length:", len(dataset))

# # refs.jsonの生成
# references = {}

# # with tqdm
# for i in tqdm(range(len(dataset))):
#     # 画像、キャプション、ファイル名を取得
#     image, captions, img_filename = dataset[i]
    
#     # 画像ファイル名から拡張子 .jpg を除去
#     img_name = os.path.splitext(img_filename)[0]
    
#     # 画像ファイル名をキーとしてキャプションリストを保存
#     if dataset_name == "cc3m":
#         captions = [captions]
#     references[img_name] = captions

# # nocaps_refs.jsonファイルに保存
# with open(f"{dataset_name}_refs.json", "w") as f:
#     json.dump(references, f, indent=4)

# print(f"References saved to {dataset_name}_refs.json")