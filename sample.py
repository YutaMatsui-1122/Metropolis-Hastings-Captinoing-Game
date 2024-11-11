import json
import os
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO

class CustomCocoDataset(Dataset):
    def __init__(self, dataset_file, captions_file, image_dir, transform=None):
        """
        Args:
            dataset_file (str): データセットA/BのJSONファイルパス。
            captions_file (str): COCOデータセットのキャプションアノテーションファイルパス。
            image_dir (str): 画像が保存されているディレクトリパス。
            transform (callable, optional): 画像に適用する変換。
        """
        self.image_dir = image_dir
        self.transform = transform

        # データセットファイルの読み込み
        with open(dataset_file, 'r') as f:
            self.dataset_info = json.load(f)
        
        # COCOキャプションデータの読み込み
        self.coco = COCO(captions_file)
        
        # カテゴリ情報と画像IDとカテゴリのマッピング
        self.categories = self.dataset_info["categories"]
        self.image_categories = self.dataset_info["image_ids"]
        
        # 画像IDとキャプションのペアを生成
        self.image_caption_pairs = []
        for image_id in self.image_categories.keys():
            ann_ids = self.coco.getAnnIds(imgIds=int(image_id))
            captions_info = self.coco.loadAnns(ann_ids)
            for caption_info in captions_info:
                self.image_caption_pairs.append((image_id, caption_info['caption']))
                
    def __len__(self):
        return len(self.image_caption_pairs)
    
    def __getitem__(self, idx):
        image_id, caption = self.image_caption_pairs[idx]
        
        # 画像ファイルパスの取得 (pycocotools.cocoを使用)
        image_info = self.coco.loadImgs(int(image_id))[0]
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        
        # 画像を開いて、必要に応じて変換
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # 画像に関連するカテゴリの名前を取得
        category_ids = self.image_categories[image_id]
        category_names = [self.categories[str(cat_id)] for cat_id in category_ids]
        
        return image, caption, category_names

    def get_image_categories(self, image_id):
        """
        指定した画像IDに関連するカテゴリの名前を取得
        """
        if image_id in self.image_categories:
            category_ids = self.image_categories[image_id]
            return [self.categories[str(cat_id)] for cat_id in category_ids]
        else:
            return []

# 使用例
dataset_file = "dataset_split_info/custom_split_dataset_a_info.json"  # データセットAのファイル
captions_file = "dataset/coco/annotations/annotations/captions_train2014.json"  # COCOキャプションファイル
image_dir = "dataset/coco/train2014"  # 画像ディレクトリ

dataset = CustomCocoDataset(dataset_file=dataset_file, captions_file=captions_file, image_dir=image_dir, transform=None)

dataset_size = len(dataset)
print("データセットサイズ:", dataset_size)

# get random sample
import random

for i in range(10):
    sample_image, sample_caption, sample_categories = dataset[i]
    print("キャプション:", sample_caption)
    print("カテゴリ:", sample_categories)

    # save the sample image
    import matplotlib.pyplot as plt
    plt.imshow(sample_image)
    plt.axis('off')
    plt.savefig(f"image_{i}.png")