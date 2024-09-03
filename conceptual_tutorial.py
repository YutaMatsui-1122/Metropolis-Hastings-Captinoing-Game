from utils import *
from PIL import Image
import torch
from torch.utils.data import DataLoader, random_split
import clip
import time

clip_model, preprocess = clip.load("ViT-B/32", device="cpu")

# dataset = ConceptualDataset(root = "dataset/conceptual_captions_subset", transform=preprocess,data_mode="test", prefix_length=40, normalize_prefix=True)

# print(dataset.data["image_path"][38919])

# conceptual dataset
data_file = "dataset/conceptual_captions_subset/clean_train.tsv"

train_dataset, test_dataset = get_conceptual_dataset(data_file, split_ratio=0.9, root = "dataset/conceptual_captions_subset", transform=preprocess, prefix_length=40, normalize_prefix=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=1)

batch = next(iter(train_loader))
print(batch[0].shape)


with open("dataset/dataset_cache/conceptual_captions_subset_train_dataset.pkl", "wb") as f:
    pickle.dump(train_dataset, f)

with open("dataset/dataset_cache/conceptual_captions_subset_test_dataset.pkl", "wb") as f:
    pickle.dump(test_dataset, f)

# conceptual_dataloader = DataLoader(conceptual_dataset, batch_size=10, shuffle=False)
# batch = next(iter(conceptual_dataloader))
# print(batch[0].shape)
# print(batch[1])
# print(batch[2].shape)
# print(batch[3].shape)
# print(batch[4].shape)

# # save dataset with pickle
# # with open("dataset/dataset_cache/conceptual_captions_subset_dataset.pkl", "wb") as f:
# #     pickle.dump(conceptual_dataset, f)


# # coco dataset

# data_path = 'dataset/'
# prefix_length = 40
# normalize_prefix = True

# coco_test_dataset = CocoDataset(root = data_path, transform=preprocess,data_mode="test", prefix_length=prefix_length, normalize_prefix=normalize_prefix)
# coco_test_loader_fix = torch.utils.data.DataLoader(coco_test_dataset, batch_size=16, shuffle=False, num_workers=8)

# batch = next(iter(coco_test_loader_fix))
# print(batch[0].shape)
# print(batch[1])
# print(batch[2].shape)
# print(batch[3].shape)
# print(batch[4].shape)