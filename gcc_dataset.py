import torch
from torch.utils.data import Dataset, DataLoader

from PIL import Image

from CLIP_prefix_caption.parse_conceptual import ConceptualDS

import clip
import pickle

clip_model, preprocess = clip.load("ViT-B/32", device="cuda:3", jit=False)

conceptual_train = ConceptualDS(data_root="../dataset/conceptual", preprocess=preprocess, suffix="train")

dl = DataLoader(conceptual_train, batch_size=200, shuffle=False, num_workers=8, drop_last=False)

batch = next(iter(dl))
print(batch[0].shape)
print(batch[1])
print(batch[2])