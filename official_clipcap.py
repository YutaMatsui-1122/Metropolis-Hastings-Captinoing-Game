from one_agent import OneAgent
from utils import * 
import pickle, clip, argparse
from torch.nn import functional as nnf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from CLIP_prefix_caption.train import *
from bert_score import score as bert_score
import time

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

mode = "generate" # "generate" or "eval"
generate_type = "random_fast" # "beam" or "random" or "random_fast"
dataset_name = "coco" # "coco" or "conceptual"
datacache_name = f"{dataset_name}_train_dataset_10000"
datacache_path = f"dataset/dataset_cache/{datacache_name}.pkl"

clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
data_path = 'dataset/'
prefix_length = 10  # 適切なprefix_lengthを設定してください
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

with open(datacache_path, "rb") as f:
    test_dataset = pickle.load(f)
    test_dataset.prefix_length = prefix_length

print("test_dataset:", len(test_dataset))

num_workers = 1
batch_size = 32

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

for agent_name in ["conceptual", "coco"]:
    model = ClipCaptionModel(prefix_length, mapping_type="mlp")
    model = model.to(device)
    model.load_state_dict(torch.load(f"models/official_model/clipcap_{agent_name}_weights.pt"), strict=False)
    for i in range(100):
        print("Iteration:", i)
        for t in [0.8, 0.9, 1]:
            refs = []
            generated_text_list = []
            gens_2 = []
            image_features = []
            text_features = []
            for batch in tqdm(test_loader):
                img, caption, vlm_token, gpt_token, gpt_mask, index= batch
                img, gpt_token, gpt_mask = img.to(device), gpt_token.to(device), gpt_mask.to(device)

                refs.extend(list(caption))
                prefix = clip_model.encode_image(img).to(device, dtype=torch.float32)
                prefix_embeds = model.clip_project(prefix).reshape(prefix.shape[0], prefix_length, -1)
                image_features.append(prefix.cpu().detach())
                if generate_type == "random_fast":
                    generated_texts = generate_batch(model, tokenizer, embed=prefix_embeds, temperature=t)

                generated_text_list.extend(generated_texts)
            print("generated text", generated_text_list[:5])
            df = pd.DataFrame({'Reference': refs, 'Generated': generated_text_list})
            df.to_csv(f"models/official_model/generated_sentences/{agent_name}_agent/generated_sentences_{datacache_name}_{generate_type}_{agent_name}_agent_{i}_{t}.csv")