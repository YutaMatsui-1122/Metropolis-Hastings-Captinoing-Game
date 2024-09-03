from one_agent import OneAgent
from utils import * 
import pickle, clip, argparse
from torch.nn import functional as nnf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from CLIP_prefix_caption.train import *
import time

parser = argparse.ArgumentParser()
parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
parser.add_argument('--exp_name', default="debug")
parser.add_argument('--MH_iter', default=100, type=int)
parser.add_argument('--annealing', default="None")
parser.add_argument('--mode', default="MHNG")
args = parser.parse_args()

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# device = torch.device("cpu")

mode = "generate" # "generate" or "eval"
generate_type = "random_fast" # "beam" or "random" or "random_fast"
dataset_name = "coco" # "coco" or "conceptual"
agent_name = "conceptual" # "coco" or "conceptual"
datacache_name = f"{dataset_name}_train_dataset_10000"
datacache_path = f"dataset/dataset_cache/{datacache_name}.pkl"

for t in [0.8, 0.9, 1.0]:
    data_path = 'dataset/'
    prefix_length = 10  # 適切なprefix_lengthを設定してください

    with open(datacache_path, "rb") as f:
        test_dataset = pickle.load(f)
        test_dataset.prefix_length = prefix_length

    num_workers = 1
    batch_size = 128

    fix_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    shuffle_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    adapter = "mlp"
    agent = OneAgent(agent_name='A',adapter=adapter, device=device)
    agent = agent.to(device)

    save_dir_file = f"models/mh_chain/mh_chain_conceptual2coco_{t}.csv"

    #空のdataframeを作成
    columns = ["Reference"]
    df = pd.DataFrame(columns=columns)

    # fix_loaderのキャプションをReferenceとする
    df["Reference"] = [caption for caption in test_dataset.captions]

    df.to_csv(save_dir_file)

    agent.load_pretrain(probvlm_path=f"models/probVLM_coco_prefix-036.pth", clipcap_path=f"models/official_model/clipcap_coco_weights.pt", strict_clipcap=False)

    agent.communication_field_setup(fix_loader, shuffle_loader, MH_iter=100)

    agent.initialize_sign()
    df["Initial"] = agent.dataloader_MHNG_fix.dataset.captions

    agent.perception()


    for i in range(100):
        proposed_w_caption = pd.read_csv(f"models/official_model/generated_sentences/conceptual_agent/generated_sentences_coco_train_dataset_10000_random_fast_conceptual_agent_{i}_{t}.csv")["Generated"].tolist()
        proposed_w = tokenize(proposed_w_caption)
        agent.judge(proposed_w.to(device))

        df[f"Generated_{i}"] = agent.dataloader_MHNG_fix.dataset.captions
        df.to_csv(save_dir_file)

