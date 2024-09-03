# Evaluate the communication between the agents in the vlm captioning game
# Two kinds of evaluation:
# 1. Evaluate the likelihood of the latent representation given the shared sign
# 2. Evaluate the captioning performance for each pretraining dataset

# 0.1. Import the required libraries
import torch
import pickle
from one_agent import OneAgent
from utils import * 
import pickle, clip, argparse
from torch.nn import functional as nnf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from bert_score import score as bert_score
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType
import copy

# 0.2. Define the device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 0.3. Set the experiment name, directory, and the dataset
exp_name = "gemcg_unified_dataset_coco5000_cc3m5000_alpha003_beta003_1"
exp_dir = f"exp/{exp_name}"
observation_file = "communication_coco_50_cc3m_50"
temperature = 0.7
epoch = 6

agent = OneAgent(agent_name='A', device=device,temperature=temperature)
agent = agent.to(device)
agent.lora_setting()

with open(f"dataset/dataset_cache/{observation_file}.pkl", "rb") as f:
    observationA_dataset = pickle.load(f)
    observationA_dataset.prefix_length = agent.prefix_length

dataloader = torch.utils.data.DataLoader(observationA_dataset, batch_size=32, shuffle=False, num_workers=1)

agent.load_pretrain(probvlm_path=f"{exp_dir}/A/probvlm_A_{epoch}-epoch-9.pth", clipcap_path=f"{exp_dir}/A/clipcap_A_{epoch}-009.pt")

agent.dataloader_MHNG_fix = dataloader

agent.perception()
caption = agent.propose()
print(caption[:10])
for i in range(10):
    print(tokenizer_decode(caption[i]))