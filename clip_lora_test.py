import torch
import pandas as pd
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType
from one_agent import OneAgent
import pickle, clip, argparse
from utils import *
from tqdm import tqdm
import numpy as np
from torch.optim import AdamW
from torch.nn import functional as nnf
from ProbVLM.src.losses import *
import nltk
from nltk.translate.meteor_score import meteor_score
from transformers import get_linear_schedule_with_warmup
import sacrebleu
from bert_score import score as bert_score

argparser = argparse.ArgumentParser()
argparser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
argparser.add_argument('--dataset', default="coco", choices=('coco', 'conceptual'))
argparser.add_argument('--exp_name', default="debug")
argparser.add_argument('--initial_epoch', type=int, default=0)
argparser.add_argument('--num_workers', type=int, default=1)
argparser.add_argument('--batch_size', type=int, default=128)

args = argparser.parse_args()

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

clip_model, preprocess = clip.load(args.clip_model_type, device=device)
r=2

test_file = "conceptual_test_dataset_10000"
model_dir = "models/DERPP_conceptual_gencap2coco"

agent = OneAgent(agent_name='A')
with open(f"dataset/dataset_cache/{test_file}.pkl", "rb") as f:
    coco_valid_dataset = pickle.load(f)
    coco_valid_dataset.prefix_length = agent.prefix_length

test_dataloader = torch.utils.data.DataLoader(coco_valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

for i in [17]:
    agent = OneAgent(agent_name='A')
    agent.load_pretrain(probvlm_path="models/probVLM_coco_prefix-030.pth", clipcap_path="models/official_model/clipcap_coco_weights.pt", strict_clipcap=False)
    agent = agent.to(device)
    print(f"Epoch {i}")  
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=r, lora_alpha=32, lora_dropout=0.1, target_modules=["c_fc", "c_proj"])
    agent.ClipCap.gpt = get_peft_model(agent.ClipCap.gpt, peft_config)
    agent.ClipCap.load_state_dict(torch.load(f"{model_dir}/coco_prefix-{i:03d}.pt"))

    print(f"Device: {device}")

    sample_num = len(coco_valid_dataset)

    t = generate_test(agent.ClipCap, clip_model, test_dataloader, agent.tokenizer, sample_num=sample_num, generate_mode="greedy", device=device, prefix_length=10, temperature=0.7)
    print(len(coco_valid_dataset.captions), len(t))
    df = pd.DataFrame({'Reference': coco_valid_dataset.captions[:sample_num], 'Generated': t})
    df.to_csv(f"{model_dir}/{test_file}-{i:03d}.csv", index=False)

    exit()


    refs = []
    gens = []
    image_ids = []
    text_features = []
    image_features = []
    test_loss = 0

    for idx, batch in enumerate(test_dataloader):
        img, caption, vlm_token, gpt_token, gpt_mask, index, image_id  = batch
        img, gpt_token, gpt_mask = img.to(device), gpt_token.to(device), gpt_mask.to(device)

        # prefix = clip_model.encode_image(img).to(device, dtype=torch.float32)
        prefix = clip_model.encode_image(img).to(device)

        
        exit()
        image_ids.extend([id.item() for id in image_id])
        print(t[0])
        refs.extend(list(caption))
        gens.extend(list(t))

    
    df = pd.DataFrame({'Reference': refs, 'Generated': gens, 'ImageID': image_ids})
    df.to_csv(f"{model_dir}/clipcap-{i:03d}.csv", index=False)
