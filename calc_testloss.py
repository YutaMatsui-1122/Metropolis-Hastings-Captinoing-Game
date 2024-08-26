import torch
from one_agent import OneAgent
import pickle, clip, argparse
from utils import *
import sys
from torch.optim import AdamW
from torch.nn import functional as nnf
from ProbVLM.src.losses import *
from transformers import get_linear_schedule_with_warmup
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType
import pandas as pd

argparser = argparse.ArgumentParser()
argparser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
argparser.add_argument('--dataset', default="coco", choices=('coco', 'conceptual', 'fine_tune', 'cc3m','10000_pretrain'))
argparser.add_argument('--exp_name', default="debug")
argparser.add_argument('--initial_epoch', type=int, default=0)
argparser.add_argument('--num_workers', type=int, default=1)
argparser.add_argument('--batch_size', type=int, default=32)
argparser.add_argument('--save_dir', default="debug/")
argparser.add_argument('--device', default="cuda:0")
argparser.add_argument('--cl_mode', default="None", choices=('None','DER', 'DERPP', 'ER', 'ER_RS'))
argparser.add_argument('--reservoir_false', action='store_false', dest='reservoir')

args = argparser.parse_args()


os.makedirs(f"models/{args.save_dir}", exist_ok=True)

device = torch.device(args.device)

torch.autograd.set_detect_anomaly(True)

print(device)

clip_model, preprocess = clip.load(args.clip_model_type, device=device)


agent = OneAgent(agent_name='A', buffer_size=1000, device=device)
agent = agent.to(device)
with open("dataset/dataset_cache/coco_test_dataset_5000.pkl", "rb") as f:
    coco_valid_dataset = pickle.load(f)
    coco_valid_dataset.prefix_length = agent.prefix_length
agent.load_pretrain(probvlm_path="models/probVLM_conceptual_prefix-040.pth", clipcap_path="models/official_model/clipcap_conceptual_weights.pt", strict_clipcap=False)

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
with open("dataset/dataset_cache/conceptual_test_dataset_11467.pkl", "rb") as f:
    conceptual_test_dataset = pickle.load(f)
    conceptual_test_dataset.prefix_length = agent.prefix_length


coco_test_loader = torch.utils.data.DataLoader(coco_valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
conceptual_test_loader = torch.utils.data.DataLoader(conceptual_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

root_paths = [
    "models/lora_mlp_adapter",
    # "models/lora_mlp_adapter_DER",
    # "models/lora_mlp_adapter_DERPP",    
]

for root_path in root_paths:
    coco_test_loss_list = []
    conceptual_test_loss_list = []
    for i in range(20):
        agent = OneAgent(agent_name='A', buffer_size=1000, device=device)
        agent = agent.to(device)
        print(i)
        agent.load_pretrain(probvlm_path=f"models/probVLM_conceptual_prefix-030.pth", clipcap_path=f"models/official_model/clipcap_conceptual_weights.pt", strict_clipcap=False)
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=2, lora_alpha=32, lora_dropout=0.1, target_modules=["c_fc","c_proj","c_attn"])
        agent.ClipCap.gpt = get_peft_model(agent.ClipCap.gpt, peft_config)
        agent.ClipCap.load_state_dict(torch.load(f"{root_path}/fine_tune_prefix-{i:03}.pt"))

        # calculate test loss for coco
        
        coco_test_loss = 0
        for idx, batch in tqdm(enumerate(coco_test_loader), total=len(coco_test_loader), desc="eval"):
            agent.ClipCap.zero_grad() 
            img, _, _, gpt_token, gpt_mask,_,_ = batch

            img, gpt_token, gpt_mask = img.to(device), gpt_token.to(device), gpt_mask.to(device)

            prefix = agent.CLIP_Net.encode_image(img).to(device, dtype=torch.float32)

            outputs = agent.ClipCap(gpt_token, prefix, gpt_mask)
            logits = outputs.logits[:, coco_test_loader.dataset.prefix_length - 1: -1]

            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), gpt_token.flatten(), ignore_index=0)
            coco_test_loss += loss.item()
        coco_test_loss /= len(coco_test_loader)
        coco_test_loss_list.append(coco_test_loss)

        # calculate test loss for conceptual
        conceptual_test_loss = 0
        for idx, batch in tqdm(enumerate(conceptual_test_loader), total=len(conceptual_test_loader), desc="eval"):
            agent.ClipCap.zero_grad() 
            img, _, _, gpt_token, gpt_mask,_ = batch

            img, gpt_token, gpt_mask = img.to(device), gpt_token.to(device), gpt_mask.to(device)

            prefix = agent.CLIP_Net.encode_image(img).to(device, dtype=torch.float32)

            outputs = agent.ClipCap(gpt_token, prefix, gpt_mask)
            logits = outputs.logits[:, conceptual_test_loader.dataset.prefix_length - 1: -1]

            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), gpt_token.flatten(), ignore_index=0)
            conceptual_test_loss += loss.item()

        conceptual_test_loss /= len(conceptual_test_loader)
        conceptual_test_loss_list.append(conceptual_test_loss)
    
    np.save(f"{root_path}/coco_test_loss.npy", np.array(coco_test_loss_list))
    np.save(f"{root_path}/conceptual_test_loss.npy", np.array(conceptual_test_loss_list))


