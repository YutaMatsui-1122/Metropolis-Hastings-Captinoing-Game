import torch
from one_agent import OneAgent
import pickle, clip, argparse
from utils import *
from ProbVLM.src.losses import *
import pandas as pd
from update_models import *

argparser = argparse.ArgumentParser()
argparser.add_argument('--num_workers', type=int, default=1)
argparser.add_argument('--batch_size', type=int, default=64)
argparser.add_argument('--save_dir', default="debug/")
argparser.add_argument('--device', default="cuda:0")
argparser.add_argument('--alpha_beta', type=float, default=0.5)
argparser.add_argument('--pretrain_arch', default="ViT-B/32", choices=('ViT-B/32', 'ViT-B/16', ))
argparser.add_argument('--td_update_epochs', default=10, type=int)

args = argparser.parse_args()
device = torch.device(args.device)

if args.pretrain_arch == "ViT-B/32":
    clipcap_path = "models/official_model/clipcap_coco_weights.pt"
    probvlm_path = "models/official_model/probvlm/COCO/probvlm_0.2_0.3_20-epoch-15.pth"
elif args.pretrain_arch == "ViT-B/16":
    clipcap_path = "models/clipcap_vit16_cc3m/clipcap_009.pt"
    probvlm_path = "models/official_model/probvlm/CC3M/probvlm_0.2_0.2_20_arch_ViT-B-16-epoch-45.pth"

# agent_cc3m = OneAgent(agent_name='A', device=device, td_update_epochs=10, temperature=0.7, clip_arch="ViT-B/32", td_train_mode="DERPP", td_alpha_beta=args.alpha_beta)
agent_cc3m = OneAgent(agent_name='A', device=device, td_update_epochs=args.td_update_epochs, temperature=0.7, clip_arch=args.pretrain_arch, td_train_mode="DERPP", td_alpha_beta=args.alpha_beta)
agent_cc3m = agent_cc3m.to(device)
agent_cc3m.save_dir = f"models/{args.save_dir}"
agent_coco = OneAgent(agent_name='A', device=device, td_update_epochs=args.td_update_epochs, temperature=0.63, clip_arch="ViT-B/32", td_train_mode="DERPP", td_alpha_beta=args.alpha_beta)
agent_coco = agent_coco.to(device)
agent_coco.save_dir = f"models/{args.save_dir}"
# agent_cc3m.load_pretrain(probvlm_path="models/official_model/probvlm/CC3M/probvlm_0.2_0.2_20_arch_ViT-B-16-epoch-45.pth", clipcap_path="models/clipcap_vit16_cc3m/clipcap_009.pt", strict_clipcap=False)
# agent_cc3m.load_pretrain(probvlm_path="models/official_model/probvlm/CC3M/probvlm_0.2_0.2_20-epoch-69.pth", clipcap_path="models/official_model/clipcap_conceptual_weights.pt", strict_clipcap=False)
print(probvlm_path, clipcap_path)
agent_cc3m.load_pretrain(probvlm_path=probvlm_path, clipcap_path=clipcap_path, strict_clipcap=False)
agent_coco.load_pretrain(probvlm_path="models/official_model/probvlm/COCO/probvlm_0.2_0.3_20-epoch-99.pth", clipcap_path="models/official_model/clipcap_coco_weights.pt", strict_clipcap=False)

def print_weight_statistics(model):
    total_params = 0
    total_mean = 0
    total_max = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:  # 学習中の重みのみを対象にする
            mean = param.data.abs().mean().item()
            max_val = param.data.abs().max().item()
            total_mean += mean
            total_max = max(total_max, max_val)
            total_params += param.numel()

            print(f"Layer: {name} | Mean: {mean:.4f} | Max Abs: {max_val:.4f}")
    
    print(f"Total Mean: {total_mean / total_params:.4f}")
    print(f"Total Max Absolute Value: {total_max:.4f}")

# print_weight_statistics(agent_cc3m.ClipCap)

if "debug" in args.save_dir:
    finetune_train_file = "communication_coco_50_cc3m_50"
    pretrain_train_file = "conceptual_train_dataset_30000"
    buffer_size = 50
else:
    finetune_train_file = "coco_10000_cc3m_0_train"
    pretrain_train_file = "conceptual_train_dataset_30000"
    buffer_size = 10000

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

with open(f"dataset/dataset_cache/{finetune_train_file}.pkl", "rb") as f:
    finetune_train_dataset = pickle.load(f)
    finetune_train_dataset.prefix_length = agent_cc3m.prefix_length

with open(f"dataset/dataset_cache/{pretrain_train_file}.pkl", "rb") as f:
    pretrain_train_dataset = pickle.load(f)
    pretrain_train_dataset.prefix_length = agent_cc3m.prefix_length

finetune_train_dataloader = torch.utils.data.DataLoader(finetune_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
finetune_train_dataloader_fix = torch.utils.data.DataLoader(finetune_train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

pretrain_train_dataloader = torch.utils.data.DataLoader(pretrain_train_dataset, batch_size=100, shuffle=False, num_workers=args.num_workers)

print(len(finetune_train_dataset), len(pretrain_train_dataset))

agent_coco.communication_field_setup(finetune_train_dataloader_fix, finetune_train_dataloader, MH_iter=10)
agent_coco.perception()
agent_cc3m.communication_field_setup(finetune_train_dataloader_fix, finetune_train_dataloader, MH_iter=10)
agent_cc3m.perception()
agent_cc3m.lora_setting()
agent_cc3m.initialize_td_buffer(pretrain_train_dataloader, buffer_size=buffer_size)

agent_cc3m.save_dir = f"models/{args.save_dir}/"
os.makedirs(f"models/{args.save_dir}", exist_ok=True)

for epoch in range(10):
    _, coco_caption= agent_coco.propose(return_caption=True)
    # lower case
    coco_caption = [caption.lower() for caption in coco_caption]
    # coco_gpt_token = torch.tensor(agent_coco.tokenizer.encode(coco_caption))
    coco_gpt_token = [torch.tensor(agent_coco.tokenizer.encode(caption)) for caption in coco_caption]
    print(coco_caption[:10], coco_gpt_token[:10])
    for i in range(len(agent_cc3m.dataloader_MHNG_fix.dataset)):
        agent_cc3m.dataloader_MHNG_fix.dataset.dataset[i]["caption"] = coco_caption[i]
        agent_cc3m.dataloader_MHNG_fix.dataset.dataset[i]["gpt_token"] = coco_gpt_token[i]

    agent_cc3m.update_text_decoder(0)