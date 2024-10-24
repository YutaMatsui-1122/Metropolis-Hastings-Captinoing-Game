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

args = argparser.parse_args()
device = torch.device(args.device)

if args.pretrain_arch == "ViT-B/32":
    clipcap_path = "models/official_model/clipcap_conceptual_weights.pt"
    probvlm_path = "models/official_model/probvlm/CC3M/probvlm_0.2_0.3_20-epoch-15.pth"
elif args.pretrain_arch == "ViT-B/16":
    clipcap_path = "models/clipcap_vit16_cc3m_2nd/clipcap_018.pt"
    probvlm_path = "models/official_model/probvlm/CC3M/probvlm_0.2_0.2_20_arch_ViT-B-16-epoch-45.pth"

# agent = OneAgent(agent_name='A', device=device, td_update_epochs=10, temperature=0.7, clip_arch="ViT-B/32", td_train_mode="DERPP", td_alpha_beta=args.alpha_beta)
agent = OneAgent(agent_name='A', device=device, td_update_epochs=10, temperature=0.7, clip_arch=args.pretrain_arch, td_train_mode="DERPP", td_alpha_beta=args.alpha_beta)
agent = agent.to(device)
agent.save_dir = f"models/{args.save_dir}"
# agent.load_pretrain(probvlm_path="models/official_model/probvlm/CC3M/probvlm_0.2_0.2_20_arch_ViT-B-16-epoch-45.pth", clipcap_path="models/clipcap_vit16_cc3m/clipcap_009.pt", strict_clipcap=False)
# agent.load_pretrain(probvlm_path="models/official_model/probvlm/CC3M/probvlm_0.2_0.2_20-epoch-69.pth", clipcap_path="models/official_model/clipcap_conceptual_weights.pt", strict_clipcap=False)
print(probvlm_path, clipcap_path)
agent.load_pretrain(probvlm_path=probvlm_path, clipcap_path=clipcap_path, strict_clipcap=False)

finetune_train_file = "coco_10000_cc3m_0_train"
pretrain_train_file = "conceptual_train_dataset_30000"

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

with open(f"dataset/dataset_cache/{finetune_train_file}.pkl", "rb") as f:
    finetune_train_dataset = pickle.load(f)
    finetune_train_dataset.prefix_length = agent.prefix_length

with open(f"dataset/dataset_cache/{pretrain_train_file}.pkl", "rb") as f:
    pretrain_train_dataset = pickle.load(f)
    pretrain_train_dataset.prefix_length = agent.prefix_length

finetune_train_dataloader = torch.utils.data.DataLoader(finetune_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
finetune_train_dataloader_fix = torch.utils.data.DataLoader(finetune_train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

pretrain_train_dataloader = torch.utils.data.DataLoader(pretrain_train_dataset, batch_size=100, shuffle=False, num_workers=args.num_workers)

agent.communication_field_setup(finetune_train_dataloader_fix, finetune_train_dataloader, MH_iter=10)
agent.lora_setting()
agent.perception()
agent.initialize_td_buffer(pretrain_train_dataloader, buffer_size=10000)

agent.save_dir = f"models/{args.save_dir}/"
os.makedirs(f"models/{args.save_dir}", exist_ok=True)

current_caption = [agent.dataloader_MHNG_fix.dataset.dataset[i]["caption"] for i in range(len(agent.dataloader_MHNG_fix.dataset))]
current_gpt_token = [agent.dataloader_MHNG_fix.dataset.dataset[i]["gpt_token"] for i in range(len(agent.dataloader_MHNG_fix.dataset))]

print(current_caption[:50], current_caption[-50:])
print(current_gpt_token[:50], current_gpt_token[-50:])

agent.update_text_decoder(0)