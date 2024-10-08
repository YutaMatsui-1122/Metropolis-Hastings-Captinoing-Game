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
argparser.add_argument('--speaker_agent', default="None", choices=("None", "coco", "conceptual"))
argparser.add_argument('--alpha_beta', type=float, default=0.5)

import pandas as pd
sign_df = pd.read_csv("exp/mhcg_derpp_1/A/agent_A_sign.csv")
print(sign_df["EM_0_MH_9"].tolist()[:10])

args = argparser.parse_args()

os.makedirs(f"models/{args.save_dir}", exist_ok=True)

device = torch.device(args.device)

agent = OneAgent(agent_name='A', device=device, td_update_epochs=10, der_alpha=args.alpha_beta, derpp_beta=args.alpha_beta, temperature=0.7)
agent = agent.to(device)

agent.save_dir = f"models/{args.save_dir}"

# args.speaker_agentの逆のエージェントのパラメータをロード
if args.speaker_agent == "None" or args.speaker_agent == "coco":
    agent.load_pretrain(probvlm_path="models/official_model/probvlm/CC3M/probvlm_0.2_0.3_20-epoch-15.pth", clipcap_path="models/official_model/clipcap_conceptual_weights.pt", strict_clipcap=False)
    finetune_train_file = "communication_coco_5000_cc3m_5000"
    pretrain_train_file = "conceptual_train_dataset_30000"
else:
    agent.load_pretrain(probvlm_path="models/official_model/probvlm/COCO/probvlm_0.2_0.3_20-epoch-99.pth", clipcap_path="models/official_model/clipcap_coco_weights.pt", strict_clipcap=False)
    finetune_train_file =  f"communication_coco_5000_cc3m_5000"
    pretrain_train_file = "coco_train_dataset_30000"
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

for em_epoch in range(10):
    os.makedirs(f"models/{args.save_dir}/EM_{em_epoch}", exist_ok=True)
    agent.save_dir = f"models/{args.save_dir}/EM_{em_epoch}"
    previous_caption = [finetune_train_dataset.dataset[i]["caption"] for i in range(len(finetune_train_dataset))]
    previous_caption = [agent.dataloader_MHNG_fix.dataset.dataset[i]["caption"] for i in range(len(agent.dataloader_MHNG_fix.dataset))]
    previous_gpt_token = [agent.dataloader_MHNG_fix.dataset.dataset[i]["gpt_token"] for i in range(len(agent.dataloader_MHNG_fix.dataset))]
    print(previous_caption[:3])
    print(previous_gpt_token[:3])

    for i in range(len(agent.dataloader_MHNG_fix.dataset)):
        agent.dataloader_MHNG_fix.dataset.dataset[i]["caption"] = sign_df[f"EM_{em_epoch}_MH_9"][i]
        agent.dataloader_MHNG_fix.dataset.dataset[i]["gpt_token"] = torch.tensor(agent.tokenizer.encode(sign_df[f"EM_{em_epoch}_MH_9"][i]))
    
    current_caption = [agent.dataloader_MHNG_fix.dataset.dataset[i]["caption"] for i in range(len(agent.dataloader_MHNG_fix.dataset))]
    current_gpt_token = [agent.dataloader_MHNG_fix.dataset.dataset[i]["gpt_token"] for i in range(len(agent.dataloader_MHNG_fix.dataset))]

    agent.update_text_decoder(em_epoch)