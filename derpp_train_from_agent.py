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
argparser.add_argument('--initial_epoch', type=int, default=0)
argparser.add_argument('--num_workers', type=int, default=1)
argparser.add_argument('--batch_size', type=int, default=64)
argparser.add_argument('--lr', type=float, default=2e-5)
argparser.add_argument('--save_dir', default="debug/")
argparser.add_argument('--device', default="cuda:0")
argparser.add_argument('--cl_mode', default="None", choices=('None','DER', 'DERPP', 'ER', 'ER_RS'))
argparser.add_argument('--reservoir_false', action='store_false', dest='reservoir')
argparser.add_argument('--speaker_agent', default="None", choices=("None", "coco", "conceptual"))
argparser.add_argument('--use_generated_caption', action='store_true', dest='use_generated_caption')

args = argparser.parse_args()

os.makedirs(f"models/{args.save_dir}", exist_ok=True)

device = torch.device(args.device)

clip_model, preprocess = clip.load(args.clip_model_type, device=device)

agent = OneAgent(agent_name='A', device=device, td_update_epochs=100)
agent = agent.to(device)

# args.speaker_agentの逆のエージェントのパラメータをロード
if args.speaker_agent == "None" or args.speaker_agent == "coco":
    agent.load_pretrain(probvlm_path="models/probVLM_conceptual_prefix-035.pth", clipcap_path="models/official_model/clipcap_conceptual_weights.pt", strict_clipcap=False)
    finetune_train_file = "coco_train_dataset_10000"
    finetune_test_file = "coco_test_dataset_5000"
    pretrain_train_file = "conceptual_train_dataset_30000"
    pretrain_test_file = "conceptual_test_dataset_11467"
    der_alpha = 0.5
    derpp_beta = 0.5
else:
    agent.load_pretrain(probvlm_path="models/probVLM_coco_prefix-035.pth", clipcap_path="models/official_model/clipcap_coco_weights.pt", strict_clipcap=False)
    finetune_train_file =  "coco_train_dataset_10000"
    finetune_test_file = "conceptual_test_dataset_11467"
    pretrain_train_file = "coco_train_dataset_30000"
    pretrain_test_file = "coco_test_dataset_5000"
    der_alpha = 0.5
    derpp_beta = 0.5

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

with open(f"dataset/dataset_cache/{finetune_train_file}.pkl", "rb") as f:
    finetune_train_dataset = pickle.load(f)
    finetune_train_dataset.prefix_length = agent.prefix_length
with open(f"dataset/dataset_cache/{finetune_test_file}.pkl", "rb") as f:
    finetune_valid_dataset = pickle.load(f)
    finetune_valid_dataset.prefix_length = agent.prefix_length

with open(f"dataset/dataset_cache/{pretrain_train_file}.pkl", "rb") as f:
    pretrain_train_dataset = pickle.load(f)
    pretrain_train_dataset.prefix_length = agent.prefix_length
with open(f"dataset/dataset_cache/{pretrain_test_file}.pkl", "rb") as f:
    pretrain_test_dataset = pickle.load(f)
    pretrain_test_dataset.prefix_length = agent.prefix_length

if args.speaker_agent == "coco" and args.use_generated_caption:
    captions_df = pd.read_csv(f"models/mh_chain/mh_chain_coco2conceptual.csv")
    captions = captions_df["Generated_99"].tolist()
    finetune_train_dataset = set_caption_to_dataset(finetune_train_dataset, captions=captions)
    print("chain_last_captions",finetune_train_dataset.captions[:10])

elif args.speaker_agent == "conceptual" and args.use_generated_caption:
    captions_df = pd.read_csv(f"models/mh_chain/mh_chain_conceptual2coco_0.9.csv")
    captions = captions_df["Generated_99"].tolist()
    finetune_train_dataset = set_caption_to_dataset(finetune_train_dataset, captions=captions)

finetune_train_dataloader = torch.utils.data.DataLoader(finetune_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
finetune_train_dataloader_fix = torch.utils.data.DataLoader(finetune_train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
finetune_test_dataloader = torch.utils.data.DataLoader(finetune_valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

pretrain_train_dataloader = torch.utils.data.DataLoader(pretrain_train_dataset, batch_size=100, shuffle=False, num_workers=args.num_workers)
pretrain_test_dataloader = torch.utils.data.DataLoader(pretrain_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

print("finetune_caption:", finetune_train_dataset.captions[:10])

agent.communication_field_setup(finetune_train_dataloader_fix, finetune_train_dataloader,10)
agent.initialize_td_buffer(pretrain_train_dataloader, buffer_size=10000)
agent.save_dir = f"models/{args.save_dir}"

agent.perception()
agent.update_text_decoder(0)


