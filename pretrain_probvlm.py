import torch
from one_agent import OneAgent
import pickle, clip, argparse
from utils import *



argparser = argparse.ArgumentParser()
argparser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
argparser.add_argument('--dataset', default="coco", choices=('coco', 'conceptual'))
argparser.add_argument('--exp_name', default="debug")
args = argparser.parse_args()

clip_model, preprocess = clip.load(args.clip_model_type, device="cuda")

if args.dataset == "coco":
    with open("dataset/dataset_cache/coco_train_dataset.pkl", "rb") as f:
        train_dataset = pickle.load(f)
    with open("dataset/dataset_cache/coco_test_dataset.pkl", "rb") as f:
        test_dataset = pickle.load(f)
elif args.dataset == "conceptual":
    with open("dataset/dataset_cache/conceptual_captions_subset_train_dataset.pkl", "rb") as f:
        train_dataset = pickle.load(f)
    with open("dataset/dataset_cache/conceptual_captions_subset_test_dataset.pkl", "rb") as f:
        test_dataset = pickle.load(f)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

print("train_dataset:", len(train_dataset))

agent = OneAgent(agent_name='A')
agent = agent.cuda()
agent.pretrain_probvlm(train_dataloader, test_dataloader, output_prefix="probVLM_"+args.dataset+"_prefix")