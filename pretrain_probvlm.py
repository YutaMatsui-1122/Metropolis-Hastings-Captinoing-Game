import torch
from one_agent import OneAgent
import pickle, clip, argparse
from utils import *
from update_models import *

argparser = argparse.ArgumentParser()
argparser.add_argument('--clip_model_type', default="ViT-B/32", choices=('ViT-B/32', 'ViT-B/16',))
argparser.add_argument('--dataset', default="COCO", choices=("COCO", "CC3M", "COCO_A", "COCO_B"))
argparser.add_argument('--epoch', default=50, type=int)
argparser.add_argument('--batch_size', default=64, type=int)
argparser.add_argument('--num_workers', default=1, type=int)
argparser.add_argument('--cross_modal_lambda_init', default=0.2, type=float)
argparser.add_argument('--cross_modal_lambda_final', default=0.2, type=float)
argparser.add_argument('--annealing_epoch', default=20, type=int)
argparser.add_argument('--device', default="cuda:0")
argparser.add_argument('--prefix', default="coco_2017_common_person_only")
args = argparser.parse_args()

debug = False

if args.dataset == "COCO_A":
    with open(f"dataset/dataset_cache/{args.prefix}/train_split_dataset_a.pkl", "rb") as f:
        train_dataset = pickle.load(f)
        train_dataset.prefix_length = 10

elif args.dataset == "COCO_B":
    # with open("dataset/dataset_cache/coco_split_dataset_b.pkl", "rb") as f:
    with open(f"dataset/dataset_cache/{args.prefix}/train_split_dataset_b.pkl", "rb") as f:
        train_dataset = pickle.load(f)
        train_dataset.prefix_length = 10

else:
    raise ValueError("Invalid dataset")

while True:

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    print("train_dataset:", len(train_dataset))

    save_dir = os.path.join("pretrain_models", args.prefix, args.dataset, f"probvlm")
    os.makedirs(save_dir, exist_ok=True)

    output_prefix = f"probvlm"
    print("output_prefix:", output_prefix)

    agent = OneAgent(agent_name='A', device=args.device, clip_arch=args.clip_model_type)
        
    device = torch.device(args.device)
    
    # pretrain_probvlm(agent.CLIP_Net, train_dataloader, agent.ProbVLM_Net, save_dir, args.epoch, device, output_prefix = output_prefix, cross_modal_lambda_init=args.cross_modal_lambda_init, cross_modal_lambda_final=args.cross_modal_lambda_final, start_annealing_epoch=args.annealing_epoch)
    _, _, loss_i, loss_t, _, _ = pretrain_probvlm(agent.CLIP_Net, train_dataloader, agent.ProbVLM_Net, save_dir, args.epoch, device, output_prefix = output_prefix, cross_modal_lambda_init=args.cross_modal_lambda_init, cross_modal_lambda_final=args.cross_modal_lambda_final, start_annealing_epoch=args.annealing_epoch)

    if loss_i < 0 and loss_t < 0:
        break