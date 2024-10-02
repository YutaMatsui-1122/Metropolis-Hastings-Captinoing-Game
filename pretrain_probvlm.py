import torch
from one_agent import OneAgent
import pickle, clip, argparse
from utils import *
from update_models import *

argparser = argparse.ArgumentParser()
argparser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
argparser.add_argument('--dataset', default="COCO", choices=("COCO", "CC3M"))
argparser.add_argument('--epoch', default=100, type=int)
argparser.add_argument('--batch_size', default=64, type=int)
argparser.add_argument('--num_workers', default=1, type=int)
argparser.add_argument('--cross_modal_lambda_init', default=0.2, type=float)
argparser.add_argument('--cross_modal_lambda_final', default=0.25, type=float)
argparser.add_argument('--annealing_epoch', default=20, type=int)
argparser.add_argument('--device', default="cuda:0")
args = argparser.parse_args()

clip_model, preprocess = clip.load(args.clip_model_type, device="cuda")

if args.dataset == "COCO":
    with open("dataset/dataset_cache/coco_train_dataset.pkl", "rb") as f:
        train_dataset = pickle.load(f)

elif args.dataset == "CC3M":
    with open("dataset/dataset_cache/cc3m-pretrain.pkl", "rb") as f:
        train_dataset = pickle.load(f)

else:
    raise ValueError("Invalid dataset")

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

print("train_dataset:", len(train_dataset))


save_dir = f"models/official_model/probvlm/{args.dataset}"
output_prefix = f"probvlm_{args.cross_modal_lambda_init}_{args.cross_modal_lambda_final}_{args.annealing_epoch}"

agent = OneAgent(agent_name='A')

device = torch.device(args.device)

pretrain_probvlm(agent.CLIP_Net, train_dataloader, agent.ProbVLM_Net, save_dir, args.epoch, device, output_prefix = output_prefix, cross_modal_lambda_init=args.cross_modal_lambda_init, cross_modal_lambda_final=args.cross_modal_lambda_final, start_annealing_epoch=args.annealing_epoch)
# def pretrain_probvlm_step_anneal(CLIP_Net, train_loader, BayesCap_Net, save_dir, epochs, device, lr=1e-4, output_prefix="probvlm", Cri=TempCombLoss(), T1=1e0, T2=5e-2, cross_modal_lambda_init=1e-4, cross_modal_lambda_final=1.0, annealing_epoch=10, train_mode="pretrain"):
    
# pretrain_probvlm_step_anneal(agent.CLIP_Net, train_dataloader, agent.ProbVLM_Net, save_dir, args.epoch, device, output_prefix = output_prefix+"step_anneal", cross_modal_lambda_init=args.cross_modal_lambda_init, cross_modal_lambda_final=args.cross_modal_lambda_final, annealing_epoch=args.annealing_epoch)