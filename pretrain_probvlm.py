import torch
from one_agent import OneAgent
import pickle, clip, argparse
from utils import *
from update_models import *

argparser = argparse.ArgumentParser()
argparser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32', 'ViT-B/16', ))
argparser.add_argument('--dataset', default="COCO", choices=("COCO", "CC3M"))
argparser.add_argument('--epoch', default=100, type=int)
argparser.add_argument('--batch_size', default=64, type=int)
argparser.add_argument('--num_workers', default=1, type=int)
argparser.add_argument('--cross_modal_lambda_init', default=0.2, type=float)
argparser.add_argument('--cross_modal_lambda_final', default=0.2, type=float)
argparser.add_argument('--annealing_epoch', default=20, type=int)
argparser.add_argument('--device', default="cuda:0")
args = argparser.parse_args()

if args.dataset == "COCO":
    with open("dataset/dataset_cache/coco_train.pkl", "rb") as f:
        train_dataset = pickle.load(f)

elif args.dataset == "CC3M":
    with open("dataset/dataset_cache/cc3m_train.pkl", "rb") as f:
        train_dataset = pickle.load(f)

else:
    raise ValueError("Invalid dataset")

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

print("train_dataset:", len(train_dataset))


save_dir = f"models/official_model/probvlm/{args.dataset}"
output_prefix = f"probvlm_{args.cross_modal_lambda_init}_{args.cross_modal_lambda_final}_{args.annealing_epoch}_arch_{args.clip_model_type}"

# test_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

# batch = next(iter(test_dataloader))
# image = batch["image"].to(args.device)
# vlm_token = batch["vlm_token"].to(args.device)


# import time

# for clip_arch in ["ViT-B/32", "ViT-B/16",]:
#     agent = OneAgent(agent_name='A', device="cuda", temperature=0.62, clip_arch=clip_arch)
#     device = torch.device(args.device)

#     s = time.time()
#     # print embedding from image and vlm_token
#     print("image:", image.shape)
#     print("vlm_token:", vlm_token.shape)
#     print("image_embedding:", agent.CLIP_Net.encode_image(image).shape)
#     print("vlm_token_embedding:", agent.CLIP_Net.encode_text(vlm_token).shape)
#     print("image_embedding:", agent.CLIP_Net.encode_image(image)[0, :10])
#     print("vlm_token_embedding:", agent.CLIP_Net.encode_text(vlm_token)[0, :10])
#     print("time:", time.time()-s)

agent = OneAgent(agent_name='A', device=args.device, temperature=0.62, clip_arch=args.clip_model_type)
    

device = torch.device(args.device)


pretrain_probvlm(agent.CLIP_Net, train_dataloader, agent.ProbVLM_Net, save_dir, args.epoch, device, output_prefix = output_prefix, cross_modal_lambda_init=args.cross_modal_lambda_init, cross_modal_lambda_final=args.cross_modal_lambda_final, start_annealing_epoch=args.annealing_epoch)
# def pretrain_probvlm_step_anneal(CLIP_Net, train_loader, BayesCap_Net, save_dir, epochs, device, lr=1e-4, output_prefix="probvlm", Cri=TempCombLoss(), T1=1e0, T2=5e-2, cross_modal_lambda_init=1e-4, cross_modal_lambda_final=1.0, annealing_epoch=10, train_mode="pretrain"):
    
# pretrain_probvlm_step_anneal(agent.CLIP_Net, train_dataloader, agent.ProbVLM_Net, save_dir, args.epoch, device, output_prefix = output_prefix+"step_anneal", cross_modal_lambda_init=args.cross_modal_lambda_init, cross_modal_lambda_final=args.cross_modal_lambda_final, annealing_epoch=args.annealing_epoch)