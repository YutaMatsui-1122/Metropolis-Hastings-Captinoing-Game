# # Evaluate the communication between the agents in the vlm captioning game
import os, torch
import json
from PIL import Image
import pandas as pd
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pickle, clip, argparse
from tqdm import tqdm

from one_agent import OneAgent
from utils import *
from transformers import GPT2Tokenizer

if __name__ == '__main__':
    # argparseを使ってコマンドライン引数を取得
    parser = argparse.ArgumentParser(description='PAC-S evaluation')
    parser.add_argument('--exp_name', type=str, default='default')
    parser.add_argument('--dataset_name', type=str, default='coco_all', choices=['coco_a', 'coco_b', 'coco_all'])
    parser.add_argument('--mode', type=str, default='eval', choices=['eval', 'train'])
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--em_iter', default=0, type=int)
    parser.add_argument('--temperature', default=0.7, type=float)
    parser.add_argument('--use_official_model', action='store_true', default=False)

    args = parser.parse_args()

    dataset_name = args.dataset_name

    # データ変換（例として、画像サイズを256x256にリサイズ）
    clip_model, transform = clip.load('ViT-B/32', jit=False)
    if args.mode == 'eval':
        if dataset_name == 'coco_a':
            file_path = "dataset/dataset_cache/coco_2017_val_split_dataset_a.pkl"
        elif dataset_name == 'coco_b':
            file_path = "dataset/dataset_cache/coco_2017_val_split_dataset_b.pkl"
        elif dataset_name == 'coco_all':
            file_path = "dataset/dataset_cache/coco_2017_val_split_dataset_whole.pkl"
    elif args.mode == 'train':
        if dataset_name == 'coco_a':
            file_path = "dataset/dataset_cache/coco_2017_train_split_dataset_a.pkl"
        elif dataset_name == 'coco_b':
            file_path = "dataset/dataset_cache/coco_2017_train_split_dataset_b.pkl"

    with open(file_path, 'rb') as f:
        train_dataset = pickle.load(f)
        train_dataset.prefix_length = 10

    data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"Number of images in the dataset: {len(train_dataset)}")
    # 0.2. Define the device
    device = torch.device(args.device)

    # 0.3. Set the experiment name, directory, and the dataset
    exp_name = args.exp_name
    exp_dir = f"exp/{exp_name}"
    exp_eval_dir = f"exp_eval/{exp_name}/{dataset_name}"
    os.makedirs(exp_eval_dir, exist_ok=True)
    temperature = args.temperature

    use_official_model = args.use_official_model

    agent_clip_arch = {"A": "ViT-B/16", "B": "ViT-B/32"}

    for epoch in [29, 10]:
        for agent_name in ["A", "B"]:
            agent = OneAgent(agent_name=agent_name, device=device,temperature=temperature, clip_arch=agent_clip_arch[agent_name])
            agent = agent.to(device)

            if use_official_model:
                exp_name = "pretrain"
                exp_eval_dir = f"exp_eval/{exp_name}"
                os.makedirs(exp_eval_dir, exist_ok=True)
                
                if agent_name == 'A':
                    # agent.load_pretrain(probvlm_path="models/official_model/probvlm/CC3M/probvlm_0.2_0.3_20-epoch-15.pth", clipcap_path="models/official_model/clipcap_conceptual_weights.pt", strict_clipcap=False)
                    agent.load_pretrain(probvlm_path="models/official_model/probvlm/COCO_A/probvlm_0.2_0.2_20_arch_ViT-B-16-epoch-49.pth", clipcap_path=f"models/coco_dataset_a_vit16/clipcap_019.pt", strict_clipcap=False)
                    candidate_path = f"{exp_eval_dir}/{dataset_name}_candidate_a_temperature_{temperature}_vit16_{args.mode}.json"
                else:
                    candidate_path = f"{exp_eval_dir}/{dataset_name}_candidate_b_temperature_{temperature}_vit32_{args.mode}.json"
                    agent.load_pretrain(probvlm_path="models/official_model/probvlm/COCO_B/probvlm_0.2_0.2_20_arch_ViT-B-32-epoch-49.pth", clipcap_path="models/coco_dataset_b_vit32/clipcap_019.pt", strict_clipcap=False)
            else:
                candidate_path = f"{exp_eval_dir}/{dataset_name}_candidate_{agent_name}_epoch_{epoch}_temperature_{temperature}_{args.mode}.json"
                agent.lora_setting()
                agent.load_pretrain(probvlm_path=f"{exp_dir}/{agent_name}/probvlm_{agent_name}_{epoch}-epoch-9.pth", clipcap_path=f"{exp_dir}/{agent_name}/clipcap_{agent_name}_{epoch}-009.pt")

            print(f"EM iter {args.em_iter} , Candidate path {candidate_path}")

            candidate = {}
            for j, data in enumerate(tqdm(data_loader)):
                image = data["image"].to(device)
                captions = data["caption"]
                index = data["index"]
                filenames = [train_dataset.images[i] for i in index]

                mu_img, alpha_img, sigma_img, z = agent.image_encoder(image)
                if use_official_model:
                    captions = agent.text_decoder(z)
                else:
                    captions = agent.text_decoder(mu_img)

                for caption, filename in zip(captions, filenames):
                    filename = filename.split('/')[-1].split('.')[0]
                    candidate[filename] = caption
                if j % 100 == 0:
                    print(filenames[:3])
                    print(captions[:3])


            # save the candidate 
            # print length of the candidate
            print(f"Number of captions in the candidate: {len(candidate)}")
            
            with open(candidate_path, 'w') as f:
                json.dump(candidate, f, indent=4)
            
            print(f"Saved the candidate to {candidate_path}")
        
        if args.use_official_model:
            break

        