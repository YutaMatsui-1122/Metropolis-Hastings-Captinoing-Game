import os, torch
import json
from PIL import Image
import pandas as pd
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
import pickle, clip, argparse
from tqdm import tqdm
import time

from one_agent import OneAgent
from utils import *
from ensemble_utils import *

# COCOデータセット用のカスタムデータセットクラス
class CocoDataset(Dataset):
    def __init__(self, annotation_file, img_dir, transform=None):
        self.coco = COCO(annotation_file)
        self.img_dir = img_dir
        self.transform = transform
        self.img_ids = self.coco.getImgIds()

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        return image, img_info['file_name']

if __name__ == '__main__':
    coco_annotation_file = 'dataset/coco/annotations/captions_val2017.json'
    coco_images_dir = 'dataset/coco/val2017'
    nocap_annotation_file = 'dataset/nocaps/nocaps_val_4500_captions.json'
    nocap_images_dir = 'dataset/nocaps/validation'
    cc3m_annotation_file = 'dataset/conceptual_captions/annotations/validation_imgtxt.tsv'

    # argparseを使ってコマンドライン引数を取得
    parser = argparse.ArgumentParser(description='PAC-S evaluation')
    parser.add_argument('--dataset_name', type=str, default='coco_all', choices=['coco_a', 'coco_b', 'coco_all'])
    parser.add_argument('--mode', type=str, default='eval', choices=['eval', 'train', 'mhcg'])
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--temperature', default=0.7, type=float)
    parser.add_argument('--num_samples', default=1, type=int)
    parser.add_argument('--dataset_prefix', default="coco_2017_common_person_only", type=str)
    parser.add_argument('--ensemble_method', default="ensemble", choices=['ensemble', 'packllm_sim'])
    
    args = parser.parse_args()

    dataset_name = args.dataset_name

    # データ変換（例として、画像サイズを256x256にリサイズ）
    clip_model, transform = clip.load('ViT-B/32', jit=False)

    if args.mode == 'eval':
        if dataset_name == 'coco_a':
            file_path = f"dataset/dataset_cache/{args.dataset_prefix}/val_split_dataset_a.pkl"
        elif dataset_name == 'coco_b':
            file_path = f"dataset/dataset_cache/{args.dataset_prefix}/val_split_dataset_b.pkl"
        elif dataset_name == 'coco_all':
            coco_annotation_file = 'dataset/coco/annotations/captions_val2017.json'
            coco_images_dir = 'dataset/coco/val2017'
            train_dataset = CocoDataset(annotation_file=coco_annotation_file, img_dir=coco_images_dir, transform=transform)
    elif args.mode == 'train':
        if dataset_name == 'coco_a':
            file_path = f"dataset/dataset_cache/{args.dataset_prefix}/train_split_dataset_a.pkl"
        elif dataset_name == 'coco_b':
            file_path = f"dataset/dataset_cache/{args.dataset_prefix}/train_split_dataset_b.pkl"
        elif dataset_name == 'coco_all':
            file_path = f"dataset/dataset_cache/{args.dataset_prefix}/train_split_dataset_all.pkl"

    if dataset_name != 'coco_all':
        with open(file_path, 'rb') as f:
            train_dataset = pickle.load(f)
            train_dataset.prefix_length = 10

    data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device(args.device)

    temperature = args.temperature

    exp_name = args.ensemble_method
    exp_eval_dir = f"exp_eval/{exp_name}/{dataset_name}"
    os.makedirs(exp_eval_dir, exist_ok=True)

    agent_clip_arch = {"A": "ViT-B/16", "B": "ViT-B/32", "AB": "ViT-B/32"}

    agentA = OneAgent(agent_name='A', device=device, temperature=temperature, clip_arch=agent_clip_arch['A'])
    agentB = OneAgent(agent_name='B', device=device, temperature=temperature, clip_arch=agent_clip_arch['B'])
    agentA.load_pretrain(probvlm_path=f"pretrain_models/{args.dataset_prefix}/COCO_A/probvlm/probvlm-epoch-49.pth", 
                         clipcap_path=f"pretrain_models/{args.dataset_prefix}/COCO_A/clipcap/clipcap_019.pt", strict_clipcap=False)
    
    agentB.load_pretrain(probvlm_path=f"pretrain_models/{args.dataset_prefix}/COCO_B/probvlm/probvlm-epoch-49.pth", 
                         clipcap_path=f"pretrain_models/{args.dataset_prefix}/COCO_B/clipcap/clipcap_019.pt", strict_clipcap=False)
    agentA = agentA.to(device)
    agentB = agentB.to(device)
    candidate = {}

    for num_sample in range(args.num_samples):
        candidate_path = f"{exp_eval_dir}/{args.dataset_prefix}_candidate_AB_temperature_{temperature}_{args.mode}.json"
        sum_time = 0

        if args.num_samples > 1:
            candidate_path = candidate_path.replace(".json", f"_{num_sample}.json")

        for j, data in enumerate(tqdm(data_loader)):            
            start = time.time()

            if args.dataset_name == 'coco_all':
                image = data[0].to(device)
                filenames = data[1]
            else:
                image = data["image"].to(device)
                # captions = data["caption"]
                index = data["index"]
                filenames = [train_dataset.dataset[i]["image"] for i in index]

            mu_img_A, alpha_img_A, sigma_img_A, z_A = agentA.image_encoder(image)
            mu_img_B, alpha_img_B, sigma_img_B, z_B = agentB.image_encoder(image)

            prefix_embeds_A = agentA.ClipCap.clip_project(z_A.float()).reshape(z_A.shape[0], agentA.prefix_length, -1)
            prefix_embeds_B = agentB.ClipCap.clip_project(z_B.float()).reshape(z_B.shape[0], agentB.prefix_length, -1)

            captions = ensemble_generate_batch(agentA.ClipCap, agentB.ClipCap, agentA.tokenizer, embed1=prefix_embeds_A, embed2=prefix_embeds_B, temperature=temperature, fusion_method=args.ensemble_method)

            for caption, filename in zip(captions, filenames):
                filename = filename.split('/')[-1].split('.')[0]
                candidate[filename] = caption
            if j % 100 == 0:
                print(captions[:3])

            end = time.time()
            sum_time += end - start


        # save the candidate 
        # print length of the candidate
        print(f"Number of captions in the candidate: {len(candidate)}")

        print(f"Average time per batch: {sum_time / len(data_loader)}")
        print(f"Total time: {sum_time}")
        
        with open(candidate_path, 'w') as f:
            json.dump(candidate, f, indent=4)
        
        print(f"Saved the candidate to {candidate_path}")


