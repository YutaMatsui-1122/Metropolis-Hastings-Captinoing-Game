# # Evaluate the communication between the agents in the vlm captioning game
import os, torch
import json
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
import pickle, clip, argparse
from tqdm import tqdm
import time

from one_agent import OneAgent
from utils import *
from safetensors.torch import load_file  # safetensors 用の読み込み
from model_merge_setting import HuggingFaceClipCaptionModel
from huggingface_hub import hf_hub_download

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
    # argparseを使ってコマンドライン引数を取得
    parser = argparse.ArgumentParser(description='PAC-S evaluation')
    parser.add_argument('--exp_name', type=str, default='mhcg_person_only_0')
    parser.add_argument('--dataset_name', type=str, default='coco_all', choices=['coco_a', 'coco_b', 'coco_all'])
    parser.add_argument('--mode', type=str, default='eval', choices=['eval', 'train'])
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--epochs', nargs='+', type=int, default=[0])
    parser.add_argument('--num_samples', default=1, type=int)
    parser.add_argument('--temperature', default=0.7, type=float)
    parser.add_argument('--dataset_prefix', type=str, default='coco_2017_common_person_only')

    args = parser.parse_args()


    dataset_name = args.dataset_name

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

    if dataset_name != 'coco_all':
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

    agent_clip_arch = {"A": "ViT-B/16", "B": "ViT-B/32", "ALL": "ViT-B/32"}

    if "pretrain" in exp_name:
        agent_names = ["A", "B", "ALL"]
    elif "merge" in exp_name:
        agent_names = ["A"]
    else:
        agent_names = ["A", "B"]
    
    for num_sample in range(args.num_samples):
        if args.num_samples > 1:
            print(f"Sample {num_sample}")
        
        total_time = 0
        total_time_for_encoding = 0
        
        for epoch in args.epochs:
            for agent_name in agent_names:
                agent = OneAgent(agent_name=agent_name, device=device,temperature=temperature, clip_arch=agent_clip_arch[agent_name])
                if agent_name == 'A':
                    pretrain_probvlm_path = f"pretrain_models/{args.dataset_prefix}/COCO_A/probvlm/probvlm-epoch-49.pth"
                elif agent_name == 'B' or agent_name == 'ALL':
                    pretrain_probvlm_path = f"pretrain_models/{args.dataset_prefix}/COCO_B/probvlm/probvlm-epoch-49.pth"

                if "pretrain" in exp_name:
                    if agent_name == 'ALL':
                        agent.load_pretrain(probvlm_path=pretrain_probvlm_path,
                                            clipcap_path=f"pretrain_models/COCO_All/clipcap_coco_weights.pt", strict_clipcap=False)
                    else:
                        agent.load_pretrain(probvlm_path=pretrain_probvlm_path,
                                            clipcap_path=f"pretrain_models/{args.dataset_prefix}/COCO_{agent_name}/clipcap/clipcap_019.pt", strict_clipcap=False)
                    
                elif "all_acceptance" in exp_name:
                    # load exp args
                    with open(f"{exp_dir}/args.json", 'r') as f:
                        exp_args = json.load(f)
                    lora_r = exp_args["lora_r"]
                    lora_alpha = exp_args["lora_alpha"]
                    lora_dropout = exp_args["lora_dropout"]
                    agent.lora_setting(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, probvlm=False,)
                    clipcap_path = f"{exp_dir}/{agent_name}/clipcap_{agent_name}_{epoch}-009.pt"
                    agent.load_pretrain(probvlm_path=pretrain_probvlm_path, clipcap_path=clipcap_path, strict_clipcap=False) 

                elif "merge" in exp_name:
                    model_path = f"models/{args.dataset_prefix}/merged_models/model.safetensors"
                    state_dict = load_file(model_path)
                    # カスタムモデルの初期化と重みのロード
                    model = HuggingFaceClipCaptionModel.from_pretrained_weights(state_dict=state_dict, prefix_length=10, mapping_type='mlp')
                    agent.ClipCap = model
                    print("merged model")
                    
                elif "distil" in exp_name:
                    with open(f"{exp_dir}/args.json", 'r') as f:
                        exp_args = json.load(f)
                    lora_r = exp_args["lora_r"]
                    lora_alpha = exp_args["lora_alpha"]
                    lora_dropout = exp_args["lora_dropout"]
                    agent.lora_setting(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, probvlm=False)
                    agent.load_pretrain(probvlm_path=pretrain_probvlm_path, 
                                        clipcap_path=f"{exp_dir}/{agent_name}/clipcap_{agent_name}_{epoch}-009.pt", strict_clipcap=False)
                else:
                    # load exp args
                    with open(f"{exp_dir}/args.json", 'r') as f:
                        exp_args = json.load(f)
                    lora_r = exp_args["lora_r"]
                    lora_alpha = exp_args["lora_alpha"]
                    lora_dropout = exp_args["lora_dropout"]
                    agent.lora_setting(r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
                    agent.load_pretrain(probvlm_path=f"{exp_dir}/{agent_name}/probvlm_{agent_name}_{epoch}-epoch-0.pth", 
                                        clipcap_path=f"{exp_dir}/{agent_name}/clipcap_{agent_name}_{epoch}-009.pt", strict_clipcap=False)
            
                if args.num_samples > 1:
                    candidate_path = candidate_path.replace(".json", f"_{num_sample}.json")
                
                candidate_path = f"{exp_eval_dir}/{args.dataset_prefix}_candidate_{agent_name}_temperature_{temperature}_{args.mode}.json"
                print(f"Save the candidate to {candidate_path}")

                #######  Generate the candidate captions ####### 
                agent = agent.to(device)

                candidate = {}
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

                    mu_img, alpha_img, sigma_img, z = agent.image_encoder(image)

                    time_for_encoding = time.time() - start

                    if ("pretrain" in exp_name) or ("merge" in exp_name) or ("distillation" in exp_name) or ("all_acceptance" in exp_name):
                        captions = agent.text_decoder(z)
                    else:
                        captions = agent.text_decoder(mu_img)


                    for caption, filename in zip(captions, filenames):
                        filename = filename.split('/')[-1].split('.')[0]
                        candidate[filename] = caption

                    total_time += time.time() - start
                    total_time_for_encoding += time_for_encoding

                # save the candidate 
                # print length of the candidate
                print(f"Number of captions in the candidate: {len(candidate)}")

                print(f"Avg time per image: {total_time / len(train_dataset)}")
                print(f"Total time: {total_time}")

                print(f"Avg time for encoding per image: {total_time_for_encoding / len(train_dataset)}")
                print(f"Total time for encoding: {total_time_for_encoding}")
                
                with open(candidate_path, 'w') as f:
                    json.dump(candidate, f, indent=4)
                
            
            if ("pretrain" in exp_name) :
                break