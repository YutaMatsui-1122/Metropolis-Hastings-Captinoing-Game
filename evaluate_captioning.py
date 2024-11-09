# # Evaluate the communication between the agents in the vlm captioning game
# # Two kinds of evaluation:
# # 1. Evaluate the likelihood of the latent representation given the shared sign
# # 2. Evaluate the captioning performance for each pretraining dataset

# # 0.1. Import the required libraries
# import torch
# import pickle
# from one_agent import OneAgent
# from utils import * 
# import pickle, clip, argparse
# from torch.nn import functional as nnf
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import tqdm
# from bert_score import score as bert_score
# from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType
# import copy

# # 0.2. Define the device
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# # 0.3. Set the experiment name, directory, and the dataset
# exp_name = "gemcg_unified_dataset_coco5000_cc3m5000_alpha003_beta003_1"
# exp_dir = f"exp/{exp_name}"
# observation_file = "communication_coco_50_cc3m_50"
# temperature = 0.7
# epoch = 6

# agent = OneAgent(agent_name='A', device=device,temperature=temperature)
# agent = agent.to(device)
# agent.lora_setting()

# with open(f"dataset/dataset_cache/{observation_file}.pkl", "rb") as f:
#     observationA_dataset = pickle.load(f)
#     observationA_dataset.prefix_length = agent.prefix_length

# dataloader = torch.utils.data.DataLoader(observationA_dataset, batch_size=32, shuffle=False, num_workers=1)

# agent.load_pretrain(probvlm_path=f"{exp_dir}/A/probvlm_A_{epoch}-epoch-9.pth", clipcap_path=f"{exp_dir}/A/clipcap_A_{epoch}-009.pt")

# agent.dataloader_MHNG_fix = dataloader

# agent.perception()
# caption = agent.propose()
# print(caption[:10])
# for i in range(10):
#     print(tokenizer_decode(caption[i]))


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

class ConceptualCaptionsDataset(Dataset):
    def __init__(self, gpt2_type: str = "gpt2", data_mode: str = "train", img_dir=None, transform=None, return_captions=True, prefix_length=10, ids=None, datasize="full"):
        """
        Initializes the Conceptual Captions dataset class.
        
        Parameters:
        - gpt2_type: Type of GPT-2 tokenizer to use.
        - data_mode: 'train' or 'test' mode.
        - img_dir: Directory where images are stored (if provided).
        - transform: Optional transformations to apply to the images.
        - return_captions: Boolean indicating whether to return captions (default: True).
        - prefix_length: Length of the prefix for padding tokens.
        - ids: Specific indices to use for the dataset.
        - datasize: 'full' or subset of the dataset size.
        """
        self.root = "DownloadConceptualCaptions"
        self.data_mode = data_mode
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.vlm_tokenizer = tokenize
        self.datasize = datasize
        self.return_captions = return_captions
        self.transform = transform
        self.img_dir = img_dir
        self.prefix_length = prefix_length
        
        self.set_conceptual_dataset(ids)
    
    def set_conceptual_dataset(self, ids):
        if self.data_mode == "train":
            self.image_root = os.path.join(self.root, 'training')
            self.data_file = os.path.join(self.root, 'training_imgtxt.tsv')
        elif self.data_mode == "test":
            self.image_root = os.path.join(self.root, 'validation')
            self.data_file = os.path.join(self.root, 'validation_imgtxt.tsv')
        
        # Read tsv file and subset the data if ids is provided
        if ids is not None:
            self.data = pd.read_csv(self.data_file, sep='\t').iloc[ids].reset_index(drop=True)
        else:
            self.data = pd.read_csv(self.data_file, delimiter='\t', header=0)
        
        self.ids = list(range(len(self.data)))

        # Manage the dataset size
        if self.datasize == "full":
            self.ids = self.ids
        elif "full" in self.datasize:
            datasize = int(self.datasize.split("_")[1])
            self.ids = self.ids[:datasize]
        elif self.datasize is not None:
            self.ids = self.ids[::5][:int(self.datasize)]
        else:
            self.ids = self.ids[::5]
        
        self.captions = self.data['caption'].tolist()
        self.images = self.data['image'].tolist()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Get image path
        img_path = os.path.join(self.image_root, self.images[idx])
        img_filename = self.images[idx]

        # Load image
        img = Image.open(img_path).convert('RGB')

        # Apply transformations if any
        if self.transform is not None:
            img = self.transform(img)

        # If return_captions is False, return only the image and file name
        if not self.return_captions:
            return img, img_filename

        # Retrieve caption
        caption = self.captions[idx]

        # Return the image, caption, and image file name
        return img, caption, img_filename

class NoCapsDataset(Dataset):
    def __init__(self, annotation_file, img_dir, return_captions=True, transform=None):
        """
        Initializes the NoCaps dataset class.

        Parameters:
        - annotation_file: Path to the annotation file.
        - img_dir: Directory where images are stored.
        - return_captions: Boolean indicating whether to return captions (default: True).
        - transform: Optional transformations to apply to the images.
        """
        # Load the annotation file
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        self.img_dir = img_dir
        self.transform = transform
        self.return_captions = return_captions
        
        # Store image information (mapping between file name and ID)
        self.images = {img['id']: img['file_name'] for img in self.annotations['images']}
        
        # Map image IDs to their corresponding captions
        self.id_to_captions = {}
        for ann in self.annotations['annotations']:
            if ann['image_id'] not in self.id_to_captions:
                self.id_to_captions[ann['image_id']] = []
            self.id_to_captions[ann['image_id']].append(ann['caption'])

    def __len__(self):
        # Return the total number of images
        return len(self.images)

    def __getitem__(self, idx):
        # Get the image ID for the given index
        img_id = list(self.images.keys())[idx]
        
        # Retrieve the file name for the image
        img_filename = self.images[img_id]
        img_path = os.path.join(self.img_dir, img_filename)
        
        # Load the image
        image = Image.open(img_path).convert('RGB')

        # Apply any provided transformations to the image
        if self.transform:
            image = self.transform(image)
        
        # If return_captions is False, return only the image and file name
        if not self.return_captions:
            return image, img_filename

        # Retrieve the captions corresponding to the image ID
        captions = self.id_to_captions[img_id]

        # Return the image, captions, and file name
        return image, captions, img_filename

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
    coco_annotation_file = 'dataset/coco/annotations/annotations/captions_val2014.json'
    coco_images_dir = 'dataset/coco/val2014'
    nocap_annotation_file = 'dataset/nocaps/nocaps_val_4500_captions.json'
    nocap_images_dir = 'dataset/nocaps/validation'
    cc3m_annotation_file = 'dataset/conceptual_captions/annotations/validation_imgtxt.tsv'

    # argparseを使ってコマンドライン引数を取得
    parser = argparse.ArgumentParser(description='PAC-S evaluation')
    parser.add_argument('--exp_name', type=str, default='default')
    parser.add_argument('--dataset_name', type=str, default='coco', choices=['coco', 'nocaps', 'cc3m'])
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
        # データセットとデータローダーの作成
        if dataset_name == 'nocaps':
            dataset = NoCapsDataset(annotation_file=nocap_annotation_file, img_dir=nocap_images_dir, transform=transform, return_captions=False)
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        # class ConceptualCaptionsDataset(Dataset):
        # def __init__(self, gpt2_type: str = "gpt2", data_mode: str = "train", img_dir=None, transform=None, return_captions=True, prefix_length=10, ids=None, datasize="full"):
        elif dataset_name == 'cc3m':
            dataset = ConceptualCaptionsDataset(data_mode='test', img_dir=None, transform=transform, return_captions=False, prefix_length=10, ids=None, datasize="full")
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        else:
            dataset = CocoDataset(annotation_file=coco_annotation_file, img_dir=coco_images_dir, transform=transform)
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        
    elif args.mode == 'train':
        if dataset_name == "cc3m":
            with open("dataset/dataset_cache/cc3m_100000-150000_train.pkl", "rb") as f:
                dataset = pickle.load(f)
                dataset.prefix_length = 10
        elif dataset_name == "coco":
            with open("dataset/dataset_cache/coco_450000-500000_train.pkl", "rb") as f:
                dataset = pickle.load(f)
                dataset.prefix_length = 10
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"Number of images in the dataset: {len(dataset)}")
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

    for epoch in [29]:
        for agent_name in ["A", "B"]:
            agent = OneAgent(agent_name=agent_name, device=device,temperature=temperature, clip_arch=agent_clip_arch[agent_name])
            agent = agent.to(device)

            if use_official_model:
                exp_name = "pretrain"
                exp_eval_dir = f"exp_eval/{exp_name}"
                os.makedirs(exp_eval_dir, exist_ok=True)
                
                if agent_name == 'A':
                    # agent.load_pretrain(probvlm_path="models/official_model/probvlm/CC3M/probvlm_0.2_0.3_20-epoch-15.pth", clipcap_path="models/official_model/clipcap_conceptual_weights.pt", strict_clipcap=False)
                    agent.load_pretrain(probvlm_path="models/official_model/probvlm/CC3M/probvlm_0.2_0.3_20-epoch-15.pth", clipcap_path=f"models/clipcap_vit16_cc3m/clipcap_019.pt", strict_clipcap=False)
                    candidate_path = f"{exp_eval_dir}/{dataset_name}_candidate_cc3m_temperature_{temperature}_vit16_{args.mode}.json"
                else:
                    candidate_path = f"{exp_eval_dir}/{dataset_name}_candidate_coco_temperature_{temperature}_vit32_{args.mode}.json"
                    agent.load_pretrain(probvlm_path="models/probVLM_coco_prefix-035.pth", clipcap_path="models/official_model/clipcap_coco_weights.pt", strict_clipcap=False)
            else:
                candidate_path = f"{exp_eval_dir}/{dataset_name}_candidate_{agent_name}_epoch_{epoch}_temperature_{temperature}_{args.mode}.json"
                agent.lora_setting()
                agent.load_pretrain(probvlm_path=f"{exp_dir}/{agent_name}/probvlm_{agent_name}_{epoch}-epoch-9.pth", clipcap_path=f"{exp_dir}/{agent_name}/clipcap_{agent_name}_{epoch}-009.pt")

            print(f"EM iter {args.em_iter} , Candidate path {candidate_path}")

            candidate = {}

            if args.mode == 'eval':
                # 画像を順番に取り出して表示または処理 with tqdm
                for i, (images, filenames) in enumerate(tqdm(data_loader)):
                    images = images.to(device)
                    mu_img, alpha_img, sigma_img, z = agent.image_encoder(images)
                    if use_official_model:
                        captions = agent.text_decoder(z) # use CLIP's latent representation as the input of the text decoder
                    else:
                        captions = agent.text_decoder(mu_img) # use the latent representation of the image as the input of the text decoder

                    for caption, filename in zip(captions, filenames):
                        filename = filename.split('.')[0]
                        candidate[filename] = caption
                    print(captions[:3])
                
            elif args.mode == 'train':
                for data in tqdm(data_loader):
                    image = data["image"].to(device)
                    captions = data["caption"]
                    index = data["index"]
                    filenames = [dataset.dataset[i]["image"] for i in index]

                    mu_img, alpha_img, sigma_img, z = agent.image_encoder(image)
                    if use_official_model:
                        captions = agent.text_decoder(z)
                    else:
                        captions = agent.text_decoder(mu_img)

                    for caption, filename in zip(captions, filenames):
                        filename = filename.split('/')[-1].split('.')[0]
                        candidate[filename] = caption
                    print(captions[:3])


            # save the candidate 
            # print length of the candidate
            print(f"Number of captions in the candidate: {len(candidate)}")
            
            with open(candidate_path, 'w') as f:
                json.dump(candidate, f, indent=4)
            
            print(f"Saved the candidate to {candidate_path}")
        
        if args.use_official_model:
            break

        