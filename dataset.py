import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from utils import tokenize
from ProbVLM.src.ds._dataloader import _get_coco_file_paths

class CocoCc3mDataset(Dataset):
    def __init__(self, root_coco: str, root_cc3m: str, gpt2_type: str = "gpt2", data_mode: str = "train", 
                 transform=None, prefix_length=40, datasize_coco="full", datasize_cc3m="full", 
                 num_coco=5000, num_cc3m=5000, normalize_prefix=True):
        self.root_coco = root_coco
        self.root_cc3m = root_cc3m
        self.data_mode = data_mode
        self.transform = transform
        self.prefix_length = prefix_length
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.vlm_tokenizer = tokenize  # Assuming 'tokenize' is defined elsewhere

        self.coco_dataset = self.set_coco_dataset(datasize=datasize_coco, num_samples=num_coco)
        self.cc3m_dataset = self.set_conceptual_dataset(datasize=datasize_cc3m, num_samples=num_cc3m)
        
        self.dataset = self.coco_dataset + self.cc3m_dataset
        
        # 最大トークン長を計算して設定
        all_tokens = [item["gpt_token"] for item in self.dataset]
        all_len = torch.tensor([len(tokens) for tokens in all_tokens]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))

    def set_coco_dataset(self, datasize, num_samples):
        if self.data_mode == "train":
            ids, extra_ids, _, _, image_root, annFile, extra_annFile = _get_coco_file_paths(os.path.join(self.root_coco, 'coco'))
        elif self.data_mode == "test":
            _, _, _, ids, image_root, _, annFile = _get_coco_file_paths(os.path.join(self.root_coco, 'coco'))
            extra_ids = None
            extra_annFile = None

        coco = COCO(annFile)
        
        if extra_annFile is not None:
            with open(annFile, 'r') as fin1, open(extra_annFile, 'r') as fin2:
                dataset = json.load(fin1)
                extra_dataset = json.load(fin2)
                if not isinstance(dataset, dict) or not isinstance(extra_dataset, dict):
                    raise TypeError('invalid type {} {}'.format(type(dataset),
                                                                type(extra_dataset)))
                if set(dataset.keys()) != set(extra_dataset.keys()):
                    raise KeyError('key mismatch {} != {}'.format(list(dataset.keys()),
                                                                    list(extra_dataset.keys())))
                for key in ['images', 'annotations']:
                    dataset[key].extend(extra_dataset[key])
            coco.dataset = dataset
            coco.createIndex()

        ids = list(coco.anns.keys()) if ids is None else list(ids)
        if extra_ids is not None:
            ids += list(extra_ids)
        ids = [int(id_) for id_ in ids]
        
        if datasize != "full":
            ids = ids[:int(datasize)]
        ids = ids[:num_samples]

        captions = [coco.loadAnns(id_)[0]['caption'] for id_ in ids]
        images = [coco.loadImgs(coco.loadAnns(id_)[0]['image_id'])[0]['file_name'] for id_ in ids]
        
        vlm_tokens, gpt_tokens = self.get_tokens_list(captions)
        
        return [{"image": os.path.join(image_root, img), "caption": caption, "vlm_token": vlm_token, "gpt_token": gpt_token, "dataset": "COCO"} 
                for img, caption, vlm_token, gpt_token in zip(images, captions, vlm_tokens, gpt_tokens)]

    def set_conceptual_dataset(self, datasize, num_samples):
        if self.data_mode == "train":
            image_root = os.path.join(self.root_cc3m, 'training')
            data_file = os.path.join(self.root_cc3m, 'training_imgtxt.tsv')
        elif self.data_mode == "test":
            image_root = os.path.join(self.root_cc3m, 'validation')
            data_file = os.path.join(self.root_cc3m, 'validation_imgtxt.tsv')

        data = pd.read_csv(data_file, delimiter='\t', header=0)
        if datasize != "full":
            data = data[:int(datasize)]
        data = data[:num_samples]

        captions = data['caption'].tolist()
        images = data['image'].tolist()
        
        vlm_tokens, gpt_tokens = self.get_tokens_list(captions)
        
        return [{"image": os.path.join(image_root, img), "caption": caption, "vlm_token": vlm_token, "gpt_token": gpt_token, "dataset": "CC3M"} 
                for img, caption, vlm_token, gpt_token in zip(images, captions, vlm_tokens, gpt_tokens)]
    
    def get_tokens_list(self, captions):
        vlm_tokens_list = []
        gpt_tokens_list = []
        for caption in tqdm(captions):
            vlm_tokens = self.vlm_tokenizer(caption)
            gpt_tokens = torch.tensor(self.gpt_tokenizer.encode(caption))
            vlm_tokens_list.append(vlm_tokens[0])
            gpt_tokens_list.append(gpt_tokens)
        return vlm_tokens_list, gpt_tokens_list

    def pad_tokens(self, tokens):
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
        mask = tokens.ge(0)
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)
        return tokens, mask

    def __getitem__(self, index: int):
        item = self.dataset[index]
        img = Image.open(item["image"]).convert('RGB')
        gpt_token, gpt_mask = self.pad_tokens(item["gpt_token"])
        
        if self.transform is not None:
            img = self.transform(img)
        
        # データを辞書型で返す
        return {
            "image": img,
            "caption": item["caption"],
            "vlm_token": item["vlm_token"],
            "gpt_token": gpt_token,
            "gpt_mask": gpt_mask,
            "dataset_type": item["dataset"],
            "index": index
        }

    def __len__(self) -> int:
        return len(self.dataset)

if __name__ == "__main__":
    import clip

    _, transform = clip.load('ViT-B/32', jit=False)

    coco_data_path = 'dataset/'
    cc3m_data_path = 'DownloadConceptualCaptions'
    dataset = CocoCc3mDataset(root_coco=coco_data_path, root_cc3m=cc3m_data_path, data_mode="train", datasize_coco=50000, datasize_cc3m=100, transform=transform)
    
    # create dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    data_type_list = []
    for data in dataloader:
        print(data["image"].shape)
        print(data["caption"][:10])
        print(data["vlm_token"].shape)
        print(data["gpt_token"].shape)
        print(data["gpt_mask"].shape)
        print(data["dataset_type"][:10])
        print(data["index"].shape)        
        break