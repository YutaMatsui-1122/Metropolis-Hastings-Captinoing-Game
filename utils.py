import os, ujson
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from ProbVLM.src.ds.simple_tokenizer import SimpleTokenizer as ProbVLM_Tokenizer
from typing import Union, List
import ujson as json
from torch.distributions.gamma import Gamma

from tqdm import tqdm, trange
import random


from transformers import GPT2Tokenizer
import torch
from torch.utils.data import Dataset
import os, math
from PIL import Image
from pycocotools.coco import COCO
# from _dataloader import _get_coco_file_paths
from ProbVLM.src.ds._dataloader import _get_coco_file_paths
from CLIP_prefix_caption.parse_conceptual import *



_tokenizer = ProbVLM_Tokenizer()

def exponential_annealing_beta(epoch, initial_beta=0.01, growth_rate=0.1, epochs=100):
    return initial_beta + (1 - initial_beta) * (1 - np.exp(-growth_rate * epoch))

def cosine_annealing_beta(epoch, initial_beta=0.001, final_beta=1.0, epochs=100):
    """Cosine Annealingでbetaを上昇させる"""
    beta = final_beta + (initial_beta - final_beta) * (1 + np.cos(np.pi * epoch / epochs)) / 2
    return beta
def linear_annealing_beta(epoch, initial_beta=0.001, final_beta=1.0, epochs=100):
    """Linear Annealingでbetaを上昇させる"""
    beta = initial_beta + (final_beta - initial_beta) * epoch / epochs
    return beta
def tanh_annealing(epoch, initial_beta=0.01, final_beta=1.0, epochs=100, scale=10, center=0.5):
    """
    tanhを使用してbetaをアニーリングし、変化の具合を調整するパラメータを導入する
    Args:
        epoch: 現在のエポック数
        initial_beta: アニーリングの初期値
        final_beta: アニーリングの最終値
        epochs: アニーリングの合計エポック数
        scale: tanh関数のスケール
        center: 中心の位置を調整するパラメータ
    Returns:
        beta: アニーリングされた値
    """
    # Using the scale parameter to adjust the sharpness of the change
    beta = (final_beta - initial_beta) * (np.tanh(((epoch / epochs) - center) * scale) + 1) / 2 + initial_beta
    return beta


class LoRALayer(nn.Module):
    def __init__(self, original_linear_layer, r=4, alpha=16, dropout=0.1):
        super(LoRALayer, self).__init__()
        # 元の線形層を保存
        self.original_linear = original_linear_layer
        self.r = r
        self.alpha = alpha
        self.scaling = self.alpha / self.r
        
        # LoRAの低ランク行列 A と B の初期化
        # A はランダムなガウス分布で初期化、B はゼロ初期化
        self.lora_A = nn.Linear(original_linear_layer.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, original_linear_layer.out_features, bias=False)
        nn.init.zeros_(self.lora_B.weight)  # ゼロ初期化
        
        # ドロップアウトの追加
        self.lora_dropout = nn.Dropout(dropout)
        
        # 元の線形層のパラメータは更新しない（凍結）
        for param in self.original_linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        # 元の線形層の出力
        output = self.original_linear(x)
        # LoRAの出力（ドロップアウト適用）
        lora_output = self.lora_B(self.lora_A(x))
        lora_output = self.lora_dropout(lora_output) * self.scaling
        # LoRA出力を元の出力に追加
        return output + lora_output
    
# LoRAを適用する関数
def apply_lora_to_layer(module, layer_name, r=2, alpha=32, dropout=0.1):
    # 指定された層を取得し、LoRAラッピングを適用
    layer = getattr(module, layer_name)
    lora_layer = LoRALayer(layer, r=r, alpha=alpha, dropout=dropout)
    
    # モジュールの元の層をLoRAで置き換える
    setattr(module, layer_name, lora_layer)

class CocoDataset(Dataset):
    def __init__(self, root: str, gpt2_type: str = "gpt2", data_mode: str = "train" , transform=None, prefix_length = 40, normalize_prefix = True, datasize = "full", use_imageid = False):
        self.root = root
        self.data_mode = data_mode
        self.datasize = datasize
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.vlm_tokenizer = tokenize
        self.set_coco_dataset()
        self.transform = transform
        self.prefix_length = prefix_length
        self.use_imageid = use_imageid
        
    def set_coco_dataset(self):
        if self.data_mode == "train":
            ids, extra_ids, _, _, image_root, annFile, extra_annFile = _get_coco_file_paths(os.path.join(self.root, 'coco'))
        
        elif self.data_mode == "test":
            _, _, _, ids, image_root, _, annFile = _get_coco_file_paths(os.path.join(self.root, 'coco'))
            extra_ids = None
            extra_annFile = None

        self.image_root = image_root

        if extra_annFile is not None:
            self.coco = COCO()
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
            self.coco.dataset = dataset
            self.coco.createIndex()
        else:
            self.coco = COCO(annFile)
        
        self.ids = list(self.coco.anns.keys()) if ids is None else list(ids)
        if extra_ids is not None:
            self.ids += list(extra_ids)
        self.ids = [int(id_) for id_ in self.ids]
        # if self.data_mode == "test":
        if self.datasize == "full":
            self.ids = self.ids
        elif "full" in self.datasize:
            datasize = int(self.datasize.split("_")[1])
            self.ids = self.ids[:datasize]
        elif self.datasize is not None:
            self.ids = self.ids[::5][:int(self.datasize)]
        else:
            self.ids = self.ids[::5]


        self.all_image_ids = set([self.coco.loadAnns(annotation_id)[0]['image_id'] for annotation_id in self.ids])
        self.n_images = len(self.all_image_ids)
        self.vlm_tokens, self.gpt_tokens, self.captions = self.get_tokens_list()


        all_len = torch.tensor([len(self.gpt_tokens[i]) for i in range(len(self.gpt_tokens))]).float()

        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))

    def get_tokens_list(self):
        vlm_tokens_list = []
        gpt_tokens_list = []
        caption_list = []
        for index, id in enumerate(self.ids):
            ann = self.coco.loadAnns(id)[0]
            caption = ann['caption']
            caption_list.append(caption)
            vlm_tokens = self.vlm_tokenizer(caption)
            gpt_tokens = torch.tensor(self.gpt_tokenizer.encode(caption))
            vlm_tokens_list.append(vlm_tokens[0])
            gpt_tokens_list.append(gpt_tokens)
        return vlm_tokens_list, gpt_tokens_list, caption_list

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
        coco = self.coco
        annotation_id = self.ids[index]
        vlm_token = self.vlm_tokens[index]
        gpt_token = self.gpt_tokens[index]
        caption = self.captions[index]
        img_id = coco.loadAnns(annotation_id)[0]['image_id']

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.image_root, path)).convert('RGB')

        gpt_token, gpt_mask = self.pad_tokens(gpt_token)

        if self.transform is not None:
            img = self.transform(img)
        if self.use_imageid:
            return img, caption, vlm_token, gpt_token, gpt_mask, index, img_id
        else:
            return img, caption, vlm_token, gpt_token, gpt_mask, index
    
    def get_img(self, index: int):
        coco = self.coco
        annotation_id = self.ids[index]
        img_id = coco.loadAnns(annotation_id)[0]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.image_root, path)).convert('RGB')
        return img

    def __len__(self) -> int:
        return len(self.ids)

class ConceptualDataset(Dataset):
    def __init__(self, root: str, gpt2_type: str = "gpt2", data_mode: str = "train" , transform=None, prefix_length = 40, normalize_prefix = True, ids = None, datasize = None) -> None:
        self.root = root
        self.data_mode = data_mode
        self.datasize = datasize
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.vlm_tokenizer = tokenize
        self.set_conceptual_dataset(ids)
        self.transform = transform
        self.prefix_length = prefix_length
    
    def set_conceptual_dataset(self, ids):
        self.image_root = os.path.join(self.root, 'images')
        self.data_file = os.path.join(self.root, 'clean_train.tsv')
        # read tsv file
        if ids is not None:
            self.data = pd.read_csv(self.data_file, sep='\t').iloc[ids]
            self.data = self.data.reset_index(drop=True)

        else:
            self.data = pd.read_csv(self.data_file, sep='\t')
        self.data.columns = ["caption", "image_path"]
        self.ids = list(range(len(self.data)))
        if self.datasize is not None:
            self.ids = self.ids[::5][:self.datasize]
        else:
            self.ids = self.ids[::5]
        self.captions = self.data['caption'].tolist()
        self.vlm_tokens, self.gpt_tokens = self.get_tokens_list()
        all_len = torch.tensor([len(self.gpt_tokens[i]) for i in range(len(self.gpt_tokens))]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))
    
    def get_tokens_list(self):
        vlm_tokens_list = []
        gpt_tokens_list = []
        for index, caption in tqdm(enumerate(self.captions)):
            vlm_tokens = self.vlm_tokenizer(caption)
            gpt_tokens = torch.tensor(self.gpt_tokenizer.encode(caption))
            vlm_tokens_list.append(vlm_tokens[0])
            gpt_tokens_list.append(gpt_tokens)
        return vlm_tokens_list, gpt_tokens_list
    
    def __getitem__(self, index: int):
        vlm_token = self.vlm_tokens[index]
        gpt_token = self.gpt_tokens[index]
        caption = self.captions[index]
        path = self.data['image_path'][index]
        img = Image.open(os.path.join(self.image_root, path)).convert('RGB')
        gpt_token, gpt_mask = self.pad_tokens(gpt_token)
        if self.transform is not None:
            img = self.transform(img)
        return img, caption, vlm_token, gpt_token, gpt_mask, index
    
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
    
    def __len__(self) -> int:
        return len(self.ids)
    
class ConceptualDatasetFull(Dataset):
    def __init__(self, gpt2_type: str = "gpt2", data_mode: str = "train" , transform=None, prefix_length = 40, normalize_prefix = True, ids = None, datasize = "full") -> None:
        self.root = "DownloadConceptualCaptions"
        self.data_mode = data_mode
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.vlm_tokenizer = tokenize
        self.datasize = datasize
        self.set_conceptual_dataset(ids)
        self.transform = transform
        self.prefix_length = prefix_length        

    def set_conceptual_dataset(self, ids):
        if self.data_mode == "train":
            self.image_root = os.path.join(self.root, 'training')
            self.data_file = os.path.join(self.root, 'training_imgtxt.tsv')
        elif self.data_mode == "test":
            self.image_root = os.path.join(self.root, 'validation')
            self.data_file = os.path.join(self.root, 'validation_imgtxt.tsv')
        
        # read tsv file
        if ids is not None:
            self.data = pd.read_csv(self.data_file, sep='\t').iloc[ids]
            self.data = self.data.reset_index(drop=True)

        else:
            self.data = pd.read_csv(self.data_file, delimiter='\t', header=0)
        self.ids = list(range(len(self.data)))

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
        self.vlm_tokens, self.gpt_tokens = self.get_tokens_list()
        all_len = torch.tensor([len(self.gpt_tokens[i]) for i in range(len(self.gpt_tokens))]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))
    
    def get_tokens_list(self):
        vlm_tokens_list = []
        gpt_tokens_list = []
        for id in tqdm(self.ids):
            vlm_tokens = self.vlm_tokenizer(self.captions[id])
            gpt_tokens = torch.tensor(self.gpt_tokenizer.encode(self.captions[id]))
            vlm_tokens_list.append(vlm_tokens[0])
            gpt_tokens_list.append(gpt_tokens)
        return vlm_tokens_list, gpt_tokens_list
    
    def __getitem__(self, index: int):
        vlm_token = self.vlm_tokens[index]
        gpt_token = self.gpt_tokens[index]
        caption = self.captions[index]
        path = self.data['image'][index]
        img = Image.open(os.path.join(self.image_root, path)).convert('RGB')
        gpt_token, gpt_mask = self.pad_tokens(gpt_token)
        if self.transform is not None:
            img = self.transform(img)
        return img, caption, vlm_token, gpt_token, gpt_mask, index
    
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

    def get_img(self, index: int):
        path = self.data['image'][index]
        img = Image.open(os.path.join(self.image_root, path)).convert('RGB')
        return img  
    
    def __len__(self) -> int:
        return len(self.ids)

class UnifiedDataset(Dataset):
    def __init__(self, root_coco: str = 'dataset/', root_cc3m: str = 'DownloadConceptualCaptions', 
                 gpt2_type: str = "gpt2", data_mode: str = "train", transform=None, 
                 prefix_length=40, datasize_coco="full", datasize_cc3m="full", normalize_prefix=True):
        self.root_coco = root_coco
        self.root_cc3m = root_cc3m
        self.data_mode = data_mode
        self.transform = transform
        self.prefix_length = prefix_length
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.vlm_tokenizer = tokenize  # Assuming 'tokenize' is defined elsewhere

        self.coco_dataset = self.set_coco_dataset(datasize=datasize_coco)
        self.cc3m_dataset = self.set_conceptual_dataset(datasize=datasize_cc3m)
        
        self.dataset = self.coco_dataset + self.cc3m_dataset
        
        # 最大トークン長を計算して設定
        all_tokens = [item["gpt_token"] for item in self.dataset]
        all_len = torch.tensor([len(tokens) for tokens in all_tokens]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))

    def set_coco_dataset(self, datasize):
        if self.data_mode == "train":
            ids, extra_ids, _, _, image_root, annFile, extra_annFile = _get_coco_file_paths(os.path.join(self.root_coco, 'coco'))
        elif self.data_mode == "test":
            _, _, _, ids, image_root, _, annFile = _get_coco_file_paths(os.path.join(self.root_coco, 'coco'))
            extra_ids = None
            extra_annFile = None
        else:
            raise ValueError("Invalid data_mode")

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
        
        if datasize == "full":
            print("full")
            ids = ids
        elif "full" in datasize:
            print("full + ", datasize)
            datasize = int(datasize.split("_")[1])
            ids = ids[:datasize]
        elif datasize is not None:
            print(datasize)
            ids = ids[::5][:int(datasize)]
        else:
            print("else")
            ids = ids[::5]

        captions = [coco.loadAnns(id_)[0]['caption'] for id_ in ids]
        images = [coco.loadImgs(coco.loadAnns(id_)[0]['image_id'])[0]['file_name'] for id_ in ids]
        
        vlm_tokens, gpt_tokens = self.get_tokens_list(captions)
        
        return [{"image": os.path.join(image_root, img), "caption": caption, "vlm_token": vlm_token, "gpt_token": gpt_token, "dataset": "COCO"} 
                for img, caption, vlm_token, gpt_token in zip(images, captions, vlm_tokens, gpt_tokens)]

    def set_conceptual_dataset(self, datasize):
        if self.data_mode == "train":
            image_root = os.path.join(self.root_cc3m, 'training')
            data_file = os.path.join(self.root_cc3m, 'training_imgtxt_cleaned.tsv')
        elif self.data_mode == "test":
            image_root = os.path.join(self.root_cc3m, 'validation')
            data_file = os.path.join(self.root_cc3m, 'validation_imgtxt_cleaned.tsv')

        data = pd.read_csv(data_file, delimiter='\t', header=0)
        if datasize != "full":
            data = data[:int(datasize)]

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

    def get_img(self, index: int):
        item = self.dataset[index]
        img = Image.open(item["image"]).convert('RGB')
        return img

    def __getitem__(self, index: int):
        item = self.dataset[index]
        img = Image.open(item["image"]).convert('RGB')
        gpt_token, gpt_mask = self.pad_tokens(item["gpt_token"])
        
        if self.transform is not None:
            img = self.transform(img)

        # 常にキャプションを含む形式で返す
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



def set_caption_to_dataset(dataset, captions=None, caption_file=None):
    """
    set captions to dataset

    Parameters
    ----------
    dataset : Dataset
        dataset to set captions

    captions : List[str]
        list of captions

    caption_file : str
        path to caption file in csv format 
    """

    if captions is not None:
        dataset.captions = captions
    elif caption_file is not None: 
        # キャプションファイルから Generated という列をキャプションとして読み込む
        captions = pd.read_csv(caption_file)['Generated'].tolist()
        dataset.captions = captions
    else:
        raise ValueError("captions or caption_file should be set")
    dataset.vlm_tokens = [dataset.vlm_tokenizer(caption)[0] for caption in dataset.captions]
    dataset.gpt_tokens = [torch.tensor(dataset.gpt_tokenizer.encode(caption)) for caption in dataset.captions]
    # all_len = torch.tensor([len(dataset.gpt_tokens[i]) for i in range(len(dataset.gpt_tokens))]).float()
    # dataset.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))
    return dataset


def get_conceptual_dataset(datafile: str, split_ratio: str, root: str, gpt2_type: str = "gpt2", data_mode: str = "train" , transform=None, prefix_length = 40, normalize_prefix = True):
    data = pd.read_csv(datafile, sep='\t')
    data_length = len(data)

    ids = list(range(data_length))
    # random split
    np.random.seed(0)
    np.random.shuffle(ids)
    train_ids = ids[:int(data_length*split_ratio)]
    test_ids = ids[int(data_length*split_ratio):]

    train_dataset = ConceptualDataset(root=root, gpt2_type=gpt2_type, data_mode=data_mode, transform=transform, prefix_length=prefix_length, normalize_prefix=normalize_prefix, ids=train_ids)
    test_dataset = ConceptualDataset(root=root, gpt2_type=gpt2_type, data_mode=data_mode, transform=transform, prefix_length=prefix_length, normalize_prefix=normalize_prefix, ids=test_ids)
    print("train dataset length: ", len(train_dataset))
    print("test dataset length: ", len(test_dataset))
    return train_dataset, test_dataset

def tokenizer_decode(token):
    # 文をトークンデコード
    decoded_str = _tokenizer.decode(token.cpu().numpy())
    
    # 不要な部分を削除
    cleaned_str = decoded_str.split('.')[0].strip()
    
    # 不要な記号、単語、および "" を削除
    cleaned_str = cleaned_str.replace('!', '').replace('<|startoftext|>', '').replace('<|endoftext|>', '').strip()
    
    return cleaned_str + '.'

def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = True, lower: bool = True) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text, lower) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

def imagenet_normalize():
    """Standard ImageNet normalize transform
    """
#     return transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                 std=[0.229, 0.224, 0.225])
    return transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711))

def imagenet_transform_fn(resize_size=224,
                       crop_size=224,
                       random_resize_crop=False,
                       random_erasing_prob=0.0,
                       custom_transforms=None):
    """Standard ImageNet transform with resize/crop/normalize.

    Args:
        resize_size (int, Default: 256): resize for validation
            (only used when random_resize_crop is False).
        crop_size (int, Default: 224): final crop size.
        random_resize_crop (bool, Default: False): if True, use random transform (for training),
            if False, use center crop (for validation).
        custom_transforms (list of transform, Default: None): additional transforms.
    """
    if custom_transforms is not None:
        if not isinstance(custom_transforms, list):
            raise TypeError(f'custom_transforms should be list, not {type(custom_transforms)}')
    transform = []
    if random_resize_crop:
        transform.append(transforms.RandomResizedCrop(crop_size))
        transform.append(transforms.RandomHorizontalFlip())
    else:
        transform.append(transforms.Resize(resize_size))
        transform.append(transforms.CenterCrop(crop_size))
    transform.append(transforms.ToTensor())
    transform.append(imagenet_normalize())

    if custom_transforms:
        transform.extend(custom_transforms)

#     if random_erasing_prob > 0:
#         print(f'adding cutout {random_erasing_prob}')
#         transform.append(RandomErasing(random_erasing_prob,
#                                        mode='const',
#                                        max_count=1, num_splits=0, device='cpu'))
    #transform.append(RandomErasing(random_erasing_prob,
    #                               mode='const',
    #                               max_count=1, num_splits=0, device='cpu'))

    transform = transforms.Compose(transform)
    #print("Transform Called")
    return transform



def generate2(
        model,
        tokenizer,
        tokens=None,
        prompt=None,
        embed=None,
        entry_count=1,
        entry_length=40,  # maximum number of words
        top_p=0.8,
        temperature=1.,
        stop_token: str = '.',
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():

        for entry_idx in range(entry_count):
            if embed is not None:
                if entry_count > 1:
                    tokens = None
                    generated = embed[entry_idx].unsqueeze(0)
                else:
                    generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):
                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                # next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None :
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)

                if stop_token_index == next_token.item() or stop_token==tokenizer.decode(next_token.item()):
                    break
            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)
    if entry_count == 1:
        return generated_list[0]
    else:
        return generated_list

def generate_batch(
    model,
    tokenizer,
    embed=None,
    entry_length=40,  # maximum number of words
    top_p=0.8,
    temperature=1.0,
    stop_token='.'
):
    model.eval()
    stop_token_id = tokenizer.encode(stop_token)[0]
    device = next(model.parameters()).device
    
    tokens = None
    import time

    with torch.no_grad():
        generated = embed.to(device)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits

            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # まずsorted_indices_to_removeを並び替えるための補助インデックスを生成
            aux_indices = torch.argsort(sorted_indices, dim=1)

            # sorted_indices_to_removeを補助インデックスに従って並べ替え
            indices_to_remove = torch.gather(sorted_indices_to_remove, 1, aux_indices)

            # indices_to_removeがTrueの位置を-infで置き換える
            logits = torch.where(indices_to_remove, torch.full_like(logits, float('-inf')), logits)    
            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            next_token_embed = model.gpt.transformer.wte(next_token)
            if tokens is None:
                tokens = next_token
            else:
                tokens = torch.cat((tokens, next_token), dim=1)

            generated = torch.cat((generated, next_token_embed), dim=1)
            
            # tokensのすべての行にstop_token_idがあるときに終了
            if (tokens == stop_token_id).any(dim=1).all():
                break
            # entry_length - 2回目のループで終了トークンが見つからない場合
            if i == entry_length - 2:
                tokens = torch.cat((tokens, torch.full((tokens.shape[0], 1), stop_token_id, dtype=tokens.dtype).to(device)), dim=1)


        output_list = tokens.cpu().numpy().tolist()
        output_text = [tokenizer.decode(output_list[i]) for i in range(len(output_list))]
        output_text = [output_text[i][:output_text[i].find(stop_token)+1] for i in range(len(output_text))]
    return output_text
        

def generate_beam(model, tokenizer, beam_size: int = 5, prompt=None, embed=None,
                  entry_length=67, temperature=1., stop_token: str = '.'):

    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts


def save_MH_naming_game(before_w, proposed_w, after_w, updated_index_list, self_z, mu_Sp_list, mu_Li_list, alpha_Sp_list, alpha_Li_list, beta_Sp_list, beta_Li_list, save_path):
    df = pd.DataFrame()
    
    before_w_cap = []
    after_w_cap = []
    proposed_w_cap = []
    for i in range(len(proposed_w)):
        before_w_cap.append(tokenizer_decode(before_w[i]))
        after_w_cap.append(tokenizer_decode(after_w[i]))
        proposed_w_cap.append(tokenizer_decode(proposed_w[i]))
    
    accept_index = np.zeros(len(after_w_cap))
    accept_index[updated_index_list] = 1

    self_z = self_z.cpu().numpy()
    mu_Sp_distance = np.linalg.norm(mu_Sp_list.cpu().numpy() - self_z, axis=1)
    mu_Li_distance = np.linalg.norm(mu_Li_list.cpu().numpy() - self_z, axis=1)
    alpha_Sp_list = alpha_Sp_list.mean(dim=1).cpu().numpy()
    alpha_Li_list = alpha_Li_list.mean(dim=1).cpu().numpy()
    beta_Sp_list = beta_Sp_list.mean(dim=1).cpu().numpy()
    beta_Li_list = beta_Li_list.mean(dim=1).cpu().numpy()

    df['before_w'] = before_w_cap
    df['proposed_w'] = proposed_w_cap
    df['accept'] = accept_index
    df['after_w'] = after_w_cap
    df['mu_Sp_distance'] = mu_Sp_distance
    df['mu_Li_distance'] = mu_Li_distance
    df['alpha_Sp'] = alpha_Sp_list
    df['alpha_Li'] = alpha_Li_list
    df['beta_Sp'] = beta_Sp_list
    df['beta_Li'] = beta_Li_list

    df.to_csv(save_path)


def save_loss_list(clipcap_loss_list, clipcap_loss_list_test, probvlm_loss_list, probvlm_loss_list_test, path):
    np.save(path + "clipcap_loss_list.npy", np.array(clipcap_loss_list))
    np.save(path + "clipcap_loss_list_test.npy", np.array(clipcap_loss_list_test))
    np.save(path + "probvlm_loss_list.npy", np.array(probvlm_loss_list))
    np.save(path + "probvlm_loss_list_test.npy", np.array(probvlm_loss_list_test))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# reservoir sampling
def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.

    Args:
        num_seen_examples: the number of seen examples
        buffer_size: the maximum buffer size

    Returns:
        the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1

# Buffer function for Dark Experience Replay
class ClipCapBuffer:
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.num_seen_examples = 0

        # initialize the buffer
        self.image_embed_buffer = None
        self.vlm_token_buffer = None
        self.gpt_token_buffer = None
        self.gpt_mask_buffer = None
        self.logit_buffer = None
        self.task_id_buffer = None


    def add(self, image_embed, vlm_token, gpt_token, gpt_mask, logit, task_ids):

        """
        Add an example to the buffer.

        Args:
            image_embed: the image embedding
            vlm_token: the VLM token
            gpt_token: the GPT token
            gpt_mask: the GPT mask
            logit: the logit

        Returns:
            None
        """

        # バッファが初期化されていない場合は初期化
        if self.image_embed_buffer is None:
            self.image_embed_buffer = torch.zeros((self.buffer_size, image_embed.shape[1]), dtype = image_embed.dtype)
            self.vlm_token_buffer = torch.zeros((self.buffer_size, vlm_token.shape[1]), dtype = vlm_token.dtype)
            self.gpt_token_buffer = torch.zeros((self.buffer_size, gpt_token.shape[1]), dtype = gpt_token.dtype)
            self.gpt_mask_buffer = torch.zeros((self.buffer_size, gpt_mask.shape[1]), dtype = gpt_mask.dtype)
            self.logit_buffer = torch.zeros((self.buffer_size, logit.shape[1], logit.shape[2]), dtype = logit.dtype)
            self.task_id_buffer = torch.zeros((self.buffer_size), dtype = task_ids.dtype)

        # バッファの次元と追加するデータの次元が異なる場合は、大きい方に合わせる
        if image_embed.shape[1] > self.image_embed_buffer.shape[1]:
            self.image_embed_buffer = torch.cat((self.image_embed_buffer, torch.zeros((self.buffer_size, image_embed.shape[1] - self.image_embed_buffer.shape[1]), dtype = image_embed.dtype)), dim=1)
        elif image_embed.shape[1] < self.image_embed_buffer.shape[1]:
            image_embed = torch.cat((image_embed, torch.zeros((image_embed.shape[0], self.image_embed_buffer.shape[1] - image_embed.shape[1]), dtype = image_embed.dtype)), dim=1)

        if vlm_token.shape[1] > self.vlm_token_buffer.shape[1]:
            self.vlm_token_buffer = torch.cat((self.vlm_token_buffer, torch.zeros((self.buffer_size, vlm_token.shape[1] - self.vlm_token_buffer.shape[1]), dtype = vlm_token.dtype)), dim=1)
        elif vlm_token.shape[1] < self.vlm_token_buffer.shape[1]:
            vlm_token = torch.cat((vlm_token, torch.zeros((vlm_token.shape[0], self.vlm_token_buffer.shape[1] - vlm_token.shape[1]), dtype = vlm_token.dtype)), dim=1)

        if gpt_token.shape[1] > self.gpt_token_buffer.shape[1]:
            self.gpt_token_buffer = torch.cat((self.gpt_token_buffer, torch.zeros((self.buffer_size, gpt_token.shape[1] - self.gpt_token_buffer.shape[1]), dtype = gpt_token.dtype)), dim=1)
        elif gpt_token.shape[1] < self.gpt_token_buffer.shape[1]:
            gpt_token = torch.cat((gpt_token, torch.zeros((gpt_token.shape[0], self.gpt_token_buffer.shape[1] - gpt_token.shape[1]), dtype = gpt_token.dtype)), dim=1)

        if gpt_mask.shape[1] > self.gpt_mask_buffer.shape[1]:
            self.gpt_mask_buffer = torch.cat((self.gpt_mask_buffer, torch.zeros((self.buffer_size, gpt_mask.shape[1] - self.gpt_mask_buffer.shape[1]), dtype = gpt_mask.dtype)), dim=1)
        elif gpt_mask.shape[1] < self.gpt_mask_buffer.shape[1]:
            gpt_mask = torch.cat((gpt_mask, torch.zeros((gpt_mask.shape[0], self.gpt_mask_buffer.shape[1] - gpt_mask.shape[1]), dtype = gpt_mask.dtype)), dim=1)

        if logit.shape[1] > self.logit_buffer.shape[1]:
            self.logit_buffer = torch.cat((self.logit_buffer, torch.zeros((self.buffer_size, logit.shape[1] - self.logit_buffer.shape[1], logit.shape[2]), dtype=logit.dtype)), dim=1)
        elif logit.shape[1] < self.logit_buffer.shape[1]:
            logit = torch.cat((logit, torch.zeros((logit.shape[0], self.logit_buffer.shape[1] - logit.shape[1], logit.shape[2]), dtype=logit.dtype)), dim=1)

        for i in range(image_embed.shape[0]):
            if self.num_seen_examples < self.buffer_size:
                self.image_embed_buffer[self.num_seen_examples] = image_embed[i]
                self.vlm_token_buffer[self.num_seen_examples] = vlm_token[i]
                self.gpt_token_buffer[self.num_seen_examples] = gpt_token[i]
                self.gpt_mask_buffer[self.num_seen_examples] = gpt_mask[i]
                self.logit_buffer[self.num_seen_examples] = logit[i]
                self.task_id_buffer[self.num_seen_examples] = task_ids[i]

            else:
                idx = reservoir(self.num_seen_examples, self.buffer_size)
                if idx != -1:
                    self.image_embed_buffer[idx] = image_embed[i]
                    self.vlm_token_buffer[idx] = vlm_token[i]
                    self.gpt_token_buffer[idx] = gpt_token[i]
                    self.gpt_mask_buffer[idx] = gpt_mask[i]
                    self.logit_buffer[idx] = logit[i]
                    self.task_id_buffer[idx] = task_ids[i]
            
            self.num_seen_examples += 1

    def sample(self, batch_size: int):
        """
        Sample a batch from the buffer.

        Args:
            batch_size: the batch size

        Returns:
            a batch of examples
        """
        if self.num_seen_examples < batch_size:
            return self.image_embed_buffer[:self.num_seen_examples], self.vlm_token_buffer[:self.num_seen_examples], self.gpt_token_buffer[:self.num_seen_examples], self.gpt_mask_buffer[:self.num_seen_examples], self.logit_buffer[:self.num_seen_examples]
        else:
            idx = np.random.choice(self.buffer_size, batch_size, replace=False)
            return self.image_embed_buffer[idx], self.vlm_token_buffer[idx], self.gpt_token_buffer[idx], self.gpt_mask_buffer[idx], self.logit_buffer[idx]
    
    def save_buffer(self, file_path: str):
        """
        Save the buffer to a directory with pickle.

        Args:
            file_path: the file path
        Returns:
            None
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)


    def get_task_ids(self):
        if self.num_seen_examples < self.buffer_size:
            return self.task_id_buffer[:self.num_seen_examples]
        else:
            return self.task_id_buffer

    def __len__(self):
        if self.num_seen_examples < self.buffer_size:
            return self.num_seen_examples
        else:
            return self.buffer_size
    
# text_emb, z, txt_mu, txt_alpha, txt_beta = buffer.sample(16)

class ProbVLMBuffer:
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.num_seen_examples = 0

        # initialize the buffer
        self.text_emb_buffer = None
        self.z_buffer = None
        self.txt_mu_buffer = None
        self.txt_alpha_buffer = None
        self.txt_beta_buffer = None
        self.task_id_buffer = None


    def add(self, text_emb, z, txt_mu, txt_alpha, txt_beta, task_ids):

        """
        Add an example to the buffer.

        Args:
            text_emb: the text embedding
            z: the latent vector
            txt_mu: the mean of the text embedding
            txt_alpha: the alpha of the text embedding
            txt_beta: the beta of the text embedding
        Returns:
            None
        """

        # バッファが初期化されていない場合は初期化
        if self.text_emb_buffer is None:
            self.text_emb_buffer = torch.zeros((self.buffer_size, text_emb.shape[1]), dtype = text_emb.dtype)
            self.z_buffer = torch.zeros((self.buffer_size, z.shape[1]), dtype = z.dtype)
            self.txt_mu_buffer = torch.zeros((self.buffer_size, txt_mu.shape[1]), dtype = txt_mu.dtype)
            self.txt_alpha_buffer = torch.zeros((self.buffer_size, txt_alpha.shape[1]), dtype = txt_alpha.dtype)
            self.txt_beta_buffer = torch.zeros((self.buffer_size, txt_beta.shape[1]), dtype = txt_beta.dtype)
            self.task_id_buffer = torch.zeros((self.buffer_size), dtype = task_ids.dtype)
        
        # バッファの次元と追加するデータの次元が異なることはないため、そのまま追加
        for i in range(text_emb.shape[0]):
            if self.num_seen_examples < self.buffer_size:
                self.text_emb_buffer[self.num_seen_examples] = text_emb[i]
                self.z_buffer[self.num_seen_examples] = z[i]
                self.txt_mu_buffer[self.num_seen_examples] = txt_mu[i]
                self.txt_alpha_buffer[self.num_seen_examples] = txt_alpha[i]
                self.txt_beta_buffer[self.num_seen_examples] = txt_beta[i]
                self.task_id_buffer[self.num_seen_examples] = task_ids[i]

            else:
                idx = reservoir(self.num_seen_examples, self.buffer_size)
                if idx != -1:
                    self.text_emb_buffer[idx] = text_emb[i]
                    self.z_buffer[idx] = z[i]
                    self.txt_mu_buffer[idx] = txt_mu[i]
                    self.txt_alpha_buffer[idx] = txt_alpha[i]
                    self.txt_beta_buffer[idx] = txt_beta[i]
                    self.task_id_buffer[idx] = task_ids[i]
            
            self.num_seen_examples += 1

    def sample(self, batch_size: int):
        """
        Sample a batch from the buffer.

        Args:
            batch_size: the batch size

        Returns:
            a batch of examples
        """
        if self.num_seen_examples < batch_size:
            return self.text_emb_buffer[:self.num_seen_examples], self.z_buffer[:self.num_seen_examples], self.txt_mu_buffer[:self.num_seen_examples], self.txt_alpha_buffer[:self.num_seen_examples], self.txt_beta_buffer[:self.num_seen_examples]
        else:
            idx = np.random.choice(self.buffer_size, batch_size, replace=False)
            return self.text_emb_buffer[idx], self.z_buffer[idx], self.txt_mu_buffer[idx], self.txt_alpha_buffer[idx], self.txt_beta_buffer[idx]
    
    def get_task_ids(self):
        if self.num_seen_examples < self.buffer_size:
            return self.task_id_buffer[:self.num_seen_examples]
        else:
            return self.task_id_buffer
        
    def __len__(self):
        if self.num_seen_examples < self.buffer_size:
            return self.num_seen_examples
        else:
            return self.buffer_size
        

# Buffer function for Dark Experience Replay
class Top_k_Buffer:
    def __init__(self, buffer_size: int, top_k: int, fill_value=-1e2):
        self.buffer_size = buffer_size
        self.num_seen_examples = 0
        self.top_k = top_k
        self.fill_value = fill_value

        # initialize the buffer
        self.image_embed_buffer = None
        self.vlm_token_buffer = None
        self.gpt_token_buffer = None
        self.gpt_mask_buffer = None
        self.logit_values_buffer = None
        self.logit_indices_buffer = None
        self.logit_num_token = None
        self.logit_num_vocab = None
        self.task_id_buffer = None


    def add(self, image_embed, vlm_token, gpt_token, gpt_mask, logit, task_ids):

        """
        Add an example to the buffer.

        Args:
            image_embed: the image embedding
            vlm_token: the VLM token
            gpt_token: the GPT token
            gpt_mask: the GPT mask
            logit: the logit

        Returns:
            None
        """

        # extract top k of the logit
        logit_values, logit_indices = torch.topk(logit, k=self.top_k, dim=2, largest=True, sorted=True)

        # バッファが初期化されていない場合は初期化
        if self.image_embed_buffer is None:
            self.image_embed_buffer = torch.zeros((self.buffer_size, image_embed.shape[1]), dtype = image_embed.dtype)
            self.vlm_token_buffer = torch.zeros((self.buffer_size, vlm_token.shape[1]), dtype = vlm_token.dtype)
            self.gpt_token_buffer = torch.zeros((self.buffer_size, gpt_token.shape[1]), dtype = gpt_token.dtype)
            self.gpt_mask_buffer = torch.zeros((self.buffer_size, gpt_mask.shape[1]), dtype = gpt_mask.dtype)
            self.logit_values_buffer = torch.zeros((self.buffer_size, logit.shape[1], self.top_k), dtype = logit.dtype)
            self.logit_indices_buffer = torch.zeros((self.buffer_size, logit.shape[1], self.top_k), dtype = logit.dtype)
            self.logit_num_token = logit.shape[1]
            self.logit_num_vocab = logit.shape[2]
            self.task_id_buffer = torch.zeros((self.buffer_size), dtype = task_ids.dtype)

        # バッファの次元と追加するデータの次元が異なる場合は、大きい方に合わせる
        if image_embed.shape[1] > self.image_embed_buffer.shape[1]:
            self.image_embed_buffer = torch.cat((self.image_embed_buffer, torch.zeros((self.buffer_size, image_embed.shape[1] - self.image_embed_buffer.shape[1]), dtype = image_embed.dtype)), dim=1)
        elif image_embed.shape[1] < self.image_embed_buffer.shape[1]:
            image_embed = torch.cat((image_embed, torch.zeros((image_embed.shape[0], self.image_embed_buffer.shape[1] - image_embed.shape[1]), dtype = image_embed.dtype)), dim=1)

        if vlm_token.shape[1] > self.vlm_token_buffer.shape[1]:
            self.vlm_token_buffer = torch.cat((self.vlm_token_buffer, torch.zeros((self.buffer_size, vlm_token.shape[1] - self.vlm_token_buffer.shape[1]), dtype = vlm_token.dtype)), dim=1)
        elif vlm_token.shape[1] < self.vlm_token_buffer.shape[1]:
            vlm_token = torch.cat((vlm_token, torch.zeros((vlm_token.shape[0], self.vlm_token_buffer.shape[1] - vlm_token.shape[1]), dtype = vlm_token.dtype)), dim=1)

        if gpt_token.shape[1] > self.gpt_token_buffer.shape[1]:
            self.gpt_token_buffer = torch.cat((self.gpt_token_buffer, torch.zeros((self.buffer_size, gpt_token.shape[1] - self.gpt_token_buffer.shape[1]), dtype = gpt_token.dtype)), dim=1)
        elif gpt_token.shape[1] < self.gpt_token_buffer.shape[1]:
            gpt_token = torch.cat((gpt_token, torch.zeros((gpt_token.shape[0], self.gpt_token_buffer.shape[1] - gpt_token.shape[1]), dtype = gpt_token.dtype)), dim=1)

        if gpt_mask.shape[1] > self.gpt_mask_buffer.shape[1]:
            self.gpt_mask_buffer = torch.cat((self.gpt_mask_buffer, torch.zeros((self.buffer_size, gpt_mask.shape[1] - self.gpt_mask_buffer.shape[1]), dtype = gpt_mask.dtype)), dim=1)
        elif gpt_mask.shape[1] < self.gpt_mask_buffer.shape[1]:
            gpt_mask = torch.cat((gpt_mask, torch.zeros((gpt_mask.shape[0], self.gpt_mask_buffer.shape[1] - gpt_mask.shape[1]), dtype = gpt_mask.dtype)), dim=1)

        if logit_values.shape[1] > self.logit_values_buffer.shape[1]:
            self.logit_values_buffer = torch.cat((self.logit_values_buffer, torch.zeros((self.buffer_size, logit_values.shape[1] - self.logit_values_buffer.shape[1], logit_values.shape[2]), dtype=logit_values.dtype)), dim=1)
        elif logit_values.shape[1] < self.logit_values_buffer.shape[1]:
            logit_values = torch.cat((logit_values, torch.zeros((logit_values.shape[0], self.logit_values_buffer.shape[1] - logit_values.shape[1], logit_values.shape[2]), dtype=logit_values.dtype)), dim=1)
        
        if logit_indices.shape[1] > self.logit_indices_buffer.shape[1]:
            self.logit_indices_buffer = torch.cat((self.logit_indices_buffer, torch.zeros((self.buffer_size, logit_indices.shape[1] - self.logit_indices_buffer.shape[1], logit_indices.shape[2]), dtype=logit_indices.dtype)), dim=1)
        elif logit_indices.shape[1] < self.logit_indices_buffer.shape[1]:
            logit_indices = torch.cat((logit_indices, torch.zeros((logit_indices.shape[0], self.logit_indices_buffer.shape[1] - logit_indices.shape[1], logit_indices.shape[2]), dtype=logit_indices.dtype)), dim=1)


        for i in range(image_embed.shape[0]):
            if self.num_seen_examples < self.buffer_size:
                self.image_embed_buffer[self.num_seen_examples] = image_embed[i]
                self.vlm_token_buffer[self.num_seen_examples] = vlm_token[i]
                self.gpt_token_buffer[self.num_seen_examples] = gpt_token[i]
                self.gpt_mask_buffer[self.num_seen_examples] = gpt_mask[i]
                self.logit_values_buffer[self.num_seen_examples] = logit_values[i]
                self.logit_indices_buffer[self.num_seen_examples] = logit_indices[i]
                self.task_id_buffer[self.num_seen_examples] = task_ids[i]

            else:
                idx = reservoir(self.num_seen_examples, self.buffer_size)
                if idx != -1:
                    self.image_embed_buffer[idx] = image_embed[i]
                    self.vlm_token_buffer[idx] = vlm_token[i]
                    self.gpt_token_buffer[idx] = gpt_token[i]
                    self.gpt_mask_buffer[idx] = gpt_mask[i]
                    self.logit_values_buffer[idx] = logit_values[i]
                    self.logit_indices_buffer[idx] = logit_indices[i]
                    self.task_id_buffer[idx] = task_ids[i]
            
            self.num_seen_examples += 1
        

    def sample(self, batch_size: int):
        """
        Sample a batch from the buffer.

        Args:
            batch_size: the batch size

        Returns:
            a batch of examples
        """
        idx = np.random.choice(self.buffer_size, batch_size, replace=False)
        # logit = self.reconstruct_logits(self.logit_values_buffer[idx], self.logit_indices_buffer[idx])

        if self.num_seen_examples < batch_size:
            return self.image_embed_buffer[:self.num_seen_examples], self.vlm_token_buffer[:self.num_seen_examples], self.gpt_token_buffer[:self.num_seen_examples], self.gpt_mask_buffer[:self.num_seen_examples], self.logit_values_buffer[:self.num_seen_examples], self.logit_indices_buffer[:self.num_seen_examples]
        else:
            
            return self.image_embed_buffer[idx], self.vlm_token_buffer[idx], self.gpt_token_buffer[idx], self.gpt_mask_buffer[idx], self.logit_values_buffer[idx], self.logit_indices_buffer[idx]

    def get_task_ids(self):
        if self.num_seen_examples < self.buffer_size:
            return self.task_id_buffer[:self.num_seen_examples]
        else:
            return self.task_id_buffer
    
    def reconstruct_logits(self, reconstructed_logits, logits_value, logits_index):
        # logits の top_k の値を予測logitsに上書きする形で復元        
        # バッチサイズとnum_tokenの範囲を生成
        batch_size, num_tokens = logits_value.shape[0], self.logit_num_token
        b, t = torch.meshgrid(torch.arange(batch_size), torch.arange(num_tokens), indexing='ij')
        
        # scatter_を使用して一括でログを復元
        reconstructed_logits[b, t] = reconstructed_logits[b, t].scatter_(2, logits_index.to(torch.long), logits_value.float())
        
        return reconstructed_logits

    def __len__(self):
        if self.num_seen_examples < self.buffer_size:
            return self.num_seen_examples
        else:
            return self.buffer_size
        

def generate_test(ClipCap, CLIP_Net, dataloader, tokenizer, sample_num, device = "cuda:0", prefix_length = 10, temperature = 1):
    ClipCap.eval().to(device)
    generated_list = []
    num_text = 0
    for i, batch in enumerate(tqdm(dataloader)):
        # img = batch[0].to(device)
        # gpt_token = batch[3].to(device)
        # gpt_mask = batch[4].to(device)
        img = batch['image'].to(device)
        gpt_token = batch['gpt_token'].to(device)
        gpt_mask = batch['gpt_mask'].to(device)

        img, gpt_token, gpt_mask = img.to(device), gpt_token.to(device), gpt_mask.to(device)
        
        prefix = CLIP_Net.encode_image(img).to(device, dtype=torch.float32)
        prefix_embeds = ClipCap.clip_project(prefix).reshape(prefix.shape[0], prefix_length, -1)

        generated_text = generate_batch(ClipCap, tokenizer, embed=prefix_embeds, temperature=temperature)
        generated_list.extend(generated_text)
        num_text += len(generated_text)
        if num_text >= sample_num:
            return generated_list[:sample_num]

        # for idx, prefix_embed in enumerate(prefix_embeds):
        # for idx, prefix_embed in enumerate(prefix_embeds):
        #     prefix_embed = prefix_embed.unsqueeze(0)
        #     generated_list.append(generated_text)
        #     if len(generated_list) == sample_num:
        #         return generated_list

def sample_ggd(mu, sigma, beta):
    """生成されたmu, sigma, betaを使用して一般化ガウス分布からサンプルを一度だけ生成"""

    """
    一般化ガウス分布のサンプリング
    mu: 平均値の行列
    sigma: スケールの行列
    beta: 形状パラメータの行列

    """

    shape = torch.reciprocal(beta)
    rate = torch.ones_like(beta)

    # ガンマ分布からのサンプリング
    gamma_dist = Gamma(shape, rate)
    gamma_samples = gamma_dist.sample()

    # ラデマッハー分布からサンプリング
    uniform = torch.rand_like(beta)
    rademacher_samples = torch.where(uniform < 0.5, torch.tensor(-1.0), torch.tensor(1.0))

    # 一般化ガウス分布への変換
    samples = mu + sigma * rademacher_samples * torch.pow(torch.abs(gamma_samples), torch.reciprocal(beta))
    return samples

def save_args_to_json(args, filename="config.json", save_dir="save_models"):
    args_dict = vars(args)
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'w') as f:
        ujson.dump(args_dict, f, indent=4)

if __name__ == "__main__":
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    datasize_coco = 10000
    datasize_cc3m = 0
    data_mode = "train"
    save_file_name = f"coco_{datasize_coco}_cc3m_{datasize_cc3m}_{data_mode}"

    # データセットの初期化
    dataset = UnifiedDataset(data_mode=data_mode, transform=preprocess, datasize_coco=f"{datasize_coco}", datasize_cc3m=f"{datasize_cc3m}")
    print("Dataset length:", len(dataset))

    with open(f"dataset/dataset_cache/{save_file_name}.pkl", "wb") as f:
        pickle.dump(dataset, f)

    # # save dataset
    # with open(f"dataset/dataset_cache/communication_coco_{datasize_coco}_cc3m_{datasize_cc3m}.pkl", "wb") as f:
    # # with open(f"dataset/dataset_cache/test_coco_{datasize_coco}_cc3m_{datasize_cc3m}.pkl", "wb") as f:
    #     pickle.dump(dataset, f)

    exit()

    # coco_dataset = CocoDataset(root="dataset/", transform=preprocess, datasize="1000",data_mode='train')

    # DataLoaderの作成
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=8)
    # coco_dataloader = DataLoader(coco_dataset, batch_size=16, shuffle=False, num_workers=1)

    # データセットに含まれる画像を保存
    import matplotlib.pyplot as plt
    # make directory (communication_coco_5000_cc3m_5000)
    os.makedirs(f"dataset_images/communication_coco_{datasize_coco}_cc3m_{datasize_cc3m}", exist_ok=True)
    for i in tqdm(range(5000, len(dataset))):
        img = dataset.get_img(i)
        plt.imshow(img)
        plt.savefig(f"dataset_images/communication_coco_{datasize_coco}_cc3m_{datasize_cc3m}/image_{i}.png")


    exit()
    for batch in coco_dataloader:
        # img, caption, vlm_token, gpt_token, gpt_mask, index
        image = batch[0]
        caption = batch[1]
        vlm_token = batch[2]
        gpt_token = batch[3]
        gpt_mask = batch[4]
        index = batch[5]
        print("COCO Dataset content:")
        print(f"Images: {image.shape}")
        print(caption[:10])
        print(f"VLM Tokens: {vlm_token.shape}")
        print(f"GPT Tokens: {gpt_token.shape}")
        print(f"GPT Masks: {gpt_mask.shape}")
        print(f"Index: {index}")



    print("Dataset length:", len(dataset))

    coco_num = 0
    cc3m_num = 0

    # バッチを取得して内容を確認する
    for batch in tqdm(dataloader):
        # print("Batch content:")
        image = batch['image']
        caption = batch['caption']
        vlm_token = batch['vlm_token']
        gpt_token = batch['gpt_token']
        gpt_mask = batch['gpt_mask']
        datatype = batch['dataset_type']
        index = batch['index']
        print("Unified Dataset content:")
        print(f"Images: {image.shape}")
        print(caption)
        print(f"VLM Tokens: {vlm_token.shape}")
        print(f"GPT Tokens: {gpt_token.shape}")
        print(f"GPT Masks: {gpt_mask.shape}")
        print(f"Dataset Type: {datatype}")
        print(f"Index: {index}")
        # save image
        import matplotlib.pyplot as plt
        for j, img in enumerate(image):
            img = img.permute(1, 2, 0)
            plt.imshow(img)
            plt.savefig(f"image_{j}.png")
        break

        # print(f"Images: {image.size()}")

        # print(f"Images: {batch['image'].size()}")
        # print(f"Captions: {batch['caption']}")
        # print(f"VLM Tokens: {batch['vlm_token']}")
        # print(f"GPT Tokens: {batch['gpt_token']}")
        # print(f"GPT Masks: {batch['gpt_mask']}")
        # print(f"Dataset Type: {batch['dataset_type']}")
        # print(f"Index: {batch['index']}")
        # datatype_list.extend(batch['dataset_type'])
        # batch['dataset_type'] に含まれる各要素が'COCO'か'CC3M'かをカウント
        coco_num += batch["dataset_type"].count("COCO")
        cc3m_num += batch["dataset_type"].count("CC3M")
        # coco_num += (batch['dataset_type'] == 'COCO')
        # cc3m_num += (batch['dataset_type'] == 'CC3M').sum().item()
        # break  # 最初のバッチだけ確認するためにループを終了
        break
    
    print(f"COCO: {coco_num}, CC3M: {cc3m_num}")



    # テストモードの確認
    dataset_test = UnifiedDataset(data_mode='test', transform=preprocess, datasize_coco="1000", datasize_cc3m="1000")

    dataloader_test = DataLoader(dataset_test, batch_size=4, shuffle=False)

    for batch in dataloader_test:
        print("\nTest mode batch content:")
        print(f"Images: {batch['image'].size()}")
        print(f"Captions: {batch['caption']}")
        print(f"VLM Tokens: {batch['vlm_token']}")
        print(f"GPT Tokens: {batch['gpt_token']}")
        print(f"GPT Masks: {batch['gpt_mask']}")
        print(f"Dataset Type: {batch['dataset_type']}")
        print(f"Index: {batch['index']}")
        break  # 最初のバッチだけ確認するためにループを終了