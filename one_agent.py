import os, copy
import argparse
import time

from os.path import join as ospj
from os.path import expanduser
from munch import Munch as mch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from peft import get_peft_model, LoraConfig, TaskType

from ProbVLM.src.utils import load_data_loader

from ProbVLM.src.ds import prepare_coco_dataloaders

from ProbVLM.src.networks import *
# from ProbVLM.src.train_probVLM import *
import clip
from CLIP_prefix_caption.train import *

from utils import *

import argparse

from ProbVLM.src.ds.simple_tokenizer import SimpleTokenizer as _Tokenizer
from update_models import *

import torch.multiprocessing as mp

_tokenizer = _Tokenizer()

class OneAgent(nn.Module):
    def __init__(self, agent_name='A', device='cuda:0', adapter="mlp", td_update_epochs=10, te_update_epochs=10, temperature = 0.62, te_alpha_beta=0.5, td_alpha_beta=0.5, te_train_mode="DERPP", td_train_mode="DERPP"):
        super().__init__()
        self.agent_name = agent_name
        self.device = device
        self.ProbVLM_Net = BayesCap_for_CLIP(inp_dim=512, out_dim=512, hid_dim=256, num_layers=3, p_drop=0.01,)
        self.ProbVLM_Net.eval()
        self.CLIP_Net, self.preprocess = clip.load("ViT-B/32", device = self.device)
        self.CLIP_Net = self.CLIP_Net.float()
        self.CLIP_Net.eval()
        self.adapter = adapter
        if self.adapter == "mlp":
            self.prefix_length = 10
        elif self.adapter == "transformer":
            self.prefix_length = 40
        self.prefix_length_clip = 40
        self.prefix_dim = 512
        self.num_layers = 8
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.ClipCap = ClipCaptionPrefix(self.prefix_length, self.prefix_length_clip, self.prefix_dim, self.num_layers, self.adapter)

        self.GGL = GenGaussLoss(reduction='batchsum')
        self.clipcap_loss_list = []
        self.clipcap_loss_list_test = []
        self.probvlm_loss_list = []
        self.probvlm_loss_list_test = []
        self.td_buffer = None
        self.te_buffer = None

        self.td_update_epochs = td_update_epochs
        self.te_update_epochs = te_update_epochs
        self.temperature = temperature
        self.te_alpha_beta = te_alpha_beta
        self.td_alpha_beta = td_alpha_beta
        self.te_train_mode = te_train_mode
        self.td_train_mode = td_train_mode
    
    def initialize_sign(self):
        print("Agent", self.agent_name, " initialize sign")
        with torch.no_grad():
            i = 0
            # for batch in self.dataloader_MHNG_fix:
            for batch in tqdm(self.dataloader_MHNG_fix, desc="initialize sign"):
                # img = batch[0].to(self.device)
                img = batch["image"].to(self.device)
                mu, _, _, _ = self.image_encoder(img)
                texts = self.text_decoder(mu)
                for b, text in enumerate(texts):
                    caption = text
                    vlm_token = tokenize(text)[0]
                    gpt_token = torch.tensor(self.tokenizer.encode(text))
                    
                    datatype = batch["dataset_type"][b]
                    self.dataloader_MHNG_fix.dataset.dataset[i]["caption"] = caption
                    self.dataloader_MHNG_fix.dataset.dataset[i]["vlm_token"] = vlm_token
                    self.dataloader_MHNG_fix.dataset.dataset[i]["gpt_token"] = gpt_token
                    self.dataloader_MHNG_fix.dataset.dataset[i]["dataset"] = datatype
                    i += 1
    
    def save_sign(self, status):
        # captions = self.dataloader_MHNG_fix.dataset.captions
        captions = [self.dataloader_MHNG_fix.dataset.dataset[i]["caption"] for i in range(len(self.dataloader_MHNG_fix.dataset))]
        self.df_sign[status] = captions
        # save sign in csv
        self.df_sign.to_csv(f"{self.save_dir}/agent_{self.agent_name}_sign.csv")
    
    def save_proposed_w(self, proposed_w, status):
        # captions = self.dataloader_MHNG_fix.dataset.captions
        proposed_captions = [tokenizer_decode(proposed_w[i]) for i in range(len(proposed_w))]
        self.df_proposed_w[status] = proposed_captions
        # save sign in csv
        self.df_proposed_w.to_csv(f"{self.save_dir}/agent_{self.agent_name}_proposed_w.csv")

    def load_pretrain(self, probvlm_path, clipcap_path, strict_clipcap=True):
        self.ProbVLM_Net.load_state_dict(torch.load(probvlm_path))
        self.ClipCap.load_state_dict(torch.load(clipcap_path), strict=strict_clipcap)

    def lora_setting(self):
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=2, lora_alpha=32, lora_dropout=0.1, target_modules=["c_fc"])
        self.ClipCap.gpt = get_peft_model(self.ClipCap.gpt, peft_config)

    def communication_field_setup(self, dataloader_MHNG_fix, dataloader_MHNG_shuffle, MH_iter, annealing_beta="None", mode="MHNG"):
        self.dataloader_MHNG_fix = dataloader_MHNG_fix
        self.dataloader_MHNG_shuffle = dataloader_MHNG_shuffle
        self.MH_epochs = MH_iter
        self.annealing_beta = annealing_beta
        self.mode = mode
        self.df_proposed_w = pd.DataFrame()
        self.df_sign = pd.DataFrame()

    def image_encoder(self, o): # o: image
        with torch.no_grad():
            z = self.CLIP_Net.encode_image(o)
            mu_img, alpha_img, sigma_img = self.ProbVLM_Net.img_BayesCap(z)
        return mu_img, alpha_img, sigma_img, z
    
    def text_encoder(self, w, return_z = False): # w: tokenized by CLIP tokenizer
        with torch.no_grad():
            z = self.CLIP_Net.encode_text(w)
            mu_cap, alpha_cap, sigma_cap = self.ProbVLM_Net.txt_BayesCap(z)
        if return_z:
            return mu_cap, alpha_cap, sigma_cap, z
        else:
            return mu_cap, alpha_cap, sigma_cap
        
    def text_decoder(self, z): # z: latent vector
        with torch.no_grad():
            prefix_embeds = self.ClipCap.clip_project(z.float()).reshape(z.shape[0], self.prefix_length, -1)
            t = generate_batch(self.ClipCap, self.tokenizer, embed=prefix_embeds, temperature=self.temperature)
        return t
    
    def perception(self): # tempral version of perception 
        print("Agent", self.agent_name, " perception")
        z = []
        for batch in self.dataloader_MHNG_fix:
            o = batch["image"].to(self.device)
            mu_img, alpha_img, sigma_img, _ = self.image_encoder(o)
            z.append(mu_img)
        self.z = torch.cat(z, dim=0)
    
    def propose(self):
        print("Agent", self.agent_name, " propose")
        with torch.no_grad():
            proposed_w = []
            max_index = len(self.dataloader_MHNG_fix.dataset)
            batch_size = 500
            indices = torch.arange(max_index)
            for i in range(0, max_index, batch_size):
                index = indices[i:i + batch_size]
                z = self.z[index]
                w = self.text_decoder(z)
                proposed_w.append(tokenize(w))
            proposed_w = torch.cat(proposed_w, dim=0).to(self.device)
        return proposed_w

    def propose_ddp(self, world_size=2):
        print(f"Agent {self.agent_name} propose_ddp with world size {world_size}")

        # DDP用の関数
        def run(args):
            rank, world_size = args  # タプルとして受け取る
            print(f"Rank {rank} process running")
            dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
            torch.cuda.set_device(rank)
            
            # モデルをDDPでラップ
            model_ddp = DDP(self.to(rank), device_ids=[rank])

            with torch.no_grad():
                proposed_w = []
                max_index = len(self.dataloader_MHNG_fix.dataset)
                batch_size = 200
                indices = torch.arange(max_index)

                # 各プロセスにデータを分割
                num_samples_per_rank = max_index // world_size
                start_idx = rank * num_samples_per_rank
                end_idx = start_idx + num_samples_per_rank if rank != world_size - 1 else max_index

                for i in range(start_idx, end_idx, batch_size):
                    index = indices[i:i + batch_size]
                    z_part = self.z[index]

                    # モデルのDDP版を使ってテキストデコードを実行
                    w = model_ddp.module.text_decoder(z_part)
                    proposed_w.append(tokenize(w))

                proposed_w = torch.cat(proposed_w, dim=0).to(self.device)

                # 全プロセスの結果を集める
                gathered_proposed_w = [torch.zeros_like(proposed_w) for _ in range(world_size)]
                dist.all_gather(gathered_proposed_w, proposed_w)

                dist.destroy_process_group()

                if rank == 0:
                    # ランク0で結果をまとめて返す
                    proposed_w_final = torch.cat(gathered_proposed_w, dim=0)
                    return proposed_w_final
            return None

        # 各プロセスでrun関数を実行し、ランク0の結果を取得
        proposed_captions = mp.get_context("spawn").Pool(world_size).map(run, [(rank, world_size) for rank in range(world_size)])

        # ランク0のキャプションを返す
        return proposed_captions[0] if proposed_captions is not None else None

    def judge(self, proposed_w, iter = 0):
        print("Agent", self.agent_name, " judge")
        with torch.no_grad():
            
            # before_self_w = torch.cat(self.dataloader_MHNG_fix.dataset.vlm_tokens, dim=0).reshape(len(self.dataloader_MHNG_fix.dataset), -1).clone().to(self.device)
            vlm_token = [self.dataloader_MHNG_fix.dataset.dataset[i]["vlm_token"] for i in range(len(self.dataloader_MHNG_fix.dataset))]
            before_self_w = torch.cat(vlm_token, dim=0).reshape(len(self.dataloader_MHNG_fix.dataset), -1).clone().to(self.device)
            # before_self_w = torch.cat(self.dataloader_MHNG_fix.dataset.dataset["vlm_token"], dim=0).reshape(len(self.dataloader_MHNG_fix.dataset), -1).clone().to(self.device)
            updated_index_list = []

            max_index = len(self.dataloader_MHNG_fix.dataset)
            batch_size = 100
            indices = torch.arange(max_index)

            for i in range(0, max_index, batch_size):
                index = indices[i:i + batch_size]
                mu_Sp, alpha_Sp, beta_Sp = self.text_encoder(proposed_w[index])
                mu_Li, alpha_Li, beta_Li = self.text_encoder(before_self_w[index])
                
                # calculate p(z|w,\phi)
                p_li = -self.GGL(mu_Li, alpha_Li, beta_Li, self.z[index])
                p_sp = -self.GGL(mu_Sp, alpha_Sp, beta_Sp, self.z[index])
                # calculate acceptance rate r
                
                if self.mode == "MHNG":
                    r = np.exp(np.where((p_sp-p_li).detach().cpu().numpy()<0,(p_sp-p_li).detach().cpu().numpy(),0))
                elif self.mode == "all_accept":
                    r = np.ones(len(p_sp))
                elif self.mode == "no_communication":
                    r = np.zeros(len(p_sp))
                u = np.random.rand(len(r),)

                # update w
                global_update_index = index[np.where(u < r)[0]]
                updated_index_list = updated_index_list + list(global_update_index)
                j = 0
                for index in global_update_index:
                    self.dataloader_MHNG_fix.dataset.dataset[index]["caption"] = tokenizer_decode(proposed_w[index])
                    self.dataloader_MHNG_fix.dataset.dataset[index]["vlm_token"] = proposed_w[index].cpu()
                    self.dataloader_MHNG_fix.dataset.dataset[index]["gpt_token"] = torch.tensor(self.tokenizer.encode(self.dataloader_MHNG_fix.dataset.dataset[index]["caption"]))
                    j += 1
                
            print("acceptance rate:", len(updated_index_list)/len(self.dataloader_MHNG_fix.dataset))
            acceptance_rate = len(updated_index_list)/len(self.dataloader_MHNG_fix.dataset)
            return acceptance_rate

    def initialize_td_buffer(self, dataloader, buffer_size):
        print("Agent", self.agent_name, " initialize buffer")

        self.td_buffer = ClipCapBuffer(buffer_size=buffer_size)

        with torch.no_grad():
            for batch in dataloader:
                img, caption, vlm_token, gpt_token, gpt_mask, _  = batch
                img, gpt_token, gpt_mask = img.to(self.device), gpt_token.to(self.device), gpt_mask.to(self.device)
                mu_img, alpha_img, sigma_img, z = self.image_encoder(img)
                img = img.cpu()
                outputs = self.ClipCap(gpt_token, mu_img, gpt_mask)
                logits = outputs.logits[:, dataloader.dataset.prefix_length - 1: -1]
                task_index = torch.tensor([0] * img.shape[0], device="cpu")
                self.td_buffer.add(mu_img.detach().cpu(), vlm_token.cpu(), gpt_token.cpu(), gpt_mask.cpu(), logits.detach().cpu(), task_index)
                if self.td_buffer.num_seen_examples > buffer_size:
                    break
    
    def initialize_te_buffer(self, dataloader, buffer_size):
        print("Agent", self.agent_name, " initialize buffer")
        self.te_buffer = ProbVLMBuffer(buffer_size=buffer_size)
        with torch.no_grad():
            for batch in dataloader:
                # def add(self, text_emb, z, txt_mu, txt_alpha, txt_beta, task_ids):
                img, _, vlm_token, _, _, _  = batch
                img, vlm_token = img.to(self.device), vlm_token.to(self.device)
                mu_cap, alpha_cap, sigma_cap, text_emb = self.text_encoder(vlm_token, return_z=True)
                mu_img, _, _, _ = self.image_encoder(img)

                # add(self, text_emb, z, txt_mu, txt_alpha, txt_beta, task_ids):
                self.te_buffer.add(text_emb.detach().cpu(), mu_img.detach().cpu(), mu_cap.detach().cpu(), alpha_cap.detach().cpu(), sigma_cap.detach().cpu(), torch.tensor([0] * img.shape[0], device="cpu"))
                if self.te_buffer.num_seen_examples > buffer_size:
                    break


    def update_text_decoder(self, em_epoch):
        print("Agent", self.agent_name, " update text decoder")
        self.ClipCap.train()
        #update_clipcap_derpp(agent.CLIP_Net, agent.ClipCap, agent.tokenizer, finetune_train_dataloader, f"models/{args.save_dir}", epochs = 10, lr=args.lr, train_mode=args.cl_mode, device=device, buffer=buffer, alpha=der_alpha, beta=derpp_beta)
        updated_clipcap = update_clipcap_derpp(self.z, self.CLIP_Net, self.ClipCap, self.tokenizer, self.dataloader_MHNG_shuffle, self.dataloader_MHNG_fix, self.save_dir, epochs = self.td_update_epochs, lr=1e-5, train_mode=self.td_train_mode, device=self.device, buffer=self.td_buffer, output_prefix="clipcap_"+self.agent_name+f"_{em_epoch}", save_every=5, alpha=self.td_alpha_beta, beta=self.td_alpha_beta)
        self.ClipCap = updated_clipcap.eval()
    
    def update_text_encoder(self, em_epoch):
        print("Agent", self.agent_name, " update text encoder")
        self.ProbVLM_Net.train()
        # updated_probvlm = update_probvlm(self.z, self.CLIP_Net, self.ProbVLM_Net, self.dataloader_MHNG_shuffle, self.save_dir, epochs = self.te_update_epochs, lr=1e-6, device=self.device, output_prefix="probvlm_"+self.agent_name+f"_{em_epoch}", save_every=5)
        updated_probvlm = update_probvlm_derpp(self.z, self.CLIP_Net, self.ProbVLM_Net, self.dataloader_MHNG_shuffle, self.save_dir, epochs = self.te_update_epochs, lr=1e-5, device=self.device, output_prefix="probvlm_"+self.agent_name+f"_{em_epoch}", save_every=5, buffer=self.te_buffer, train_mode = self.te_train_mode, cross_modal_lambda=1, alpha=self.te_alpha_beta, beta=self.te_alpha_beta)
        self.ProbVLM_Net = updated_probvlm.eval()
    
    def calculate_p_z_w(self, image, caption):
        # tokenized caption
        caption = tokenize(caption).to(self.device)
        mu_img, alpha_img, sigma_img, z = self.image_encoder(image)
        mu_cap, alpha_cap, sigma_cap = self.text_encoder(caption)
        p_z_w = -self.GGL(mu_cap, alpha_cap, sigma_cap, mu_img)
        return p_z_w

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    args = parser.parse_args()
    agent = OneAgent(agent_name='A')
    agent = agent.cuda()
    agent.eval()

    _, preprocess = clip.load(args.clip_model_type, device='cuda')

    data_path = 'dataset/'
    prefix_length = 40
    normalize_prefix = True
    print("Loading train dataset")
    with open("dataset/dataset_cache/coco_train_dataset.pkl", "rb") as f:
        train_dataset_A = pickle.load(f)
        train_dataset_B = copy.deepcopy(train_dataset_A)

    print("Loading test dataset")
    with open("dataset/dataset_cache/coco_test_dataset.pkl", "rb") as f:
        test_dataset_A = pickle.load(f)
        test_dataset_B = copy.deepcopy(test_dataset_A)

    train_dataloader_A = torch.utils.data.DataLoader(train_dataset_A, batch_size=64, shuffle=True, num_workers=4)
    test_dataloader_A = torch.utils.data.DataLoader(test_dataset_A, batch_size=64, shuffle=True, num_workers=4)
    train_dataloader_B = torch.utils.data.DataLoader(train_dataset_B, batch_size=64, shuffle=True, num_workers=4)
    test_dataloader_B = torch.utils.data.DataLoader(test_dataset_B, batch_size=64, shuffle=True, num_workers=4)

    agent.pretrain()