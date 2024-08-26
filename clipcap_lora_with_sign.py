import torch
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType
from one_agent import OneAgent
import pickle, clip, argparse
from utils import *

import os
import sys
from tqdm import tqdm
import numpy as np
from torch.optim import AdamW
from torch.nn import functional as nnf
from ProbVLM.src.losses import *

from transformers import get_linear_schedule_with_warmup

argparser = argparse.ArgumentParser()
argparser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
argparser.add_argument('--dataset', default="coco", choices=('coco', 'conceptual'))
argparser.add_argument('--exp_name', default="debug")
argparser.add_argument('--initial_epoch', type=int, default=0)
argparser.add_argument('--num_workers', type=int, default=1)
argparser.add_argument('--batch_size', type=int, default=32)

args = argparser.parse_args()

clip_model, preprocess = clip.load(args.clip_model_type, device="cuda")

agent = OneAgent(agent_name='A')
agent = agent.cuda()
agent.load_pretrain(probvlm_path="models/probVLM_conceptual_prefix-020.pth", clipcap_path="models/conceptual_prefix-020.pt")
# agent.load_pretrain(probvlm_path="models/probVLM_coco_prefix-020.pth", clipcap_path="models/coco_prefix-020.pt")
# if args.dataset == "coco":
# with open("dataset/dataset_cache/coco_train_dataset.pkl", "rb") as f:
#     train_dataset = pickle.load(f)
with open("dataset/dataset_cache/coco_test_dataset.pkl", "rb") as f:
    test_dataset = pickle.load(f)
# elif args.dataset == "conceptual":
# with open("dataset/dataset_cache/conceptual_captions_subset_train_dataset.pkl", "rb") as f:
#     train_dataset = pickle.load(f)
# with open("dataset/dataset_cache/conceptual_captions_subset_test_dataset.pkl", "rb") as f:
#     test_dataset = pickle.load(f)
data_path = 'dataset/'
prefix_length = 40
normalize_prefix = True
coco_test_dataset_A = CocoDataset(root = data_path, transform=preprocess,data_mode="test", prefix_length=prefix_length, normalize_prefix=normalize_prefix, datasize=100)

df = pd.read_csv("exp/wo_BP_100_1/agent_A_99.csv")
print(df["after_w"][0])

# print 

# coco_test_dataset_A = CocoDataset(root = data_path, transform=preprocess,data_mode="train", prefix_length=prefix_length, normalize_prefix=normalize_prefix, datasize=500)
# train_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=args.num_workers)
# test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=args.num_workers)
train_dataloader = torch.utils.data.DataLoader(coco_test_dataset_A, batch_size=16, shuffle=True, num_workers=args.num_workers)
test_dataloader = torch.utils.data.DataLoader(coco_test_dataset_A, batch_size=16, shuffle=False, num_workers=args.num_workers)

batch = next(iter(train_dataloader))
print(batch[3][0])

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
for i in range(100):
    train_dataloader.dataset.captions[i] = df["after_w"][i]
    train_dataloader.dataset.gpt_tokens[i] = torch.tensor(tokenizer.encode(df["after_w"][i]))

batch = next(iter(train_dataloader))
print(batch[3][0])

print("train_dataloader:", len(train_dataloader))

# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=args.num_workers)
# test_dataloader = torch.utils.data.DataLoader(tfest_dataset, batch_size=32, shuffle=False, num_workers=args.num_workers)

def pretrain_clipcap(CLIP_Net, train_loader, test_loader, model, save_dir, epochs, lr = 2e-6, warmup_steps_percent = 0.1, output_prefix = "clipcap", save_every = 1, train_mode = "pretain", initial_epoch = 0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=2, lora_alpha=32, lora_dropout=0.1, target_modules=["c_fc"])
    print(type(model.gpt))
    state_dict = model.state_dict()
    for key in state_dict.keys():
        print(key)
    model.gpt = get_peft_model(model.gpt, peft_config)
    model.gpt.print_trainable_parameters()

    model = model.to(device)
    model.train()

    CLIP_Net.eval()
    CLIP_Net.to(device)

    if initial_epoch > 0:
        model.load_state_dict(torch.load(os.path.join(save_dir, f"{output_prefix}-{initial_epoch:03d}.pt")))
        print("load model from", os.path.join(save_dir, f"{output_prefix}-{initial_epoch:03d}.pt"))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    warmup_steps = int(warmup_steps_percent * epochs * len(train_loader))
    print("clipcap lr:", lr)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_loader)
    )
    
    loss_list = []
    test_loss_list = []
    
    for epoch in range(initial_epoch+1, epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_loader), desc=output_prefix)
        for idx, batch in enumerate(train_loader):
            model.zero_grad() 
            img, _, _, gpt_token, gpt_mask, _  = batch

            img, gpt_token, gpt_mask = img.to(device), gpt_token.to(device), gpt_mask.to(device)

            prefix = CLIP_Net.encode_image(img).to(device, dtype=torch.float32)

            outputs = model(gpt_token, prefix, gpt_mask)
            logits = outputs.logits[:, train_loader.dataset.prefix_length - 1: -1]

            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), gpt_token.flatten(), ignore_index=0)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})
            progress.update()

            loss_list.append(loss.item())

            if (idx + 1) % 10 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(save_dir, f"{output_prefix}_latest.pt"),
                )       
                np.save(os.path.join(save_dir, f"{output_prefix}_loss.npy"), loss_list)

        progress.close()
        np.save(os.path.join(save_dir, f"{output_prefix}_loss.npy"), np.array(loss_list))

        if "MHNG" in train_mode:
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, f"{output_prefix}-{train_mode}.pt"),
            )

        elif train_mode == "pretrain":
            print(f">>> Evaluating")
            test_loss = 0
            for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc="eval"):
                model.zero_grad() 
                img, _, _, gpt_token, gpt_mask,_ = batch

                img, gpt_token, gpt_mask = img.to(device), gpt_token.to(device), gpt_mask.to(device)

                prefix = CLIP_Net.encode_image(img).to(device, dtype=torch.float32)

                outputs = model(gpt_token, prefix, gpt_mask)
                logits = outputs.logits[:, test_loader.dataset.prefix_length - 1: -1]

                loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), gpt_token.flatten(), ignore_index=0)
                test_loss += loss.item()

            test_loss /= len(test_loader)
            test_loss_list.append(test_loss)
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, f"{output_prefix}-{epoch:03d}.pt"),
            )

            
            np.save(os.path.join(save_dir, f"{output_prefix}_test_loss.npy"), test_loss_list)
        
    return model

model = pretrain_clipcap(clip_model, train_dataloader, test_dataloader, agent.ClipCap, "WithSign", 10, lr = 2e-5, warmup_steps_percent = 0.1, output_prefix = "clipcap", save_every = 1, train_mode = "pretrain", initial_epoch = args.initial_epoch)
