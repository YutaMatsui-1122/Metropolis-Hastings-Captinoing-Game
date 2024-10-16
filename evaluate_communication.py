# Evaluate the communication between the agents in the vlm captioning game
# Two kinds of evaluation:
# 1. Evaluate the likelihood of the latent representation given the shared sign
# 2. Evaluate the captioning performance for each pretraining dataset

# 0.1. Import the required libraries
import torch
import pickle
from one_agent import OneAgent
from utils import * 
import pickle, clip, argparse
from torch.nn import functional as nnf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from bert_score import score as bert_score
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType
import copy

# 0.2. Define the device
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# 0.3. Set the experiment name, directory, and the dataset
exp_name = "mhcg_derpp_0.05_1"
exp_dir = f"exp/{exp_name}"
observation_file = "communication_coco_5000_cc3m_5000"
eval_sign = True
eval_captioning = False

# 1. Evaluate the likelihood of the latent representation given the shared sign
# 1.1. Load the trained agents

nll_A_sign_A_list_for_plot = []
nll_B_sign_A_list_for_plot = []
nll_A_sign_B_list_for_plot = []
nll_B_sign_B_list_for_plot = []

nll_sign_A_list_for_plot = []
nll_sign_B_list_for_plot = []

for em_iter in [-1, 20, 15, 10]:
    print("EM Iteration:", em_iter)
    # Load the trained agents
    agentA = OneAgent(agent_name='A', device=device)
    agentB = OneAgent(agent_name='B', device=device)
    agentA_pretrain = OneAgent(agent_name='A', device=device)
    agentB_pretrain = OneAgent(agent_name='B', device=device)
    agentA = agentA.to(device)
    agentB = agentB.to(device)
    agentA_pretrain = agentA_pretrain.to(device)
    agentB_pretrain = agentB_pretrain.to(device)

    # Load the trained agents' parameters
    print("Load the trained agents' parameters")
    if em_iter != -1:
        agentA.lora_setting()
        agentA.load_pretrain(probvlm_path=f"{exp_dir}/{agentA.agent_name}/probvlm_A-epoch-9.pth", clipcap_path=f"{exp_dir}/{agentA.agent_name}/clipcap_A_{em_iter}-009.pt")
        agentB.lora_setting()
        agentB.load_pretrain(probvlm_path=f"{exp_dir}/{agentB.agent_name}/probvlm_B-epoch-9.pth", clipcap_path=f"{exp_dir}/{agentB.agent_name}/clipcap_B_{em_iter}-009.pt")
    agentA_pretrain.load_pretrain(probvlm_path="models/official_model/probvlm/CC3M/probvlm_0.2_0.3_20-epoch-15.pth", clipcap_path="models/official_model/clipcap_conceptual_weights.pt", strict_clipcap=False)
    agentB_pretrain.load_pretrain(probvlm_path="models/official_model/probvlm/COCO/probvlm_0.2_0.3_20-epoch-99.pth", clipcap_path="models/official_model/clipcap_coco_weights.pt", strict_clipcap=False)
    print("Load the trained agents' parameters done")
    # 1.2. Load the shared sign and set the observation dataset
    # Load the shared sign
    if eval_sign:
        if em_iter == -1:
            shared_sign_A = pd.read_csv(f"{exp_dir}/{agentA.agent_name}/agent_A_sign.csv")[f"initial"]
            shared_sign_B = pd.read_csv(f"{exp_dir}/{agentB.agent_name}/agent_B_sign.csv")[f"initial"]
        else:
            shared_sign_A = pd.read_csv(f"{exp_dir}/{agentA.agent_name}/agent_A_sign.csv")[f"EM_{em_iter}_MH_9"]
            shared_sign_B = pd.read_csv(f"{exp_dir}/{agentB.agent_name}/agent_B_sign.csv")[f"EM_{em_iter}_MH_9"]

        with open(f"dataset/dataset_cache/{observation_file}.pkl", "rb") as f:
            observation_dataset = pickle.load(f)
            observation_dataset.prefix_length = agentA.prefix_length

        observation_dataloader = torch.utils.data.DataLoader(observation_dataset, batch_size=64, shuffle=False, num_workers=8)

        # 1.3. Calculate the likelihood of the latent representation given the shared sign
        # Calculate the likelihood of the latent representation given the shared sign
        # nll_sign_A_list = []
        # nll_sign_B_list = []
        nll_sign_A_pretrain_list = []
        nll_sign_B_pretrain_list = []
        nll_A_sign_A_pretrain_list = []
        nll_B_sign_A_pretrain_list = []
        nll_A_sign_B_pretrain_list = []
        nll_B_sign_B_pretrain_list = []
        for batch in tqdm.tqdm(observation_dataloader):
            # Set the batch
            images = batch["image"].to(device)
            index = batch["index"].tolist()
            captionA = shared_sign_A[index]
            captionB = shared_sign_B[index]

            # Calculate the likelihood of the latent representation given the shared sign

            nll_A_sign_A_pretrain = agentA_pretrain.calculate_p_z_w(images, captionA)
            nll_B_sign_A_pretrain = agentB_pretrain.calculate_p_z_w(images, captionA)
            nll_A_sign_B_pretrain = agentA_pretrain.calculate_p_z_w(images, captionB)
            nll_B_sign_B_pretrain = agentB_pretrain.calculate_p_z_w(images, captionB)

            nll_A_sign_A_pretrain_list.append(nll_A_sign_A_pretrain)
            nll_B_sign_A_pretrain_list.append(nll_B_sign_A_pretrain)
            nll_A_sign_B_pretrain_list.append(nll_A_sign_B_pretrain)
            nll_B_sign_B_pretrain_list.append(nll_B_sign_B_pretrain)

            nll_sign_A_pretrain_list.append(nll_A_sign_A_pretrain + nll_B_sign_A_pretrain)
            nll_sign_B_pretrain_list.append(nll_A_sign_B_pretrain + nll_B_sign_B_pretrain)

        # Save the likelihood of the latent representation given the shared sign
        # calculate the sum of the negative log likelihood
        # nll_sign_A = torch.cat(nll_sign_A_list).mean().item()
        # nll_sign_B_list = torch.cat(nll_siggn_B_list).mean().item()

        nll_A_sign_A_pretrain = torch.cat(nll_A_sign_A_pretrain_list).mean().item()
        nll_B_sign_A_pretrain = torch.cat(nll_B_sign_A_pretrain_list).mean().item()
        nll_A_sign_B_pretrain = torch.cat(nll_A_sign_B_pretrain_list).mean().item()
        nll_B_sign_B_pretrain = torch.cat(nll_B_sign_B_pretrain_list).mean().item()

        nll_sign_A_pretrain = torch.cat(nll_sign_A_pretrain_list).mean().item()
        nll_sign_B_pretrain = torch.cat(nll_sign_B_pretrain_list).mean().item()

        print(f"EM_{em_iter}")
        # print(f"nll_sign_A: {nll_sign_A}")
        # print(f"nll_sign_B: {nll_sign_B_list}")
        print(f"nll_A_sign_A_pretrain: {nll_A_sign_A_pretrain}")
        print(f"nll_B_sign_A_pretrain: {nll_B_sign_A_pretrain}")
        print(f"nll_A_sign_B_pretrain: {nll_A_sign_B_pretrain}")
        print(f"nll_B_sign_B_pretrain: {nll_B_sign_B_pretrain}")

        print(f"nll_sign_A_pretrain: {nll_sign_A_pretrain}")
        print(f"nll_sign_B_pretrain: {nll_sign_B_pretrain}")

        nll_A_sign_A_list_for_plot.append(nll_A_sign_A_pretrain)
        nll_B_sign_A_list_for_plot.append(nll_B_sign_A_pretrain)
        nll_A_sign_B_list_for_plot.append(nll_A_sign_B_pretrain)
        nll_B_sign_B_list_for_plot.append(nll_B_sign_B_pretrain)
        nll_sign_A_list_for_plot.append(nll_sign_A_pretrain)
        nll_sign_B_list_for_plot.append(nll_sign_B_pretrain)

        # 1.4. Save the likelihood of the latent representation given the shared sign
        # Plot the likelihood of the latent representation given the shared sign

        plt.plot(nll_A_sign_A_list_for_plot, label="nll_A_sign_A_pretrain")
        plt.xlabel("EM Iteration", fontsize=14)
        plt.ylabel("Log Likelihood", fontsize=14)
        plt.savefig(f"{exp_dir}/nll_A_sign_A_pretrain.png")
        plt.clf()
        plt.plot(nll_B_sign_A_list_for_plot, label="nll_B_sign_A_pretrain")
        plt.xlabel("EM Iteration", fontsize=14)
        plt.ylabel("Log Likelihood", fontsize=14)
        plt.savefig(f"{exp_dir}/nll_B_sign_A_pretrain.png")
        plt.clf()
        plt.plot(nll_A_sign_B_list_for_plot, label="nll_A_sign_B_pretrain")
        plt.xlabel("EM Iteration", fontsize=14)
        plt.ylabel("Log Likelihood", fontsize=14)
        plt.savefig(f"{exp_dir}/nll_A_sign_B_pretrain.png")
        plt.clf()
        plt.plot(nll_B_sign_B_list_for_plot, label="nll_B_sign_B_pretrain")
        plt.ylabel("Log Likelihood", fontsize=14)
        plt.xlabel("EM Iteration", fontsize=14)
        plt.savefig(f"{exp_dir}/nll_B_sign_B_pretrain.png")
        plt.clf()

        plt.plot(nll_sign_A_list_for_plot, label="nll_sign_A")
        plt.xlabel("EM Iteration", fontsize=14)
        plt.ylabel("Log Likelihood", fontsize=14)
        plt.savefig(f"{exp_dir}/nll_sign_A.png")
        plt.clf()
        plt.plot(nll_sign_B_list_for_plot, label="nll_sign_B")
        plt.xlabel("EM Iteration", fontsize=14)
        plt.ylabel("Log Likelihood", fontsize=14)
        plt.savefig(f"{exp_dir}/nll_sign_B.png")
        plt.clf()

        # Save nll_sign_A_list_for_plot and nll_sign_B_list_for_plot in txt files
        with open(f"{exp_dir}/nll_A_sign_A_pretrain_list.txt", "w") as f:
            f.write("\n".join(map(str, nll_A_sign_A_list_for_plot)))

        with open(f"{exp_dir}/nll_B_sign_A_pretrain_list.txt", "w") as f:
            f.write("\n".join(map(str, nll_B_sign_A_list_for_plot)))
            
        with open(f"{exp_dir}/nll_A_sign_B_pretrain_list.txt", "w") as f:
            f.write("\n".join(map(str, nll_A_sign_B_list_for_plot)))

        with open(f"{exp_dir}/nll_B_sign_B_pretrain_list.txt", "w") as f:
            f.write("\n".join(map(str, nll_B_sign_B_list_for_plot)))

        with open(f"{exp_dir}/nll_sign_A_list_for_plot.txt", "w") as f:
            f.write("\n".join(map(str, nll_sign_A_list_for_plot)))

        with open(f"{exp_dir}/nll_sign_B_list_for_plot.txt", "w") as f:
            f.write("\n".join(map(str, nll_sign_B_list_for_plot)))




    ######################################################################################



    # 2. Evaluate the captioning performance for each pretraining dataset
    # 2.1. Load pretraining datasets
    if eval_captioning:
        # Load the pretraining datasets
        with open("dataset/dataset_cache/coco_train_dataset_10000_20000_imageid.pkl", "rb") as f:
            coco_train_dataset = pickle.load(f)
            coco_train_dataset.prefix_length = agentA.prefix_length

        with open("dataset/dataset_cache/conceptual_train_dataset_10000.pkl", "rb") as f:
            conceptual_train_dataset = pickle.load(f)
            conceptual_train_dataset.prefix_length = agentA.prefix_length

        # Set the dataloaders
        coco_train_dataloader = torch.utils.data.DataLoader(coco_train_dataset, batch_size=16, shuffle=False, num_workers=1)
        conceptual_train_dataloader = torch.utils.data.DataLoader(conceptual_train_dataset, batch_size=16, shuffle=False, num_workers=1)

        # 2.4. Evaluate the captioning performance for each pretraining dataset
        # Evaluate the captioning performance for coco_train_dataset

        coco_train_agent_A_loss = []
        coco_train_agent_B_loss = []
        coco_train_agent_A_pretrain_loss = []
        coco_train_agent_B_pretrain_loss = []


        for batch in tqdm.tqdm(coco_train_dataloader):
            # Set the batch
            images = batch[0].to(device)
            captions = batch[1]
            gpt_tokens = batch[3].to(device)
            gpt_masks = batch[4].to(device)

            # Encode the images
            mu_img_A, alpha_img_A, beta_img_A, z_img_A = agentA.image_encoder(images)
            mu_img_B, alpha_img_B, beta_img_B, z_img_B = agentB.image_encoder(images)
            mu_img_A_pretrain, alpha_img_A_pretrain, beta_img_A_pretrain, z_img_A_pretrain = agentA_pretrain.image_encoder(images)
            mu_img_B_pretrain, alpha_img_B_pretrain, beta_img_B_pretrain, z_img_B_pretrain = agentB_pretrain.image_encoder(images)

            outputs_A = agentA.ClipCap(gpt_tokens, mu_img_A, gpt_masks)
            logits_A = outputs_A.logits[:, coco_train_dataset.prefix_length - 1: -1]
            logits_A = logits_A.reshape(-1, logits_A.shape[-1])

            outputs_B = agentB.ClipCap(gpt_tokens, mu_img_B, gpt_masks)
            logits_B = outputs_B.logits[:, coco_train_dataset.prefix_length - 1: -1]
            logits_B = logits_B.reshape(-1, logits_B.shape[-1])

            outputs_A_pretrain = agentA_pretrain.ClipCap(gpt_tokens, mu_img_A_pretrain, gpt_masks)
            logits_A_pretrain = outputs_A_pretrain.logits[:, coco_train_dataset.prefix_length - 1: -1]
            logits_A_pretrain = logits_A_pretrain.reshape(-1, logits_A_pretrain.shape[-1])

            outputs_B_pretrain = agentB_pretrain.ClipCap(gpt_tokens, mu_img_B_pretrain, gpt_masks)
            logits_B_pretrain = outputs_B_pretrain.logits[:, coco_train_dataset.prefix_length - 1: -1]
            logits_B_pretrain = logits_B_pretrain.reshape(-1, logits_B_pretrain.shape[-1])


            # Calculate the loss
            loss_A = nnf.cross_entropy(logits_A, gpt_tokens.flatten(), ignore_index=0, reduction="sum").cpu()
            loss_B = nnf.cross_entropy(logits_B, gpt_tokens.flatten(), ignore_index=0, reduction="sum").cpu()
            loss_A_pretrain = nnf.cross_entropy(logits_A_pretrain, gpt_tokens.flatten(), ignore_index=0, reduction="sum").cpu()
            loss_B_pretrain = nnf.cross_entropy(logits_B_pretrain, gpt_tokens.flatten(), ignore_index=0, reduction="sum").cpu()

            coco_train_agent_A_loss.append(loss_A.item())
            coco_train_agent_B_loss.append(loss_B.item())
            coco_train_agent_A_pretrain_loss.append(loss_A_pretrain.item())
            coco_train_agent_B_pretrain_loss.append(loss_B_pretrain.item())

            # all variables to cpu
            images = images.cpu()
            gpt_tokens = gpt_tokens.cpu()
            gpt_masks = gpt_masks.cpu()     
            logits_A = logits_A.cpu()
            logits_B = logits_B.cpu()


        print("COCO Train Dataset")
        print(f"Agent A loss: {np.mean(coco_train_agent_A_loss)}")
        print(f"Agent B loss: {np.mean(coco_train_agent_B_loss)}")
        print(f"Agent A pretrain loss: {np.mean(coco_train_agent_A_pretrain_loss)}")
        print(f"Agent B pretrain loss: {np.mean(coco_train_agent_B_pretrain_loss)}")

        # Save the captioning performance for coco_train_dataset
        with open(f"{exp_dir}/EM_{em_iter}_coco_train_agent_A_loss.txt", "w") as f:
            f.write("\n".join(map(str, coco_train_agent_A_loss)))
        with open(f"{exp_dir}/EM_{em_iter}_coco_train_agent_B_loss.txt", "w") as f:
            f.write("\n".join(map(str, coco_train_agent_B_loss)))

        with open(f"{exp_dir}/coco_train_agent_A_pretrain_loss.txt", "w") as f:
            f.write("\n".join(map(str, coco_train_agent_A_pretrain_loss)))
        with open(f"{exp_dir}/coco_train_agent_B_pretrain_loss.txt", "w") as f:
            f.write("\n".join(map(str, coco_train_agent_B_pretrain_loss)))

        # Evaluate the captioning performance for conceptual_train_dataset
        conceptual_train_agent_A_loss = []
        conceptual_train_agent_B_loss = []
        conceptual_train_agent_A_pretrain_loss = []
        conceptual_train_agent_B_pretrain_loss = []

        for batch in tqdm.tqdm(conceptual_train_dataloader):
            # Set the batch
            images = batch[0].to(device)
            captions = batch[1]
            gpt_tokens = batch[3].to(device)
            gpt_masks = batch[4].to(device)

            # Encode the images
            mu_img_A, alpha_img_A, beta_img_A, z_img_A = agentA.image_encoder(images)
            mu_img_B, alpha_img_B, beta_img_B, z_img_B = agentB.image_encoder(images)
            mu_img_A_pretrain, alpha_img_A_pretrain, beta_img_A_pretrain, z_img_A_pretrain = agentA_pretrain.image_encoder(images)
            mu_img_B_pretrain, alpha_img_B_pretrain, beta_img_B_pretrain, z_img_B_pretrain = agentB_pretrain.image_encoder(images)    
            
            outputs_A = agentA.ClipCap(gpt_tokens, mu_img_A, gpt_masks)
            logits_A = outputs_A.logits[:, conceptual_train_dataset.prefix_length - 1: -1]
            logits_A = logits_A.reshape(-1, logits_A.shape[-1])

            outputs_B = agentB.ClipCap(gpt_tokens, mu_img_B, gpt_masks)
            logits_B = outputs_B.logits[:, conceptual_train_dataset.prefix_length - 1: -1]
            logits_B = logits_B.reshape(-1, logits_B.shape[-1])

            outputs_A_pretrain = agentA_pretrain.ClipCap(gpt_tokens, mu_img_A_pretrain, gpt_masks)
            logits_A_pretrain = outputs_A_pretrain.logits[:, conceptual_train_dataset.prefix_length - 1: -1]
            logits_A_pretrain = logits_A_pretrain.reshape(-1, logits_A_pretrain.shape[-1])

            outputs_B_pretrain = agentB_pretrain.ClipCap(gpt_tokens, mu_img_B_pretrain, gpt_masks)
            logits_B_pretrain = outputs_B_pretrain.logits[:, conceptual_train_dataset.prefix_length - 1: -1]
            logits_B_pretrain = logits_B_pretrain.reshape(-1, logits_B_pretrain.shape[-1])

            # Calculate the loss
            loss_A = nnf.cross_entropy(logits_A, gpt_tokens.flatten(), ignore_index=0, reduction="sum")
            loss_B = nnf.cross_entropy(logits_B, gpt_tokens.flatten(), ignore_index=0, reduction="sum")
            loss_A_pretrain = nnf.cross_entropy(logits_A_pretrain, gpt_tokens.flatten(), ignore_index=0, reduction="sum")
            loss_B_pretrain = nnf.cross_entropy(logits_B_pretrain, gpt_tokens.flatten(), ignore_index=0, reduction="sum")

            conceptual_train_agent_A_loss.append(loss_A.item())
            conceptual_train_agent_B_loss.append(loss_B.item())
            conceptual_train_agent_A_pretrain_loss.append(loss_A_pretrain.item())
            conceptual_train_agent_B_pretrain_loss.append(loss_B_pretrain.item())
        
        print("Conceptual Train Dataset")
        print(f"Agent A loss: {np.mean(conceptual_train_agent_A_loss)}")
        print(f"Agent B loss: {np.mean(conceptual_train_agent_B_loss)}")
        print(f"Agent A pretrain loss: {np.mean(conceptual_train_agent_A_pretrain_loss)}")
        print(f"Agent B pretrain loss: {np.mean(conceptual_train_agent_B_pretrain_loss)}")

        # Save the captioning performance for conceptual_train_dataset
        with open(f"{exp_dir}/EM_{em_iter}_conceptual_train_agent_A_loss.txt", "w") as f:
            f.write("\n".join(map(str, conceptual_train_agent_A_loss)))
        with open(f"{exp_dir}/EM_{em_iter}_conceptual_train_agent_B_loss.txt", "w") as f:
            f.write("\n".join(map(str, conceptual_train_agent_B_loss)))

        with open(f"{exp_dir}/conceptual_train_agent_A_pretrain_loss.txt", "w") as f:
            f.write("\n".join(map(str, conceptual_train_agent_A_pretrain_loss)))

        with open(f"{exp_dir}/conceptual_train_agent_B_pretrain_loss.txt", "w") as f:
            f.write("\n".join(map(str, conceptual_train_agent_B_pretrain_loss)))
