# Evaluate the shared sign likelihood between the agents in the captioning game
import os
import json
import torch
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import argparse
from one_agent import OneAgent

# Argument Parser
def get_args():
    parser = argparse.ArgumentParser(description="Evaluate the shared sign likelihood between agents")
    parser.add_argument("--exp_name", type=str, default="mhcg_person_only_0", help="Experiment name")
    parser.add_argument("--dataset_prefix", type=str, default="coco_2017_common_person_only", help="Dataset prefix")
    parser.add_argument("--observation_file", type=str, default="train_split_dataset_mhcg", help="Observation dataset file")
    parser.add_argument("--MH_iter", type=int, default=3, help="Number of Metropolis-Hastings iterations")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of EM iterations")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for dataloader")
    parser.add_argument("--device", type=str, default="cuda:3" if torch.cuda.is_available() else "cpu", help="Computation device")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for dataloader")
    return parser.parse_args()

# Function to load agents
def load_agents(exp_dir, dataset_prefix, em_iter, device):
    agentA = OneAgent(agent_name='A', device=device, clip_arch="ViT-B/16").to(device)
    agentB = OneAgent(agent_name='B', device=device, clip_arch="ViT-B/32").to(device)

    if em_iter != -1:
        with open(f"{exp_dir}/args.json", 'r') as f:
            exp_args = json.load(f)
        for agent in [agentA, agentB]:
            agent.lora_setting(r=exp_args["lora_r"], alpha=exp_args["lora_alpha"], dropout=exp_args["lora_dropout"])
            agent.load_pretrain(
                probvlm_path=f"{exp_dir}/{agent.agent_name}/probvlm_{agent.agent_name}_{em_iter}-epoch-0.pth",
                clipcap_path=f"{exp_dir}/{agent.agent_name}/clipcap_{agent.agent_name}_{em_iter}-009.pt",
                strict_clipcap=False
            )
    else:
        agentA.load_pretrain(
            probvlm_path=f"pretrain_models/{dataset_prefix}/COCO_A/probvlm/probvlm-epoch-49.pth",
            clipcap_path=f"pretrain_models/{dataset_prefix}/COCO_A/clipcap/clipcap_019.pt",
            strict_clipcap=False
        )
        agentB.load_pretrain(
            probvlm_path=f"pretrain_models/{dataset_prefix}/COCO_B/probvlm/probvlm-epoch-49.pth",
            clipcap_path=f"pretrain_models/{dataset_prefix}/COCO_B/clipcap/clipcap_019.pt",
            strict_clipcap=False
        )
    
    return agentA, agentB

# Function to load shared signs
def load_shared_signs(exp_dir, agentA, agentB, em_iter, mh_iter):
    sign_key = "initial" if em_iter == -1 else f"EM_{em_iter}_MH_{mh_iter}"
    shared_sign_A = pd.read_csv(f"{exp_dir}/{agentA.agent_name}/agent_A_sign.csv")[sign_key]
    shared_sign_B = pd.read_csv(f"{exp_dir}/{agentB.agent_name}/agent_B_sign.csv")[sign_key]
    return shared_sign_A, shared_sign_B

# Function to calculate log likelihoods
def calculate_log_likelihoods(observation_dataloader, shared_sign_A, shared_sign_B, agentA, agentB, device):
    nll_A_sign_A_list, nll_B_sign_A_list = [], []
    nll_A_sign_B_list, nll_B_sign_B_list = [], []
    nll_sign_A_list, nll_sign_B_list = [], []

    for batch in tqdm.tqdm(observation_dataloader):
        images = batch["image"].to(device)
        index = batch["index"].tolist()
        captionA, captionB = shared_sign_A[index], shared_sign_B[index]

        nll_A_sign_A = agentA.calculate_p_z_w(images, captionA)
        nll_B_sign_A = agentB.calculate_p_z_w(images, captionA)
        nll_A_sign_B = agentA.calculate_p_z_w(images, captionB)
        nll_B_sign_B = agentB.calculate_p_z_w(images, captionB)

        nll_A_sign_A_list.append(nll_A_sign_A)
        nll_B_sign_A_list.append(nll_B_sign_A)
        nll_A_sign_B_list.append(nll_A_sign_B)
        nll_B_sign_B_list.append(nll_B_sign_B)

        nll_sign_A_list.append(nll_A_sign_A + nll_B_sign_A)
        nll_sign_B_list.append(nll_A_sign_B + nll_B_sign_B)

    return {
        "nll_A_sign_A": torch.cat(nll_A_sign_A_list).mean().item(),
        "nll_B_sign_A": torch.cat(nll_B_sign_A_list).mean().item(),
        "nll_A_sign_B": torch.cat(nll_A_sign_B_list).mean().item(),
        "nll_B_sign_B": torch.cat(nll_B_sign_B_list).mean().item(),
        "nll_sign_A": torch.cat(nll_sign_A_list).mean().item(),
        "nll_sign_B": torch.cat(nll_sign_B_list).mean().item(),
    }

# Function to plot and save log likelihoods
def plot_log_likelihoods(nll_dict, exp_eval_dir):
    for key, values in nll_dict.items():
        plt.plot(values, label=key)
        plt.xlabel("MH Iteration", fontsize=14)
        plt.ylabel("Log Likelihood", fontsize=14)
        plt.savefig(f"{exp_eval_dir}/{key}.png")
        plt.clf()

# Function to save log likelihoods to file
def save_log_likelihoods(nll_dict, exp_eval_dir):
    for key, values in nll_dict.items():
        with open(f"{exp_eval_dir}/{key}.txt", "w") as f:
            f.write("\n".join(map(str, values)))

# Main evaluation function
def main(args):
    exp_dir = f"exp/{args.exp_name}"
    exp_eval_dir = f"exp_eval/{args.exp_name}/likelihood"
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(exp_eval_dir, exist_ok=True)

    nll_results = {key: [] for key in ["nll_A_sign_A", "nll_B_sign_A", "nll_A_sign_B", "nll_B_sign_B", "nll_sign_A", "nll_sign_B"]}

    for em_iter in range(-1, args.num_epochs):
        agentA, agentB = load_agents(exp_dir, args.dataset_prefix, em_iter, args.device)

        with open(f"dataset/dataset_cache/{args.dataset_prefix}/{args.observation_file}.pkl", "rb") as f:
            observation_dataset = pickle.load(f)
            observation_dataset.prefix_length = agentA.prefix_length

        for mh_iter in range(args.MH_iter):
            shared_sign_A, shared_sign_B = load_shared_signs(exp_dir, agentA, agentB, em_iter, mh_iter)

            observation_dataloader = torch.utils.data.DataLoader(
                observation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
            )

            nll_values = calculate_log_likelihoods(observation_dataloader, shared_sign_A, shared_sign_B, agentA, agentB, args.device)

            for key, value in nll_values.items():
                nll_results[key].append(value)

            if em_iter == -1:
                break

    plot_log_likelihoods(nll_results, exp_eval_dir)
    save_log_likelihoods(nll_results, exp_eval_dir)

    print("Evaluation completed.")

if __name__ == "__main__":
    args = get_args()
    main(args)
