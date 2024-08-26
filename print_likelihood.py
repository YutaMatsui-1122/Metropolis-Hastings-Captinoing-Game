from one_agent import OneAgent
from utils import * 
import pickle, clip, argparse
import pandas as pd
import time, torch, os, copy
import matplotlib.pyplot as plt
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
parser.add_argument('--exp_name', default="debug")
parser.add_argument('--MH_iter', default=100, type=int)
parser.add_argument('--annealing', default="None")
parser.add_argument('--mode', default="MHNG")
args = parser.parse_args()

datasize = 100
epochs = 100

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
data_path = 'dataset/'
prefix_length = 40
normalize_prefix = True
coco_test_dataset_A = CocoDataset(root = data_path, transform=preprocess,data_mode="test", prefix_length=prefix_length, normalize_prefix=normalize_prefix,datasize=datasize)
coco_test_dataset_B = copy.deepcopy(coco_test_dataset_A)


num_workers = 1



coco_test_loader_fix_A = torch.utils.data.DataLoader(coco_test_dataset_A, batch_size=datasize, shuffle=False, num_workers=num_workers)
coco_test_loader_fix_B = torch.utils.data.DataLoader(coco_test_dataset_B, batch_size=datasize, shuffle=False, num_workers=num_workers)

agentA = OneAgent(agent_name='A')
agentA = agentA.to(device)
agentB = OneAgent(agent_name='B')
agentB = agentB.to(device)

agentA.load_pretrain(probvlm_path="models/probVLM_coco_prefix-020.pth", clipcap_path="models/coco_prefix-020.pt")
agentB.load_pretrain(probvlm_path="models/probVLM_conceptual_prefix-020.pth", clipcap_path="models/conceptual_prefix-020.pt")

clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
data_path = 'dataset/'
prefix_length = 40
normalize_prefix = True

exp_root_path = f"exp/wo_BP_{datasize}"

exp_list = [0,1,2,3,4,5,6,7,8,9,20,21,22,23,24,25,26,27,28,29]
print(len(exp_list))

batch = next(iter(coco_test_loader_fix_A))
img = batch[0].to(device)

z = agentA.CLIP_Net.encode_image(img).to(device)
print(z.shape)

log_likelihood_list_A = np.zeros((len(exp_list), epochs))
log_likelihood_list_B = np.zeros((len(exp_list), epochs))
for e, exp in enumerate(exp_list):
    exp_path = exp_root_path + f"_{exp}"
    for i in range(epochs):

        file_path_A = os.path.join(exp_path, f"agent_A_{i}.csv")
        df_A = pd.read_csv(file_path_A)

        sign_A = df_A["before_w"].to_list()
        sige_A_token = tokenize(sign_A).to(device)


        mu_A, alpha_A, beta_A = agentA.text_encoder(sige_A_token)
        mu_B, alpha_B, beta_B = agentB.text_encoder(sige_A_token)
        
        log_likelihood_A = -agentA.GGL(mu_A, alpha_A, beta_A, z) - agentB.GGL(mu_B, alpha_B, beta_B, z)
        

        log_likelihood_list_A[e, i] = log_likelihood.sum().item()

        file_path_B = os.path.join(exp_path, f"agent_B_{i}.csv")
        df_B = pd.read_csv(file_path_B)

        sign_B = df_B["before_w"].to_list()
        sige_B_token = tokenize(sign_B).to(device)

        mu_A, alpha_A, beta_A = agentA.text_encoder(sige_B_token)
        mu_B, alpha_B, beta_B = agentB.text_encoder(sige_B_token)

        log_likelihood = -agentA.GGL(mu_A, alpha_A, beta_A, z) - agentB.GGL(mu_B, alpha_B, beta_B, z)
        log_likelihood_list_B[e, i] = log_likelihood.sum().item()