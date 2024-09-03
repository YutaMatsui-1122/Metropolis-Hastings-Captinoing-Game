from one_agent import OneAgent
from utils import * 
import pickle, clip, argparse
import pandas as pd
import time, torch, os, copy
import matplotlib.pyplot as plt
import sys
from sample_captions import *


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

dataloader = torch.utils.data.DataLoader(coco_test_dataset_A, batch_size=datasize, shuffle=False, num_workers=1)

agentA = OneAgent(agent_name='A')
agentA = agentA.to(device)
agentB = OneAgent(agent_name='B')
agentB = agentB.to(device)

agentA.load_pretrain(probvlm_path="models/probVLM_coco_prefix-036.pth", clipcap_path="models/coco_prefix-020.pt")
agentB.load_pretrain(probvlm_path="models/probVLM_conceptual_prefix-020.pth", clipcap_path="models/conceptual_prefix-020.pt")

acceptance_rate = np.zeros((4, 4))
acceptance_probability = np.zeros((4, 4))

batch = next(iter(dataloader))
img = batch[0].to(device)


plt.imshow(img[0].cpu().permute(1,2,0))
plt.savefig("image_for_acceptance_rate.pdf")
z = agentA.CLIP_Net.encode_image(img)[0].to(device)

for Li_cat, Li_captions in all_captions.items():
    for Sp_cat, Sp_captions in all_captions.items():
        Li_tokens = tokenize(Li_captions).to(device)
        Sp_tokens = tokenize(Sp_captions).to(device)
        for Li_cap in Li_tokens:
            for Sp_cap in Sp_tokens:
                mu_li, alpha_li, beta_li = agentA.text_encoder(Li_cap.unsqueeze(0))
                mu_sp, alpha_sp, beta_sp = agentB.text_encoder(Sp_cap.unsqueeze(0))

                p_li = -agentA.GGL(mu_li, alpha_li, beta_li, z)
                p_sp = -agentB.GGL(mu_sp, alpha_sp, beta_sp, z)
                r = np.exp(np.where((p_sp-p_li).detach().cpu().numpy()<0,(p_sp-p_li).detach().cpu().numpy(),0))
                u = np.random.rand()
                acceptance_probability[Li_cat-1, Sp_cat-1] += r
                if u < r:
                    acceptance_rate[Li_cat-1, Sp_cat-1] += 1
        print(acceptance_rate)

# show heatmap
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt