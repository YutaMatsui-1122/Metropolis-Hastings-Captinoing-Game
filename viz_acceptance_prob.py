from sample_captions import *
from PIL import Image
import pandas as pd
import torch
import clip
from one_agent import OneAgent
from utils import *

if __name__ == "__main__":

    exp_dir = "exp/mhcg_derpp_0.05_1"

    device = "cuda:3" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device="cpu")

    pretrain = False
    agent_name = "coco"

    agent = OneAgent(agent_name='A', device=device)
    if pretrain:
        if agent_name == "coco":
            agent.load_pretrain(probvlm_path="models/official_model/probvlm/COCO/probvlm_0.2_0.3_20-epoch-99.pth", clipcap_path="models/official_model/clipcap_coco_weights.pt", strict_clipcap=False)
        else:
            agent.load_pretrain(probvlm_path="models/official_model/probvlm/CC3M/probvlm_0.2_0.3_20-epoch-15.pth", clipcap_path="models/official_model/clipcap_conceptual_weights.pt", strict_clipcap=False)
    else:
        if agent_name == "coco":
            agent.lora_setting()
            agent.load_pretrain(probvlm_path=f"{exp_dir}/B/probvlm_B-epoch-9.pth", clipcap_path=f"{exp_dir}/B/clipcap_B_20-009.pt")
        else:
            agent.lora_setting()
            agent.load_pretrain(probvlm_path=f"{exp_dir}/A/probvlm_A-epoch-9.pth", clipcap_path=f"{exp_dir}/A/clipcap_A_20-009.pt")
    
    agent = agent.to(device)
    print("Agent initialized")

    df = pd.read_csv("generated_caption_test_coco_100_cc3m_100.csv")

    image_paths = df["Image Path"].tolist()
    print(image_paths[0])

    image = Image.open(image_paths[0])
    plt.imshow(image)
    plt.savefig("test_image.jpg")
    image = preprocess(image).unsqueeze(0).to(agent.device)

    mu_i, alpha_i, beta_i, z_i = agent.image_encoder(image)

    print(mu_i.shape, alpha_i.shape, beta_i.shape, z_i.shape)

    caption1 = df["Coco Generated Captions 0"][0]
    caption1 = df["Coco Generated Captions 1"][0]


