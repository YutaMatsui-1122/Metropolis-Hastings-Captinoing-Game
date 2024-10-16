from sample_captions import *
from PIL import Image
import pandas as pd
import torch
import clip
from one_agent import OneAgent
from utils import *
from matplotlib import pyplot as plt

if __name__ == "__main__":

    exp_dir = "exp/mhcg_derpp_0.05_1"

    device = "cuda:3" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device="cpu")

    pretrain = False
    agent_name = "cc3m"

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

    caption_sps = ["a man riding on the back of a motorcycle.", "a man riding on the back of a motorcycle on a dirt road.", "a woman is putting a spoon into a cake."]
    caption_li = "a man on a motorcycle in the mountains."
    anneal_values = [1000, 100, 10, 1, 0.5, 0.1, 0.01, 0.001, 0.0001, 1e-5, 1e-6, 1e-7, 1e-10, 1e-20]
    for anneal_value in anneal_values:
        for caption_sp in caption_sps:

            print(caption_sp, caption_li)

            token_sp = tokenize(caption_sp).to(agent.device)
            token_li = tokenize(caption_li).to(agent.device)

            mu_t__sp, alpha_t__sp, beta_t__sp, z_t__sp = agent.text_encoder(token_sp, return_z=True)
            mu_t__li, alpha_t__li, beta_t__li, z_t__li = agent.text_encoder(token_li, return_z=True)

            beta_t__li = beta_t__li * anneal_value
            beta_t__sp = beta_t__sp * anneal_value

            p_sp = -agent.MGGL(mu_t__sp, alpha_t__sp, beta_t__sp, mu_i).cpu().detach().numpy()
            p_li = -agent.MGGL(mu_t__li, alpha_t__li, beta_t__li, mu_i).cpu().detach().numpy()

            print(p_sp, p_li)
            r = np.exp(np.where((p_sp-p_li)<0,(p_sp-p_li),0))
            print(anneal_value, r, "\n")



