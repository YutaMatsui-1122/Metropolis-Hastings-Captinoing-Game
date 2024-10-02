from sample_captions import *
from PIL import Image
import pandas as pd
import numpy as np
import torch
import clip
from one_agent import OneAgent
from utils import *
from scipy.stats import spearmanr

def eval_prbvlm(model, loader, device, Cri, T1, T2, cross_modal_lambda):
    model.eval()
    test_loss = 0
    test_loss_i = 0
    test_loss_t = 0
    test_loss_i2t = 0
    test_loss_t2i = 0

    for idx, batch in tqdm(enumerate(loader), total=len(loader), desc="eval"):
        image = batch[0].to(device)
        vlm_token = batch[2].to(device)
        index = batch[5]
        
        with torch.no_grad():
            xfI = model.CLIP_Net.encode_image(image)
            xfT = model.CLIP_Net.encode_text(vlm_token)
        
        (img_mu, img_alpha, img_beta), (txt_mu, txt_alpha, txt_beta) = model.ProbVLM_Net(xfI, xfT)

        loss_i = Cri(img_mu, img_alpha, img_beta, xfI, T1=T1, T2=T2)
        loss_t = Cri(txt_mu, txt_alpha, txt_beta, xfT, T1=T1, T2=T2)

        loss_i2t = Cri(img_mu, img_alpha, img_beta, xfT, T1=T1, T2=T2)
        loss_t2i = Cri(txt_mu, txt_alpha, txt_beta, xfI, T1=T1, T2=T2)

        loss = loss_i + loss_t + cross_modal_lambda * (loss_i2t + loss_t2i)

        test_loss += loss.item()
        test_loss_i += loss_i.item()
        test_loss_t += loss_t.item()
        test_loss_i2t += loss_i2t.item()
        test_loss_t2i += loss_t2i.item()

    test_loss /= len(loader)
    test_loss_i /= len(loader)
    test_loss_t /= len(loader)
    test_loss_i2t /= len(loader)
    test_loss_t2i /= len(loader)

    return test_loss, test_loss_i, test_loss_t, test_loss_i2t, test_loss_t2i

def clip_score(clip, image_features, caption_features):
    image_features /= np.linalg.norm(image_features, axis=1, keepdims=True)
    caption_features /= np.linalg.norm(caption_features, axis=1, keepdims=True)
    similarities = np.dot(image_features, caption_features.T)

    clip_score = 1.0 * np.clip(np.diag(similarities), 0, None)
    return clip_score




def eval_probvlm_acceptance_prob(agent, preprocess):
    # all_captions_images は、all_captions_image_0, all_captions_image_1, ... などの辞書をリストに格納
    all_captions_images = [all_captions_image_0, all_captions_image_1, all_captions_image_2, all_captions_image_4,
                           all_captions_image_5, all_captions_image_6, all_captions_image_7, all_captions_image_8, all_captions_image_9]

    likelihood_lists = [[] for _ in range(len(all_captions_images))]
    clip_score_lists = [[] for _ in range(len(all_captions_images))]
    spearman_scores = []  # Spearmanスコアを格納するリスト

    for idx, all_captions_image in enumerate(all_captions_images):
        image_path = all_captions_image["path"]
        image = Image.open(image_path)
        image = preprocess(image).unsqueeze(0).to(agent.device)
        mu_i, alpha_i, beta_i, z_i = agent.image_encoder(image)

        # キャプションの数に応じて動的にループを回す
        for i in range(2, 5):  # キャプションのインデックスに基づいて動的に対応
            caption = all_captions_image[i]
            token = tokenize(caption).to(agent.device)

            mu_t, alpha_t, beta_t, z_t = agent.text_encoder(token, return_z=True)

            mu_i_lent = mu_i.repeat(len(mu_t), 1)
            z_c = -agent.GGL(mu_t, alpha_t, beta_t, mu_i_lent)

            caption_features = z_t.cpu().detach().numpy()
            image_features = z_i.cpu().detach().numpy().repeat(len(caption_features), 0)
            score = clip_score(agent.CLIP_Net, image_features, caption_features)

            likelihood_lists[idx].extend(z_c.cpu().detach().numpy())
            clip_score_lists[idx].extend(score)

    # calc spearman correlation  
    for idx in range(len(all_captions_images)):
        # Spearmanの相関係数を計算し、スコアのみを抽出
        score, _ = spearmanr(clip_score_lists[idx], likelihood_lists[idx])
        spearman_scores.append(score)
    
    # スコアの平均を計算して出力
    avg_spearman_score = sum(spearman_scores) / len(spearman_scores)
    print(f"Average Spearman score: {avg_spearman_score:.4f}")

def eval_probvlm_likelihood(agent, preprocess, batch_size=100):
    # read dataframes
    df = pd.read_csv("generated_captions.csv")

    image_paths = df["Image Path"].tolist()

    image_list = []
    for path in image_paths:
        image = Image.open(path)
        image = preprocess(image).unsqueeze(0).to(agent.device)
        image_list.append(image)

    # Create empty lists to hold the final outputs
    mu_list, alpha_list, beta_list, z_list = [], [], [], []

    # Process images in batches to avoid running out of memory
    for i in range(0, len(image_list), batch_size):
        # Get a batch of images
        batch_images = torch.cat(image_list[i:i+batch_size], dim=0)
        
        # Pass the batch through the image encoder
        mu_i, alpha_i, beta_i, z_i = agent.image_encoder(batch_images)
        
        # Append the results for each batch
        mu_list.append(mu_i)
        alpha_list.append(alpha_i)
        beta_list.append(beta_i)
        z_list.append(z_i)

    # Concatenate all the results from the batches
    mu = torch.cat(mu_list, dim=0)
    alpha = torch.cat(alpha_list, dim=0)
    beta = torch.cat(beta_list, dim=0)
    z = torch.cat(z_list, dim=0)
    
    # Loop through coco and conceptual datasets
    for data_name in ["coco", "conceptual"]:
        likelihood_lists = []
        scores = []
        for i in range(10):
            # caption = df[f'coco Generated Caption {i}'].tolist()
            # score = df[f'coco Score {i}'].tolist()
            caption = df[f'{data_name} Generated Caption {i}'].tolist()
            score = df[f'{data_name} Score {i}'].tolist()

            # Create empty lists to store intermediate results for each batch
            likelihood_batch_list = []
            
            # Process tokens in batches
            for j in range(0, len(caption), batch_size):
                # Get a batch of tokens
                token_batch = tokenize(caption[j:j+batch_size]).to(agent.device)

                # Pass the batch through the text encoder
                mu_t, alpha_t, beta_t, z_t = agent.text_encoder(token_batch, return_z=True)
                
                # Compute GGL for the batch
                z_c_batch = -agent.GGL(mu_t, alpha_t, beta_t, mu[j:j+batch_size]).cpu().detach().numpy()

                # Append batch results
                likelihood_batch_list.extend(z_c_batch)

            # Extend results for each caption batch
            likelihood_lists.extend(likelihood_batch_list)
            scores.extend(score)
    
        # Calculate spearman correlation
        score, _ = spearmanr(scores, likelihood_lists)
        print(f"{data_name} Spearman score: {score:.4f}")

    return score

if __name__ == "__main__":

    
    device = "cuda:3" if torch.cuda.is_available() else "cpu"

    previous = False

    if previous:
        save_dir = f"models"
    else:
        # save_dir = f"models/official_model/probvlm/CC3M"
        save_dir = f"models/objective_test_1"

    print("Device:", device)

    model, preprocess = clip.load("ViT-B/32", device="cpu")

    # for i in [1, 2, 3, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91, 94, 97, 100]:
    # for i in [0, 1, 2, 3, 4, 6, 9, 12, 15, 21, 30, 39, 48, 60, 81, 99]:
    for i in [0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84, 89, 94, 99]:
        print(i)
        agent = OneAgent(agent_name='A', device=device)
        if previous:
            agent.load_pretrain(probvlm_path=f"{save_dir}/probVLM_conceptual_prefix-035.pth", clipcap_path="models/official_model/clipcap_conceptual_weights.pt", strict_clipcap=False)
            
        else:
            # agent.load_pretrain(probvlm_path=f"{save_dir}/probvlm_0.2_0.3_20-epoch-{i}.pth", clipcap_path="models/official_model/clipcap_conceptual_weights.pt", strict_clipcap=False)
            agent.load_pretrain(probvlm_path=f"{save_dir}/probvlm_A-epoch-{i}.pth", clipcap_path="models/official_model/clipcap_conceptual_weights.pt", strict_clipcap=False)
        agent = agent.to(device)

        eval_probvlm_likelihood(agent, preprocess)

        if previous:
            break