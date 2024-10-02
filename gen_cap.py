from one_agent import OneAgent
import torch
import pickle
from utils import *

def clip_score(image_features, caption_features):
    image_features /= np.linalg.norm(image_features, axis=1, keepdims=True)
    caption_features /= np.linalg.norm(caption_features, axis=1, keepdims=True)
    similarities = np.dot(image_features, caption_features.T)

    clip_score = 1.0 * np.clip(np.diag(similarities), 0, None)
    return clip_score

device = torch.device("cuda:0")

observation_file = "test_coco_1000_cc3m_1000"
with open(f"dataset/dataset_cache/{observation_file}.pkl", "rb") as f:
    dataset = pickle.load(f)
    dataset.prefix_length = 10

dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=1)
image_paths = [entry["image"] for entry in dataset.dataset]
caption = [entry["caption"] for entry in dataset.dataset]
df = pd.DataFrame(image_paths, columns=['Image Path'])
df['Caption'] = caption

for data_name in ["coco", "conceptual"]:
    agent = OneAgent(agent_name = data_name, device=device, temperature = 0.7)
    agent = agent.to(device)
    agent.load_pretrain(probvlm_path=f"models/probVLM_coco_prefix-035.pth", clipcap_path=f"models/official_model/clipcap_{data_name}_weights.pt", strict_clipcap=False)
    # for roop with tqdm
    for i in tqdm(range(10)):
        gen_cap_list = []
        score_list = []
        for idx, batch in enumerate(dataloader):
            image = batch["image"].to(device)
            mu_i, alpha_i, beta_i, z_i = agent.image_encoder(image)
            gencap = agent.text_decoder(z_i)
            gen_cap_list.extend(gencap)
            token = tokenize(gencap).to(agent.device)

            mu_t, alpha_t, beta_t, z_t = agent.text_encoder(token, return_z=True)
            score = clip_score(z_i.cpu().detach().numpy(), z_t.cpu().detach().numpy())
            score_list.extend(score)

        df[f'{data_name} Generated Caption {i}'] = gen_cap_list
        df[f'{data_name} Score {i}'] = score_list

df.to_csv(f"generated_caption_{observation_file}.csv", index=False)