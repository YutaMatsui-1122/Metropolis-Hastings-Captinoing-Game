import torch
from one_agent import OneAgent
import pickle, clip, argparse
from utils import *
import matplotlib.pyplot as plt
from ProbVLM.src.losses import *



argparser = argparse.ArgumentParser()
argparser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
argparser.add_argument('--dataset', default="coco", choices=('coco', 'conceptual'))
argparser.add_argument('--exp_name', default="debug")
args = argparser.parse_args()

clip_model, preprocess = clip.load(args.clip_model_type, device="cuda")

agent = OneAgent(agent_name='A')
agent = agent.cuda()
agent.ProbVLM_Net.load_state_dict(torch.load("models/ProbVLM_Net_last.pth"))
agent.ClipCap.load_state_dict(torch.load("models/conceptual_prefix-005.pt"))

if args.dataset == "coco":
    with open("dataset/dataset_cache/coco_test_dataset.pkl", "rb") as f:
        test_dataset = pickle.load(f)
elif args.dataset == "conceptual":
    with open("dataset/dataset_cache/conceptual_captions_subset_test_dataset.pkl", "rb") as f:
        test_dataset = pickle.load(f)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

test_batch = next(iter(test_dataloader))
caption = test_batch[1]
img = test_batch[0]
vlm_token = test_batch[2]

print("caption:", caption)
print("img.shape:", test_batch[0].shape)

agent.ProbVLM_Net.eval()
agent.ClipCap.eval()

with torch.no_grad():
    z = agent.CLIP_Net.encode_image(test_batch[0].cuda())
    mu_img, alpha_img, beta_img = agent.ProbVLM_Net.img_BayesCap(z)
    print(z[0][:10])
    print(mu_img[0][:10])
    t = agent.text_decoder(mu_img)
    for i in range(len(img)):
        print("caption:", t)
        plt.imshow(img[i].permute(1,2,0).cpu().numpy())
        plt.savefig("img_"+str(i)+".png")
        print()
    
token = tokenize(t)
print(token.dtype)

with torch.no_grad():
    mu_Li, alpha_Li, beta_Li = agent.text_encoder(token.cuda())
    mu_Sp, alpha_Sp, beta_Sp = agent.text_encoder(vlm_token.cuda())

ggl = GenGaussLoss()

cos_ann_list = []
for i in range(100):
    # beta_ratio = annealed_beta(i, epochs=100)
    beta_ratio = cosine_annealing_beta(i, epochs=100)
    beta_Sp_ = beta_Sp * beta_ratio
    beta_Li_ = beta_Li * beta_ratio
    p_Sp = -ggl(mu_Sp, alpha_Sp, beta_Sp_, z)
    p_Li = -ggl(mu_Li, alpha_Li, beta_Li, z)

    print(p_Sp, ((z-mu_Sp)**2).mean())
    print(p_Li, ((z-mu_Li)**2).mean())
    r = min(1, (p_Sp-p_Li).exp().item())
    cos_ann_list.append(r)
    print(beta_ratio, r)
    print()


exp_ann_list = []
for i in range(100):
    beta_ratio = exponential_annealing_beta(i, epochs=100)
    # beta_ratio = cosine_annealing_beta(i, epochs=100)
    beta_Sp_ = beta_Sp * beta_ratio
    beta_Li_ = beta_Li * beta_ratio
    p_Sp = -ggl(mu_Sp, alpha_Sp, beta_Sp_, z)
    p_Li = -ggl(mu_Li, alpha_Li, beta_Li_, z)

    print(p_Sp, ((z-mu_Sp)**2).mean())
    print(p_Li, ((z-mu_Li)**2).mean())
    r = min(1, (p_Sp-p_Li).exp().item())
    exp_ann_list.append(r)
    print(beta_ratio, r)
    print()

epochs = 100
epochs_range = np.arange(epochs)  
lr_exponential_increase = [exponential_annealing_beta(epoch) for epoch in epochs_range]
lr_cosine_increase = [cosine_annealing_beta(epoch, epochs=epochs) for epoch in epochs_range]
plt.figure(figsize=(10, 6))
plt.plot(range(100), cos_ann_list, label='Cosine Annealing AR', color='blue')
plt.plot(range(100), exp_ann_list, label='Exponential Annealing AR', color='orange')
plt.plot(epochs_range, lr_exponential_increase, label='Exponential Annealing beta', color='green')
plt.plot(epochs_range, lr_cosine_increase, label='Cosine Annealing beta', color='red')
plt.plot