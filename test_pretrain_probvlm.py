import torch
from one_agent import OneAgent
import pickle, clip, argparse
from utils import *
from update_models import *
from ProbVLM.src.losses import *

argparser = argparse.ArgumentParser()
argparser.add_argument('--clip_model_type', default="ViT-B/32", choices=('ViT-B/32', 'ViT-B/16', ))
argparser.add_argument('--batch_size', default=64, type=int)
argparser.add_argument('--num_workers', default=1, type=int)
argparser.add_argument('--device', default="cuda:3")
args = argparser.parse_args()

clip_model, preprocess = clip.load(args.clip_model_type, device="cuda")

conceptual_pretrain_file = "cc3m_test"
with open(f"dataset/dataset_cache/{conceptual_pretrain_file}.pkl", "rb") as f:
    conceptual_pretrain_dataset = pickle.load(f)
    conceptual_pretrain_dataset.prefix_length = 10

coco_train_file = "coco_test_dataset"
with open(f"dataset/dataset_cache/{coco_train_file}.pkl", "rb") as f:
    train_dataset = pickle.load(f)
    train_dataset.prefix_length = 10

# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
cc3m_loader = torch.utils.data.DataLoader(conceptual_pretrain_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
coco_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

print("cc3m_dataset:", len(conceptual_pretrain_dataset))
print("coco_dataset:", len(train_dataset))

agent_dataset = "CC3M"

T1=0
T2=1
cross_modal_lambda=1

save_dir = f"models/official_model/probvlm/{agent_dataset}"


device = torch.device(args.device)

Cri = TempCombLoss(reduction="sum")

# for i in [1, 2, 3, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91, 94, 97, 100]:
for i in [1, 3, 6, 9, 15, 30, 45, 48, 51]:
    agent = OneAgent(agent_name='A', device=device, clip_arch=args.clip_model_type)
    file_base = "probvlm_0.2_0.2_20_arch_ViT-B-16-epoch"
    # agent.load_pretrain(probvlm_path=f"{save_dir}/probvlm_0.2_0.3_20-epoch-{i}.pth", clipcap_path="models/official_model/clipcap_conceptual_weights.pt", strict_clipcap=False)
    agent.load_pretrain(probvlm_path=f"{save_dir}/{file_base}-{i}.pth", clipcap_path="models/official_model/clipcap_conceptual_weights.pt", strict_clipcap=False)
    print(f"ProbVLM model : {file_base}-{i}.pth")
    agent = agent.to(args.device)
    for loader_name, loader in [("cc3m", cc3m_loader),]:
        test_loss = 0
        test_loss_i = 0
        test_loss_t = 0
        test_loss_i2t = 0
        test_loss_t2i = 0
        for idx, batch in tqdm(enumerate(loader), total=len(loader), desc="eval"):
            # image = batch[0].to(args.device)
            # vlm_token = batch[2].to(args.device)
            
            image = batch["image"].to(args.device)
            vlm_token = batch["vlm_token"].to(args.device)

            with torch.no_grad():
                xfI = agent.CLIP_Net.encode_image(image)
                xfT = agent.CLIP_Net.encode_text(vlm_token)
            
            (img_mu, img_alpha, img_beta), (txt_mu, txt_alpha, txt_beta) = agent.ProbVLM_Net(xfI, xfT)

            # print(xfI[0][:10], xfT[0][:10], img_mu[0][:10], txt_mu[0][:10])
            # print(xfI[0][:8])
            # print(img_mu[0][:8])
            # print(xfT[0][:8])
            # print(txt_mu[0][:8])

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

        test_loss /= len(loader.dataset)
        test_loss_i /= len(loader.dataset)
        test_loss_t /= len(loader.dataset)
        test_loss_i2t /= len(loader.dataset)
        test_loss_t2i /= len(loader.dataset)

        # test_loss /= len(loader)
        # test_loss_i /= len(loader)
        # test_loss_t /= len(loader)
        # test_loss_i2t /= len(loader)
        # test_loss_t2i /= len(loader)

        print(f"Dataloader: {loader_name}, Epoch: {i}, Test Loss: {test_loss}, Test Loss_i: {test_loss_i}, Test Loss_t: {test_loss_t}, Test Loss_i2t: {test_loss_i2t}, Test Loss_t2i: {test_loss_t2i}")

        # Save the test loss
        with open(f"{save_dir}/test_loss.txt", "a") as f:
            f.write(f"{i},{loader_name},{test_loss},{test_loss_i},{test_loss_t},{test_loss_i2t},{test_loss_t2i}\n")


