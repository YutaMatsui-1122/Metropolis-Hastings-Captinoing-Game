from one_agent import OneAgent
from utils import * 
import pickle, argparse, copy
from ProbVLM.src.losses import *
import clip, os
from eval_probvlm_acceptance_prob import *

parser = argparse.ArgumentParser()
parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
parser.add_argument('--exp_name', default="debug")
parser.add_argument('--exp_num', default=1, type=int)
parser.add_argument('--num_workers', default=1, type=int)
parser.add_argument('--device', default="cuda:3")
parser.add_argument('--batch_size', default=64, type=int)

args = parser.parse_args()
model, preprocess = clip.load("ViT-B/32", device="cpu")

exp_name = args.exp_name

print(f"test on {exp_name}")
conceptual_pretrain_file = "conceptual_train_dataset_10000"
with open(f"dataset/dataset_cache/{conceptual_pretrain_file}.pkl", "rb") as f:
    conceptual_pretrain_dataset = pickle.load(f)
    conceptual_pretrain_dataset.prefix_length = 10

coco_pretrain_file = "coco_train_dataset_10000"
with open(f"dataset/dataset_cache/{coco_pretrain_file}.pkl", "rb") as f:
    coco_pretrain_dataset = pickle.load(f)
    coco_pretrain_dataset.prefix_length = 10

conceptual_pretrain_loader = torch.utils.data.DataLoader(conceptual_pretrain_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
coco_pretrain_loader = torch.utils.data.DataLoader(coco_pretrain_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


Cri = TempCombLoss()
T1=0
T2=1

# exp_nameに基づいて保存ディレクトリを作成
save_dir = f"models/{exp_name}"
os.makedirs(save_dir, exist_ok=True)
log_file = os.path.join(save_dir, "output_log.txt")

# ファイルに書き込むための関数
def log_to_file(message):
    with open(log_file, "a") as f:
        f.write(message + "\n")

for i in range(11):
    agentA = OneAgent(agent_name='A', device=args.device)
    agentA = agentA.to(args.device)
    # load pretrain probvlm from probvlm_derpp_test_1 directory
    agentA.load_pretrain(probvlm_path=f"models/{exp_name}/probvlm_A-epoch-{i}.pth", clipcap_path="models/official_model/clipcap_conceptual_weights.pt", strict_clipcap=False)
    
    # ファイルに書き込み
    log_to_file(f"probvlm_A-epoch-{i}.pth")
    
    agentA.eval()

    score = eval_probvlm_likelihood(agentA, preprocess)

    log_to_file(f"Spearman score {score}")

    # ファイルに書き込み
    log_to_file("test on conceptual dataset")
    
    test_loss = 0
    for idx, batch in tqdm(enumerate(conceptual_pretrain_loader), total=len(conceptual_pretrain_loader), desc="eval"):
        image = batch[0].to(args.device)
        vlm_token = batch[2].to(args.device)
        index = batch[5]
        z, _, _, _ = agentA.image_encoder(image)
        text_emb = agentA.CLIP_Net.encode_text(vlm_token).to(args.device, dtype=torch.float32)

        txt_mu, txt_alpha, txt_beta = agentA.ProbVLM_Net.txt_BayesCap(text_emb)

        loss = Cri(txt_mu, txt_alpha, txt_beta, z, T1=T1, T2=T2)
        test_loss += loss.item()

    test_loss /= len(conceptual_pretrain_loader)
    
    # ファイルに書き込み
    log_to_file(f"Test Loss on Conceptual Dataset: {test_loss}")

    # ファイルに書き込み
    log_to_file("test on coco dataset")
    
    agentA.eval()

    test_loss = 0
    for idx, batch in tqdm(enumerate(coco_pretrain_loader), total=len(coco_pretrain_loader), desc="eval"):
        image = batch[0].to(args.device)
        vlm_token = batch[2].to(args.device)
        index = batch[5]
        z, _, _, _ = agentA.image_encoder(image)
        text_emb = agentA.CLIP_Net.encode_text(vlm_token).to(args.device, dtype=torch.float32)

        txt_mu, txt_alpha, txt_beta = agentA.ProbVLM_Net.txt_BayesCap(text_emb)

        loss = Cri(txt_mu, txt_alpha, txt_beta, z, T1=T1, T2=T2)
        test_loss += loss.item()

    test_loss /= len(coco_pretrain_loader)
    
    # ファイルに書き込み
    log_to_file(f"Test Loss on COCO Dataset: {test_loss}")