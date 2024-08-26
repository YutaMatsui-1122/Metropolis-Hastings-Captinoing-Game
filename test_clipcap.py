from one_agent import OneAgent
from utils import * 
import pickle, clip, argparse
from torch.nn import functional as nnf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from bert_score import score as bert_score
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType
# from torchmetrics.text import SacreBLEUScore

parser = argparse.ArgumentParser()
parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
parser.add_argument('--exp_name', default="debug")
parser.add_argument('--MH_iter', default=100, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--annealing', default="None")
parser.add_argument('--mode', default="MHNG")
args = parser.parse_args()

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
data_path = 'dataset/'
normalize_prefix = True

# coco_test_dataset_A = CocoDataset(root = data_path, transform=preprocess,data_mode="test", prefix_length=prefix_length, normalize_prefix=normalize_prefix, datasize="10000", use_imageid=True)
with open("dataset/dataset_cache/coco_test_dataset_5000_imageid.pkl", "rb") as f:
    coco_test_dataset_A = pickle.load(f)
    coco_test_dataset_A.prefix_length = 10

with open("dataset/dataset_cache/conceptual_test_dataset_11467.pkl", "rb") as f:
# with open("dataset/dataset_cache/conceptual_test_dataset_300.pkl", "rb") as f:
    conceptual_test_dataset = pickle.load(f)
    conceptual_test_dataset.prefix_length = 10


print("coco_test_dataset_A:", len(coco_test_dataset_A))
print("conceptual_test_dataset:", len(conceptual_test_dataset))
coco_test_loader = torch.utils.data.DataLoader(coco_test_dataset_A, batch_size=args.batch_size, shuffle=False, num_workers=1)
conceptual_test_loader = torch.utils.data.DataLoader(conceptual_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

root_paths = [
    # "models/official_model",
    # "models/lora_conceptual2coco_lr5e-6",
    # "exp/MH-ClipCap-only_1/A",
    # "exp/MH-ClipCap-only_1/B",
    # "exp/GEMCG_first_1/A",
    # "exp/GEMCG_first_1/B",
    "models/derpp_coco2conceptual_lr5e-6_test"
]

lora = True
# coco_original = True
# conceptua_original = False

for em_iter in [9, 5]:
    for gen_mode in ["sample"]:
        for root_path in root_paths:
            if "exp" in root_path:
                lora = True
                adapter = "mlp"
                i = 0
                
            elif "coco_cc3m" in root_path:
                lora = False
                test_loss = np.load(f"{root_path}/coco_cc3m_prefix_test_loss.npy")
                coco_test_loader.dataset.prefix_length = 40
                conceptual_test_loader.dataset.prefix_length = 40
                adapter = "transformer"
                # テストロスを小さい順に並べ替え，３つのモデルを選択
                best_epochs = np.argsort(test_loss)[:3]
                print(best_epochs)
                i = best_epochs[0]
            elif "official" in root_path:
                lora = False
                adapter = "mlp"
                best_epochs = [0]
                i = 0
            else:
                lora = True
                test_loss = np.load(f"{root_path}/finetune_test_loss.npy")
                print(test_loss)
                pretrain_test_loss = np.load(f"{root_path}/pretrain_test_loss.npy")
                print(pretrain_test_loss)
                adapter = "mlp"
                # テストロスを小さい順に並べ替え，３つのモデルを選択
                best_epochs = np.argsort(test_loss)[:10]
                print(best_epochs)
                i = best_epochs[0]
            for t in [0.7, 0.9]:
                print("exp:", i)
                agent = OneAgent(agent_name='A',adapter=adapter)
                agent = agent.to(device)
                if "exp" in root_path:
                    if "A" in root_path:
                        agent.lora_setting()
                        agent.load_pretrain(probvlm_path=f"models/probVLM_conceptual_prefix-030.pth", clipcap_path=f"{root_path}/clipcap_A_{em_iter}-009.pt", strict_clipcap=False)
                    elif "B" in root_path:
                        agent.lora_setting()
                        agent.load_pretrain(probvlm_path=f"models/probVLM_coco_prefix-030.pth", clipcap_path=f"{root_path}/clipcap_B_{em_iter}-009.pt", strict_clipcap=False)
                elif lora == True:
                    agent.load_pretrain(probvlm_path=f"models/probVLM_conceptual_prefix-030.pth", clipcap_path=f"models/official_model/clipcap_conceptual_weights.pt", strict_clipcap=False)
                    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=2, lora_alpha=32, lora_dropout=0.1, target_modules=["c_fc"])
                    agent.ClipCap.gpt = get_peft_model(agent.ClipCap.gpt, peft_config)
                    agent.ClipCap.load_state_dict(torch.load(f"{root_path}/coco_prefix-{i:03}.pt"))
                elif "coco_cc3m" in root_path:
                    print(f"{root_path}/coco_cc3m_prefix-{i:03}.pt")
                    agent.load_pretrain(probvlm_path=f"models/probVLM_coco_prefix-036.pth", clipcap_path=f"{root_path}/coco_cc3m_prefix-{i:03}.pt")
                
                elif "official" in root_path:
                    agent.load_pretrain(probvlm_path=f"models/probVLM_coco_prefix-030.pth", clipcap_path=f"{root_path}/clipcap_coco_weights.pt", strict_clipcap=False)
                
                # if coco_original:
                #     agent.load_pretrain(probvlm_path=f"models/probVLM_conceptual_prefix-030.pth", clipcap_path=f"models/official_model/clipcap_coco_weights.pt", strict_clipcap=False)
                #     print("Load original COCO model")
                # elif conceptua_original:
                #     agent.load_pretrain(probvlm_path=f"models/probVLM_conceptual_prefix-030.pth", clipcap_path=f"models/official_model/clipcap_conceptual_weights.pt", strict_clipcap=False)
                #     print("Load original Conceptual model")
                refs = []
                gens = []
                image_ids = []
                image_features = []
                text_features = []
                image_index = 0
                for test_loader, dataset_name in zip([coco_test_loader,conceptual_test_loader], ["coco","conceptual"]):
                    print("dataset name:", dataset_name)
                    for batch in tqdm.tqdm(test_loader):
                        if dataset_name == "coco":
                            img, caption, vlm_token, gpt_token, gpt_mask, index, image_id = batch
                            image_ids.extend([id.item() for id in image_id])
                        else:
                            img, caption, vlm_token, gpt_token, gpt_mask, index = batch
                            image_ids.extend(["conceptual"]*len(caption))
                        img, gpt_token, gpt_mask = img.to(device), gpt_token.to(device), gpt_mask.to(device)

                        prefix = clip_model.encode_image(img).to(device, dtype=torch.float32)
                        prefix_embeds = agent.ClipCap.clip_project(prefix).reshape(prefix.shape[0], test_loader.dataset.prefix_length, -1)
                        
                        image_features.append(prefix.cpu().detach())

                        
                        if gen_mode == "beam":
                            for idx, prefix_embed in enumerate(prefix_embeds):
                                prefix_embed = prefix_embed.unsqueeze(0)
                                generated_text = generate_beam(agent.ClipCap, agent.tokenizer, embed=prefix_embed, temperature=1)[0]
                                gens.append(generated_text)
                                text_tokens = tokenize(generated_text).to(device)
                                text_embed = clip_model.encode_text(text_tokens).to(device, dtype=torch.float32)
                                text_features.append(text_embed.cpu().detach())
                                print(f"Image {image_index}:", generated_text)
                                image_index += 1
                        if gen_mode == "sample":
                            generated_texts = generate_batch(agent.ClipCap, agent.tokenizer, embed=prefix_embeds, temperature=0.6)
                            gens.extend(generated_texts)
                            for text in generated_texts:
                                text_tokens = tokenize(text).to(device)
                                text_embed = clip_model.encode_text(text_tokens).to(device, dtype=torch.float32)
                                text_features.append(text_embed.cpu().detach())
                            for text in generated_texts:
                                print(f"Image {image_index}:", text)
                                image_index += 1
                        
                        refs.extend(list(caption))

                        # sacre_bleu = SacreBLEUScore(lowercase=True)
                        # coco_bleu_score = sacre_bleu(gens, refs).item() 
                        # print("sacre BLEU:", coco_bleu_score)


                bscore = bert_score(gens, refs, lang="en")
                print(f"BERT Score: {bscore[2].mean().item()}")

                image_features = torch.cat(image_features, dim=0)
                text_features = torch.cat(text_features, dim=0)
                clip_score = nnf.cosine_similarity(image_features, text_features).cpu().detach().numpy()
                print(f"CLIP Score: {np.mean(clip_score)}")

                df = pd.DataFrame({'Reference': refs, 'Generated': gens, 'ImageID': image_ids, 'BERT_Score': bscore[2].cpu().numpy(), 'CLIP_Score': clip_score})

                if "exp" in root_path:
                    df.to_csv(f"{root_path}/generated_sentences_t_{t}_{gen_mode}_em_iter_{em_iter}.csv")
                else:
                    df.to_csv(f"{root_path}/generated_sentences_t_{t}_{gen_mode}.csv")