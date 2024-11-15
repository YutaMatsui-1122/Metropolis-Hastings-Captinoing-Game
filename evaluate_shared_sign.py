import torch
from utils import * 
import pandas as pd
from evaluation_metrics import *

def calc_scores(captions_A, captions_B):
    bleu = 0
    meteor = 0
    bert_s = 0

    ref_captions_A = [[captions] for captions in captions_A]
    ref_captions_B = [[captions] for captions in captions_B]

    bleu += calculate_bleu_score(captions_A, ref_captions_B)
    meteor += calculate_meteor_score(captions_A, ref_captions_B)
    bert_s += calculate_bert_score(captions_A, ref_captions_B)

    bleu += calculate_bleu_score(captions_B, ref_captions_A)
    meteor += calculate_meteor_score(captions_B, ref_captions_A)
    bert_s += calculate_bert_score(captions_B, ref_captions_A)

    bleu /= 2
    meteor /= 2 
    bert_s /= 2

    print("BLEU:", bleu)
    print("METEOR:", meteor)
    print("BERT_score:", bert_s)

    return bleu, meteor, bert_s

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

exp_name = "pretrain"
dataset_name = "cc3m"

for i in ["train"]:
    print(f"Epoch {i}")
    # file_name1 = f"{dataset_name}_candidate_A_epoch_29_temperature_0.7_{i}.json"
    # file_name2 = f"{dataset_name}_candidate_B_epoch_29_temperature_0.7_{i}.json"
    file_name1 = "coco_candidate_cc3m_temperature_0.7_vit16_train.json"
    file_name2 = "coco_candidate_coco_temperature_0.7_vit32_train.json"
    # coco_candidate_cc3m_temperature_0.7_vit16_train.json
    # coco_candidate_coco_temperature_0.7_vit32_train.json

    path1 = f"exp_eval/{exp_name}/{file_name1}"
    path2 = f"exp_eval/{exp_name}/{file_name2}"

    if not os.path.exists(path1) or not os.path.exists(path2):
        print("File not found")

    with open(path1, "r") as f:
        data1 = json.load(f)

    with open(path2, "r") as f:
        data2 = json.load(f)

    captions1 = list(data1.values())
    captions2 = list(data2.values())

    print("Number of captions:", len(captions1))
    print("Number of captions:", len(captions2))
    print("Sample captions:", captions1[0], captions2[0])

    if len(captions1) != len(captions2):
        print("Number of captions are not equal")

    # Calculate BLEU, METEOR, BERT_score

    calc_scores(captions1, captions2)




exit()

exp_dir = f"exp/{exp_name}"
em_iter = 30
mh_iter = 5
caption_type = "sign" # "sign" or "proposed_w"

bleu_list = []
meteor_list = []
bert_list = []

agentA_initial_sign = pd.read_csv(f"{exp_dir}/A/agent_A_{caption_type}.csv")["initial"].tolist()
agentB_initial_sign = pd.read_csv(f"{exp_dir}/B/agent_B_{caption_type}.csv")["initial"].tolist()

bleu, meteor, bert_s = calc_scores(agentA_initial_sign, agentB_initial_sign)

bleu_list.append(bleu)
meteor_list.append(meteor)
bert_list.append(bert_s)

# afterCG sign

for i in range(em_iter):
    if f"EM_{i}_MH_{mh_iter-1}" not in pd.read_csv(f"{exp_dir}/A/agent_A_{caption_type}.csv").columns:
        break

    agentA_afterCG_sign = pd.read_csv(f"{exp_dir}/A/agent_A_{caption_type}.csv")[f"EM_{i}_MH_{mh_iter-1}"].tolist()
    agentB_afterCG_sign = pd.read_csv(f"{exp_dir}/B/agent_B_{caption_type}.csv")[f"EM_{i}_MH_{mh_iter-1}"].tolist()

    bleu, meteor, bert_s = calc_scores(agentA_afterCG_sign, agentB_afterCG_sign)

    bleu_list.append(bleu)
    meteor_list.append(meteor)
    bert_list.append(bert_s)


# save scores with pandas dataframe and plot them
df = pd.DataFrame({"BLEU": bleu_list, "METEOR": meteor_list, "BERT": bert_list})
df.to_csv(f"{exp_dir}/{caption_type}_evaluation.csv")

import matplotlib.pyplot as plt

plt.figure()
plt.plot(bleu_list, label="BLEU")
plt.savefig(f"{exp_dir}/BLEU_{caption_type}.png")

plt.clf()

plt.figure()
plt.plot(meteor_list, label="METEOR")
plt.savefig(f"{exp_dir}/METEOR_{caption_type}.png")

plt.clf()

plt.figure()
plt.plot(bert_list, label="BERT")
plt.savefig(f"{exp_dir}/BERT_{caption_type}.png")

# agentA_afterCG_sign = pd.read_csv(f"{exp_dir}/A/sign_EM_9.csv")["captions"].tolist()
# agentB_afterCG_sign = pd.read_csv(f"{exp_dir}/B/sign_EM_9.csv")["captions"].tolist()



# # calc BLEU, METEOR, BERT_score
# # afterCG sign
# bleu = 0
# meteor = 0
# bert_s = 0

# ref_agentA_afterCG_sign = [[captions] for captions in agentA_afterCG_sign]
# ref_agentB_afterCG_sign = [[captions] for captions in agentB_afterCG_sign]

# bleu += calculate_bleu_score(agentA_afterCG_sign, ref_agentB_afterCG_sign)
# meteor += calculate_meteor_score(agentA_afterCG_sign, ref_agentB_afterCG_sign)
# bert_s += calculate_bert_score(agentA_afterCG_sign, agentB_afterCG_sign)

# bleu += calculate_bleu_score(agentB_afterCG_sign, ref_agentA_afterCG_sign)
# meteor += calculate_meteor_score(agentB_afterCG_sign, ref_agentA_afterCG_sign)
# bert_s += calculate_bert_score(agentB_afterCG_sign, agentA_afterCG_sign)

# bleu /= 2
# meteor /= 2 
# bert_s /= 2

# print("afterCG sign")
# print("BLEU:", bleu)
# print("METEOR:", meteor)
# print("BERT_score:", bert_s)

# # initial sign
# bleu = calculate_bleu_score(agentA_initial_sign, agentB_initial_sign)
# meteor = calculate_meteor_score(agentA_initial_sign, agentB_initial_sign)
# bert_s = calculate_bert_score(agentA_initial_sign, agentB_initial_sign)

# print("initial sign")
# print("BLEU:", bleu)
# print("METEOR:", meteor)
# print("BERT_score:", bert_s)

