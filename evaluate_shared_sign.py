import torch
from utils import * 
import pandas as pd
from evaluation_metrics import *

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

exp_name = "gemcg_unified_dataset_coco5000_cc3m5000_agentA05_agentBbeta005_1"
exp_dir = f"exp/{exp_name}"
observation_file = "communication_coco_5000_cc3m_5000"
eval_sign = False
eval_captioning = True
em_iter = 10

bleu_list = []
meteor_list = []
bert_list = []


agentA_initial_sign = pd.read_csv(f"{exp_dir}/A/sign_initial.csv")["captions"].tolist()
agentB_initial_sign = pd.read_csv(f"{exp_dir}/B/sign_initial.csv")["captions"].tolist()

bleu = 0
meteor = 0
bert_s = 0

ref_agentA_initial_sign = [[captions] for captions in agentA_initial_sign]
ref_agentB_initial_sign = [[captions] for captions in agentB_initial_sign]

bleu += calculate_bleu_score(agentA_initial_sign, ref_agentB_initial_sign)
meteor += calculate_meteor_score(agentA_initial_sign, ref_agentB_initial_sign)
bert_s += calculate_bert_score(agentA_initial_sign, agentB_initial_sign)

print(agentA_initial_sign[:10])
print(agentB_initial_sign[:10])

bleu += calculate_bleu_score(agentB_initial_sign, ref_agentA_initial_sign)
meteor += calculate_meteor_score(agentB_initial_sign, ref_agentA_initial_sign)
bert_s += calculate_bert_score(agentB_initial_sign, agentA_initial_sign)

bleu /= 2
meteor /= 2
bert_s /= 2

print("initial sign")
print("BLEU:", bleu)
print("METEOR:", meteor)
print("BERT_score:", bert_s)

bleu_list.append(bleu)
meteor_list.append(meteor)
bert_list.append(bert_s)

# afterCG sign

for i in range(em_iter):
    agentA_afterCG_sign = pd.read_csv(f"{exp_dir}/A/sign_EM_{i}.csv")["captions"].tolist()
    agentB_afterCG_sign = pd.read_csv(f"{exp_dir}/B/sign_EM_{i}.csv")["captions"].tolist()

    bleu = 0
    meteor = 0
    bert_s = 0

    ref_agentA_afterCG_sign = [[captions] for captions in agentA_afterCG_sign]
    ref_agentB_afterCG_sign = [[captions] for captions in agentB_afterCG_sign]

    bleu += calculate_bleu_score(agentA_afterCG_sign, ref_agentB_afterCG_sign)
    meteor += calculate_meteor_score(agentA_afterCG_sign, ref_agentB_afterCG_sign)
    bert_s += calculate_bert_score(agentA_afterCG_sign, agentB_afterCG_sign)

    bleu += calculate_bleu_score(agentB_afterCG_sign, ref_agentA_afterCG_sign)
    meteor += calculate_meteor_score(agentB_afterCG_sign, ref_agentA_afterCG_sign)
    bert_s += calculate_bert_score(agentB_afterCG_sign, agentA_afterCG_sign)

    bleu /= 2
    meteor /= 2 
    bert_s /= 2

    print(f"afterCG sign {i}")
    print("BLEU:", bleu)
    print("METEOR:", meteor)
    print("BERT_score:", bert_s)
    bleu_list.append(bleu)
    meteor_list.append(meteor)
    bert_list.append(bert_s)


# save scores with pandas dataframe and plot them
df = pd.DataFrame({"BLEU": bleu_list, "METEOR": meteor_list, "BERT": bert_list})
df.to_csv(f"{exp_dir}/sign_evaluation.csv")

import matplotlib.pyplot as plt

plt.figure()
plt.plot(bleu_list, label="BLEU")
plt.savefig(f"{exp_dir}/BLEU.png")

plt.clf()

plt.figure()
plt.plot(meteor_list, label="METEOR")
plt.savefig(f"{exp_dir}/METEOR.png")

plt.clf()

plt.figure()
plt.plot(bert_list, label="BERT")
plt.savefig(f"{exp_dir}/BERT.png")






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

