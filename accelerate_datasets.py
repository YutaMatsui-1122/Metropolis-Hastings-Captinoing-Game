from one_agent import OneAgent
from utils import * 
import pickle, argparse, copy

parser = argparse.ArgumentParser()
parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
parser.add_argument('--exp_name', default="debug")
parser.add_argument('--exp_num', default=1, type=int)
parser.add_argument('--MH_iter', default=10, type=int)
parser.add_argument('--EM_iter', default=10, type=int)
parser.add_argument('--Whole_iter', default=10, type=int)
parser.add_argument('--td_update_epochs', default=10, type=int)
parser.add_argument('--te_update_epochs', default=3, type=int)
parser.add_argument('--buffer_size', default=10000, type=int)
parser.add_argument('--num_workers', default=1, type=int)
parser.add_argument('--device', default="cuda:0")
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--temperature', default=0.62, type=float)
parser.add_argument('--agentA_alpha', default=0.5, type=float)
parser.add_argument('--agentA_beta', default=0.5, type=float)
parser.add_argument('--agentB_alpha', default=0.5, type=float)
parser.add_argument('--agentB_beta', default=0.5, type=float)   
parser.add_argument('--annealing', default="None")
parser.add_argument('--mode', default="MHNG")
args = parser.parse_args()


observation_file = f"communication_coco_5000_cc3m_5000"

with open(f"dataset/dataset_cache/{observation_file}.pkl", "rb") as f:
    observationA_dataset = pickle.load(f)
    observationA_dataset.prefix_length = 10

conceptual_pretrain_file = "conceptual_train_dataset_10000"
with open(f"dataset/dataset_cache/{conceptual_pretrain_file}.pkl", "rb") as f:
    conceptual_pretrain_dataset = pickle.load(f)
    conceptual_pretrain_dataset.prefix_length = 10

coco_pretrain_file = "coco_train_dataset_10000"
with open(f"dataset/dataset_cache/{coco_pretrain_file}.pkl", "rb") as f:
    coco_pretrain_dataset = pickle.load(f)
    coco_pretrain_dataset.prefix_length = 10

print("observationA_dataset:", len(observationA_dataset))
print("coco_pretrain_dataset:", len(coco_pretrain_dataset))
print("conceptual_pretrain_dataset:", len(conceptual_pretrain_dataset))

conceptual_pretrain_loader = torch.utils.data.DataLoader(conceptual_pretrain_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
coco_pretrain_loader = torch.utils.data.DataLoader(coco_pretrain_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

coco_test_loader_fix_A = torch.utils.data.DataLoader(observationA_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
coco_test_loader_shuffle_A = torch.utils.data.DataLoader(observationA_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

# observation_dataset と conceptual_pretrain_datasetのバッチを取得するのにかかる時間を計測

import time, tqdm

start = time.time()
for data in tqdm.tqdm(coco_pretrain_loader):
    pass
end = time.time()

print("coco_pretrain_loader:", end-start)

start = time.time()
for data in tqdm.tqdm(conceptual_pretrain_loader):
    pass
end = time.time()

print("conceptual_pretrain_loader:", end-start)

start = time.time()
for data in tqdm.tqdm(coco_test_loader_fix_A):
    pass
end = time.time()
print("coco_test_loader_fix_A:", end-start)

start = time.time()
for data in tqdm.tqdm(coco_test_loader_shuffle_A):
    pass
end = time.time()

print("coco_test_loader_shuffle_A:", end-start)