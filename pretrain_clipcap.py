import torch
from one_agent import OneAgent
import pickle, clip, argparse
from utils import *

argparser = argparse.ArgumentParser()
argparser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
argparser.add_argument('--dataset', default="coco", choices=('coco', 'conceptual', 'fine_tune', 'cc3m', 'fine_tune', '10000_pretrain', 'coco_cc3m'))
argparser.add_argument('--exp_name', default="debug")
argparser.add_argument('--initial_epoch', type=int, default=0)
argparser.add_argument('--num_workers', type=int, default=1)
argparser.add_argument('--batch_size', type=int, default=32)
argparser.add_argument('--save_dir', default="debug/")
argparser.add_argument('--device', default="cuda:0")

args = argparser.parse_args()

os.makedirs(f"models/{args.save_dir}", exist_ok=True)

device = torch.device(args.device)

clip_model, preprocess = clip.load(args.clip_model_type, device=device)

agent = OneAgent(agent_name='A')
agent = agent.cuda()
if args.initial_epoch > 0:
    agent.ClipCap.load_state_dict(torch.load(f"models/{args.dataset}_prefix-{args.initial_epoch:03d}.pt"))

if args.dataset == "coco":
    with open("dataset/dataset_cache/coco_train_dataset.pkl", "rb") as f:
        train_dataset = pickle.load(f)
    with open("dataset/dataset_cache/coco_test_dataset.pkl", "rb") as f:
        test_dataset = pickle.load(f)
elif args.dataset == "conceptual_subset":
    with open("dataset/dataset_cache/conceptual_captions_subset_train_dataset.pkl", "rb") as f:
        train_dataset = pickle.load(f)
    with open("dataset/dataset_cache/conceptual_captions_subset_test_dataset.pkl", "rb") as f:
        test_dataset = pickle.load(f)
elif args.dataset == "cc3m":
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    with open("dataset/dataset_cache/cc3m_train.pkl", "rb") as f:
        train_dataset = pickle.load(f)
    with open("dataset/dataset_cache/cc3m_valid.pkl", "rb") as f:
        test_dataset = pickle.load(f)

elif args.dataset == "fine_tune":
    print("fine_tune")
    with open("dataset/dataset_cache/coco_train_dataset_10000.pkl", "rb") as f:
        train_dataset = pickle.load(f)
    with open("dataset/dataset_cache/coco_test_dataset.pkl", "rb") as f:
        test_dataset = pickle.load(f)
    agent.load_pretrain(probvlm_path="models/probVLM_conceptual_prefix-040.pth", clipcap_path="models/official_model/clipcap_conceptual_weights.pt", strict_clipcap=False)

elif args.dataset == "10000_pretrain":
    print("10000_pretrain")
    with open("dataset/dataset_cache/coco_train_dataset_10000.pkl", "rb") as f:
        train_dataset = pickle.load(f)
    with open("dataset/dataset_cache/coco_test_dataset.pkl", "rb") as f:
        test_dataset = pickle.load(f)

elif args.dataset == "coco_cc3m":
    
    with open("dataset/dataset_cache/coco_cc3m_train.pkl", "rb") as f:
        train_dataset = pickle.load(f)
    with open("dataset/dataset_cache/coco_test_dataset.pkl", "rb") as f:
        test_dataset = pickle.load(f)
    print(f"coco_cc3m train: {len(train_dataset)}, test: {len(test_dataset)}")
    

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=40, shuffle=True, num_workers=args.num_workers)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=40, shuffle=False, num_workers=args.num_workers)

agent.pretrain_clipcap(train_dataloader, test_dataloader, output_prefix=args.dataset+"_prefix", lr = 2e-5, save_dir=f"models/{args.save_dir}/", epochs=30, device = device, ff_mode="original")