import torch
from one_agent import OneAgent
import pickle, clip, argparse
from utils import *
from update_models import *
from transformers import AdamW, get_linear_schedule_with_warmup

argparser = argparse.ArgumentParser()
argparser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32', 'ViT-B/16', ))
argparser.add_argument('--dataset', default="COCO", choices=("COCO", "CC3M"))
argparser.add_argument('--epoch', default=20, type=int)
argparser.add_argument('--lr', default=2e-5, type=float)
argparser.add_argument('--num_workers', type=int, default=1)
argparser.add_argument('--batch_size', type=int, default=40)
argparser.add_argument('--save_dir', default="debug/")
argparser.add_argument('--save_interval', default=1, type=int)
argparser.add_argument('--device', default="cuda:0")

args = argparser.parse_args()
save_path = os.path.join("models", args.save_dir)
save_args_to_json(args, save_dir=save_path)

os.makedirs(f"models/{args.save_dir}", exist_ok=True)

device = torch.device(args.device)

# clip_model, preprocess = clip.load(args.clip_model_type, device=device)



if args.dataset == "COCO":
    with open("dataset/dataset_cache/coco_test.pkl", "rb") as f:
        train_dataset = pickle.load(f)
        train_dataset.prefix_length = 10

elif args.dataset == "CC3M":
    with open("dataset/dataset_cache/cc3m_test.pkl", "rb") as f:
        train_dataset = pickle.load(f)
        train_dataset.prefix_length = 10
else:
    raise ValueError("Invalid dataset")

# with open("dataset/dataset_cache/communication_coco_50_cc3m_50.pkl", "rb") as f:
#     train_dataset = pickle.load(f)
#     train_dataset.prefix_length = agent.prefix_length
    

print("prefix_length", train_dataset.prefix_length)

test_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=40, shuffle=True, num_workers=args.num_workers)



eval_loss_list = []
best_loss = 1e9

for epoch in range(args.epoch):
    agent = OneAgent(agent_name='A', device=device, temperature=0.62, clip_arch=args.clip_model_type)
    agent.load_pretrain(probvlm_path="models/official_model/probvlm/CC3M/probvlm_0.2_0.3_20-epoch-15.pth", clipcap_path=f"models/{args.save_dir}/clipcap_{epoch:03d}.pt", strict_clipcap=False)
    model = agent.ClipCap.to(device)
    clip_model = agent.CLIP_Net.to(device)
    model.eval()
    clip_model.eval()

    print(f">>> Evaluating epoch {epoch} <<<")
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        test_progress = tqdm(test_dataloader)
        for idx, batch in enumerate(test_dataloader):
            images = batch["image"].to(device)
            tokens = batch["gpt_token"].to(device)
            mask = batch["gpt_mask"].to(device)
            with torch.no_grad():
                prefix = clip_model.encode_image(images).to(device, dtype=torch.float32)
            outputs = model(tokens, prefix, mask)
            logits = outputs.logits[:, train_dataset.prefix_length - 1: -1]
            loss = nnf.cross_entropy(logits.reshape(-1, logits.size(-1)), tokens.flatten(), ignore_index=0)
            eval_loss += loss.item()
            test_progress.set_postfix(loss=loss.item())
            test_progress.update()

    eval_loss_list.append(eval_loss / len(test_dataloader))
    test_progress.close()
    
    torch.save(eval_loss_list, f"models/{args.save_dir}/eval_loss_list.pt")
    print(f"Epoch {epoch} eval loss: {eval_loss_list[-1]}")