import torch
from one_agent import OneAgent
import pickle, clip, argparse
from utils import *
from update_models import *
from transformers import AdamW, get_linear_schedule_with_warmup

argparser = argparse.ArgumentParser()
argparser.add_argument('--clip_model_type', default="ViT-B/32", choices=('ViT-B/32', 'ViT-B/16', ))
argparser.add_argument('--dataset', default="COCO_A", choices=("COCO_A", "COCO_B"))   
argparser.add_argument('--epoch', default=20, type=int)
argparser.add_argument('--lr', default=2e-5, type=float)
argparser.add_argument('--num_workers', type=int, default=1)
argparser.add_argument('--batch_size', type=int, default=40)
argparser.add_argument('--save_interval', default=1, type=int)
argparser.add_argument('--device', default="cuda:0")
argparser.add_argument('--prefix', default="coco_2017_common_supcat1st")

set_random_seed(42)

args = argparser.parse_args()

save_path = os.path.join("pretrain_models", args.prefix, args.dataset, "clipcap")
save_args_to_json(args, save_dir=save_path)

os.makedirs(save_path, exist_ok=True)

device = torch.device(args.device)


print("clip_model_type:", args.clip_model_type)
agent = OneAgent(agent_name='A', device=device, clip_arch=args.clip_model_type, pretrain=True)
model = agent.ClipCap.to(device)
clip_model = agent.CLIP_Net.to(device)

optimizer = AdamW(model.parameters(), lr=args.lr)

# print total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')

if args.dataset == "COCO_A":
    with open(f"dataset/dataset_cache/{args.prefix}/train_split_dataset_a.pkl", "rb") as f:
        train_dataset = pickle.load(f)
        train_dataset.prefix_length = agent.prefix_length

elif args.dataset == "COCO_B":
    # with open("dataset/dataset_cache/coco_split_dataset_b.pkl", "rb") as f:
    with open(f"dataset/dataset_cache/{args.prefix}/train_split_dataset_b.pkl", "rb") as f:
        train_dataset = pickle.load(f)
        train_dataset.prefix_length = agent.prefix_length

else:
    raise ValueError("Invalid dataset")

print("train_dataset:", len(train_dataset))
    

print("prefix_length", train_dataset.prefix_length)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=5000, num_training_steps=args.epoch * len(train_dataloader))

text = generate_test(model, clip_model, train_dataloader, agent.tokenizer, sample_num=10, device=device, temperature=0.63)
print(text)

clip_model.eval()
train_loss_list = []
eval_loss_list = []
best_loss = 1e9

for epoch in range(args.epoch):
    print(f">>> Training epoch {epoch} <<<")
    model.train()
    train_loss = 0
    progress = tqdm(train_dataloader)
    for idx, batch in enumerate(train_dataloader):
        model.zero_grad()
        images = batch["image"].to(device)
        tokens = batch["gpt_token"].to(device)
        mask = batch["gpt_mask"].to(device)
        
        with torch.no_grad():
            prefix = clip_model.encode_image(images).to(device, dtype=torch.float32)

        outputs = model(tokens, prefix, mask)
        logits = outputs.logits[:, train_dataset.prefix_length - 1: -1]
        loss = nnf.cross_entropy(logits.reshape(-1, logits.size(-1)), tokens.flatten(), ignore_index=0)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        progress.set_postfix(loss=loss.item())
        progress.update()
            
        if (idx+1) % 1000 == 0:
            torch.save(train_loss_list, f"{save_path}/train_loss_list.pt")

    total_loss = train_loss / len(train_dataloader)
    train_loss_list.append(total_loss)
    progress.close()
    text = generate_test(model, clip_model, train_dataloader, agent.tokenizer, sample_num=10, device=device, temperature=0.63)
    print(f"Epoch {epoch} train loss: {total_loss}")
    print(text)
    

    if epoch % args.save_interval == 0 or epoch == args.epoch - 1:
        torch.save(model.state_dict(), f"{save_path}/clipcap_{epoch:03d}.pt")
        torch.save(train_loss_list, f"{save_path}/train_loss_list.pt")