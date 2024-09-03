import torch
from one_agent import OneAgent
import pickle, clip, argparse
from utils import *
from update_models import *
from transformers import AdamW, get_linear_schedule_with_warmup

argparser = argparse.ArgumentParser()
argparser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
argparser.add_argument('--dataset', default="coco", choices=('coco', 'conceptual', 'fine_tune', 'cc3m', 'fine_tune', '10000_pretrain', 'coco_cc3m'))
argparser.add_argument('--epoch', default=10, type=int)
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

clip_model, preprocess = clip.load(args.clip_model_type, device=device)

agent = OneAgent(agent_name='A', device=device, temperature=0.62)
model = agent.ClipCap.to(device)

if args.save_dir == "debug/":
    datasize_coco = 500
    datasize_cc3m = 500
    datasize_coco_test = 300
    datasize_cc3m_test = 300
else:
    datasize_coco = "full"
    datasize_cc3m = "full"
    datasize_coco_test = 30000
    datasize_cc3m_test = 30000

# データセットの初期化
train_dataset = UnifiedDataset(data_mode='train', transform=preprocess, datasize_coco=f"{datasize_coco}", datasize_cc3m=f"{datasize_cc3m}", prefix_length=agent.prefix_length)
test_dataset = UnifiedDataset(data_mode='test', transform=preprocess, datasize_coco=f"{datasize_coco_test}", datasize_cc3m=f"{datasize_cc3m_test}", prefix_length=agent.prefix_length)

print("prefix_length", train_dataset.prefix_length)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=40, shuffle=True, num_workers=args.num_workers)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=40, shuffle=False, num_workers=args.num_workers)

optimizer = AdamW(model.parameters(), lr=args.lr)

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=5000, num_training_steps=args.epoch * len(train_dataloader))


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
        if idx % 10000 == 0:
            torch.save(model.state_dict(), f"models/{args.save_dir}/clipcap_latest.pt")
    train_loss_list.append(train_loss / len(train_dataloader))
    progress.close()


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

    if epoch % args.save_interval == 0 or epoch == args.epoch - 1:
        torch.save(model.state_dict(), f"models/{args.save_dir}/clipcap_{epoch:03d}.pt")
        torch.save(train_loss_list, f"models/{args.save_dir}/train_loss_list.pt")
        torch.save(eval_loss_list, f"models/{args.save_dir}/eval_loss_list.pt")
        if eval_loss < best_loss:
            torch.save(model.state_dict(), f"models/{args.save_dir}/clipcap_best.pt")
            best_loss = eval_loss
