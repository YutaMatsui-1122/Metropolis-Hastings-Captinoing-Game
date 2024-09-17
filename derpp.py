import torch
from one_agent import OneAgent
import pickle, clip, argparse
from utils import *
import sys
from torch.optim import AdamW
from torch.nn import functional as nnf
from ProbVLM.src.losses import *
from transformers import get_linear_schedule_with_warmup
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType
import pandas as pd

argparser = argparse.ArgumentParser()
argparser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
argparser.add_argument('--dataset', default="coco", choices=('coco', 'conceptual', 'fine_tune', 'cc3m','10000_pretrain'))
argparser.add_argument('--initial_epoch', type=int, default=0)
argparser.add_argument('--num_workers', type=int, default=1)
argparser.add_argument('--batch_size', type=int, default=64)
argparser.add_argument('--lr', type=float, default=2e-5)
argparser.add_argument('--save_dir', default="debug/")
argparser.add_argument('--device', default="cuda:0")
argparser.add_argument('--cl_mode', default="None", choices=('None','DER', 'DERPP', 'ER', 'ER_RS'))
argparser.add_argument('--reservoir_false', action='store_false', dest='reservoir')
argparser.add_argument('--speaker_agent', default="None", choices=("None", "coco", "conceptual"))
argparser.add_argument('--use_generated_caption', action='store_true', dest='use_generated_caption')
argparser.add_argument('--top_k', action='store_true', dest='top_k')

args = argparser.parse_args()

os.makedirs(f"models/{args.save_dir}", exist_ok=True)

device = torch.device(args.device)

clip_model, preprocess = clip.load(args.clip_model_type, device=device)

agent = OneAgent(agent_name='A', device=device)
agent = agent.to(device)

# args.speaker_agentの逆のエージェントのパラメータをロード
if args.speaker_agent == "None" or args.speaker_agent == "coco":
    agent.load_pretrain(probvlm_path="models/probVLM_conceptual_prefix-035.pth", clipcap_path="models/official_model/clipcap_conceptual_weights.pt", strict_clipcap=False)
    finetune_train_file = "coco_train_dataset_10000"
    finetune_test_file = "coco_test_dataset_5000"
    pretrain_train_file = "conceptual_train_dataset_30000"
    pretrain_test_file = "conceptual_test_dataset_11467"
    der_alpha = 0.5
    derpp_beta = 0.5
else:
    agent.load_pretrain(probvlm_path="models/probVLM_coco_prefix-035.pth", clipcap_path="models/official_model/clipcap_coco_weights.pt", strict_clipcap=False)
    finetune_train_file =  "coco_train_dataset_10000"
    finetune_test_file = "conceptual_test_dataset_11467"
    pretrain_train_file = "coco_train_dataset_30000"
    pretrain_test_file = "coco_test_dataset_5000"
    der_alpha = 0.5
    derpp_beta = 0.5

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

with open(f"dataset/dataset_cache/{finetune_train_file}.pkl", "rb") as f:
    finetune_train_dataset = pickle.load(f)
    finetune_train_dataset.prefix_length = agent.prefix_length
with open(f"dataset/dataset_cache/{finetune_test_file}.pkl", "rb") as f:
    finetune_valid_dataset = pickle.load(f)
    finetune_valid_dataset.prefix_length = agent.prefix_length

with open(f"dataset/dataset_cache/{pretrain_train_file}.pkl", "rb") as f:
    pretrain_train_dataset = pickle.load(f)
    pretrain_train_dataset.prefix_length = agent.prefix_length
with open(f"dataset/dataset_cache/{pretrain_test_file}.pkl", "rb") as f:
    pretrain_test_dataset = pickle.load(f)
    pretrain_test_dataset.prefix_length = agent.prefix_length

if args.speaker_agent == "coco" and args.use_generated_caption:
    captions_df = pd.read_csv(f"models/mh_chain/mh_chain_coco2conceptual.csv")
    captions = captions_df["Generated_99"].tolist()
    finetune_train_dataset = set_caption_to_dataset(finetune_train_dataset, captions=captions)
    print("chain_last_captions",finetune_train_dataset.captions[:10])

elif args.speaker_agent == "conceptual" and args.use_generated_caption:
    captions_df = pd.read_csv(f"models/mh_chain/mh_chain_conceptual2coco_0.9.csv")
    captions = captions_df["Generated_99"].tolist()
    finetune_train_dataset = set_caption_to_dataset(finetune_train_dataset, captions=captions)

finetune_train_dataloader = torch.utils.data.DataLoader(finetune_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
finetune_test_dataloader = torch.utils.data.DataLoader(finetune_valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

pretrain_train_dataloader = torch.utils.data.DataLoader(pretrain_train_dataset, batch_size=100, shuffle=False, num_workers=args.num_workers)
pretrain_test_dataloader = torch.utils.data.DataLoader(pretrain_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

print("finetune_caption:", finetune_train_dataset.captions[:10])

def clipcap_derpp(CLIP_Net, train_loader, finetune_test_loader,pretrain_test_loader, model, save_dir, epochs, lr = 2e-5, warmup_steps_percent = 0.1, output_prefix = "clipcap", save_every = 1, train_mode = "pretain", initial_epoch = 0, device = "cuda:0", buffer = None, alpha = 0.5, beta = 0.5):
    eval_finetune = True
    eval_pretrain = True
    model = model.to(device)
    model.train()

    CLIP_Net.eval()
    CLIP_Net.to(device)

    if initial_epoch > 0:
        model.load_state_dict(torch.load(os.path.join(save_dir, f"{output_prefix}-{initial_epoch:03d}.pt")))
        print("load model from", os.path.join(save_dir, f"{output_prefix}-{initial_epoch:03d}.pt"))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # for name, param in model.gpt.named_parameters():
    #     print(f"Parameter name: {name}")
    #     print()
    warmup_steps = int(warmup_steps_percent * epochs * len(train_loader))

    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=2, lora_alpha=32, lora_dropout=0.1, target_modules=["c_fc"])
    model.gpt = get_peft_model(model.gpt, peft_config)
    model.gpt.print_trainable_parameters()
    params = list(model.parameters())

    optimizer = AdamW(params, lr=lr)
    scheduler = get_linear_schedule_with_warmup(                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_loader)
    )

    loss_list = []
    loss_der_list = []
    loss_derpp_list = []
    finetune_test_loss = []
    pretrain_test_loss = []

    for epoch in range(initial_epoch, epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_loader), desc=output_prefix)
        epoch_loss = 0
        epoch_loss_der = 0
        epoch_loss_derpp = 0
        for idx, batch in enumerate(train_loader):
            model.zero_grad()
            img, caption, vlm_token, gpt_token, gpt_mask, _  = batch
            img, gpt_token, gpt_mask = img.to(device), gpt_token.to(device), gpt_mask.to(device)

            prefix = CLIP_Net.encode_image(img).to(device, dtype=torch.float32)
            
            outputs = model(gpt_token, prefix, gpt_mask)

            logits = outputs.logits[:, train_loader.dataset.prefix_length - 1: -1]

            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), gpt_token.flatten(), ignore_index=0)
            loss.backward(retain_graph=True)
            epoch_loss += loss.item()

            # set task index to epoch  
            task_index = torch.tensor([epoch+1] * img.shape[0], device="cpu")

            if train_mode == "ER_RS" or train_mode == "DER" or train_mode == "DERPP":
                if idx == 0 and epoch == 0:
                    print("reservoir")
                buffer.add(prefix.detach().cpu(), vlm_token.detach().cpu(), gpt_token.detach().cpu(), gpt_mask.detach().cpu(), logits.detach().cpu(), task_index)
            
            # Dark experience replay (DER)
            if train_mode == "DER" or train_mode == "DERPP":
                if args.top_k:
                    emb, vlm_token, gpt_token, gpt_mask, logits_value, logits_indices = buffer.sample(train_loader.batch_size)
                    emb, vlm_token, gpt_token, gpt_mask, logits_value, logits_indices = emb.to(device), vlm_token, gpt_token.to(device), gpt_mask.to(device), logits_value.to(device), logits_indices.to(device)
                else:
                    emb, vlm_token, gpt_token, gpt_mask, logits = buffer.sample(train_loader.batch_size)
                    emb, vlm_token, gpt_token, gpt_mask, logits = emb.to(device), vlm_token, gpt_token.to(device), gpt_mask.to(device), logits.to(device)

                outputs = model(gpt_token, emb, gpt_mask)
                outputs_logits = outputs.logits[:, train_loader.dataset.prefix_length - 1: -1]
                
                if args.top_k:
                    buffer_logits = outputs_logits.detach().clone()
                    buffer_logits = buffer.reconstruct_logits(buffer_logits, logits_value, logits_indices)
                else:
                    buffer_logits = logits
                # Dark experience replay (DER)
                # Use Euclidean distance between the logits of the current model and the logits of the model trained on the buffer
                loss_der = alpha * nnf.mse_loss(outputs_logits, buffer_logits)
                loss_der.backward(retain_graph=True)
                epoch_loss_der += loss_der.item()
                loss_der_list.append(loss_der.detach().cpu().item()/alpha)
                if idx ==0 and epoch == 0:
                    print("dark knowledge loss", loss_der.item(), "alpha", alpha)

            if train_mode == "DERPP" or train_mode == "ER" or train_mode == "ER_RS":
                if args.top_k:
                    emb, vlm_token, gpt_token, gpt_mask, logits_value, logits_indices = buffer.sample(train_loader.batch_size)
                    emb, vlm_token, gpt_token, gpt_mask = emb.to(device), vlm_token, gpt_token.to(device), gpt_mask.to(device)
                else:
                    emb, vlm_token, gpt_token, gpt_mask, logits = buffer.sample(train_loader.batch_size)
                    emb, vlm_token, gpt_token, gpt_mask, logits = emb.to(device), vlm_token, gpt_token.to(device), gpt_mask.to(device), logits.to(device)

                outputs = model(gpt_token, emb, gpt_mask)
                outputs_logits = outputs.logits[:, train_loader.dataset.prefix_length - 1: -1]
                
                # Dark experience replay ++ (DER++)
                # Use cross-entropy loss between the logits of the current model and correct tokens
                loss_derpp = beta * nnf.cross_entropy(outputs_logits.reshape(-1, outputs_logits.shape[-1]), gpt_token.flatten(), ignore_index=0)
                loss_derpp.backward(retain_graph=True)
                epoch_loss_derpp += loss_derpp.item()
                loss_derpp_list.append(loss_derpp.detach().cpu().item()/beta)
                if idx ==0 and epoch == 0:
                    print("replay loss", loss_derpp.item(), "beta", beta)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})

            progress.update()

            loss_list.append(loss.item())
            
        if train_mode != "None":
            print(np.bincount(buffer.get_task_ids().cpu().numpy()))
        if "DERPP" in train_mode:
            print(f"loss, der, derpp: {epoch_loss/len(train_loader)}, {epoch_loss_der/len(train_loader)/alpha}, {epoch_loss_derpp/len(train_loader)/beta}")
        elif "DER" in train_mode:
            print(f"loss, der: {epoch_loss/len(train_loader)}, {epoch_loss_der/len(train_loader)/alpha}")   
        elif "ER" in train_mode or "ER_RS" in train_mode:
            print(f"loss: {epoch_loss/len(train_loader)}")

        progress.close()
        np.save(os.path.join(save_dir, f"{output_prefix}_loss.npy"), np.array(loss_list))
        np.save(os.path.join(save_dir, f"{output_prefix}_loss_der.npy"), np.array(loss_der_list))
        np.save(os.path.join(save_dir, f"{output_prefix}_loss_derpp.npy"), np.array(loss_derpp_list))

        torch.save(
            model.state_dict(),
            os.path.join(save_dir, f"{output_prefix}-{epoch:03d}.pt"),
        )
        if eval_finetune:
            print(f">>> Evaluating finetune epoch {epoch}")
            test_loss = 0
            for idx, batch in tqdm(enumerate(finetune_test_loader), total=len(finetune_test_loader), desc="eval"):
                model.zero_grad() 
                img, _, _, gpt_token, gpt_mask,_ = batch

                img, gpt_token, gpt_mask = img.to(device), gpt_token.to(device), gpt_mask.to(device)

                prefix = CLIP_Net.encode_image(img).to(device, dtype=torch.float32)

                outputs = model(gpt_token, prefix, gpt_mask)
                logits = outputs.logits[:, finetune_test_loader.dataset.prefix_length - 1: -1]

                loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), gpt_token.flatten(), ignore_index=0)
                test_loss += loss.item()
            test_loss /= len(finetune_test_loader)
            print(f"finetune Test Loss: {test_loss}")
            finetune_test_loss.append(test_loss)
            np.save(os.path.join(save_dir, f"finetune_test_loss.npy"), finetune_test_loss)
        
        if eval_pretrain:
            print(f">>> Evaluating pretrain epoch {epoch}")
            test_loss = 0
            for idx, batch in tqdm(enumerate(pretrain_test_loader), total=len(pretrain_test_loader), desc="eval"):
                model.zero_grad() 
                img, _, _, gpt_token, gpt_mask,_ = batch

                img, gpt_token, gpt_mask = img.to(device), gpt_token.to(device), gpt_mask.to(device)

                prefix = CLIP_Net.encode_image(img).to(device, dtype=torch.float32)

                outputs = model(gpt_token, prefix, gpt_mask)
                logits = outputs.logits[:, pretrain_test_loader.dataset.prefix_length - 1: -1]

                loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), gpt_token.flatten(), ignore_index=0)
                test_loss += loss.item()

            test_loss /= len(pretrain_test_loader)
            print(f"pretrain Test Loss: {test_loss}")
            pretrain_test_loss.append(test_loss)            
            np.save(os.path.join(save_dir, f"pretrain_test_loss.npy"), pretrain_test_loss)
            
    return model

buffer = None

if args.cl_mode != "None":
    buffer_size = 10000
    # if exists buffer, load buffer

    # buffer = Buffer(buffer_size)
    if args.top_k:
        buffer = Top_k_Buffer(buffer_size, top_k=5000)
    else:
        buffer = ClipCapBuffer(buffer_size)
    caption_list = []

    for batch in tqdm(pretrain_train_dataloader):
        img, caption, vlm_token, gpt_token, gpt_mask, _  = batch
        img, gpt_token, gpt_mask = img.to(device), gpt_token.to(device), gpt_mask.to(device)
        embeddings = agent.CLIP_Net.encode_image(img).to(device, dtype=torch.float32)
        img = img.cpu()
        outputs = agent.ClipCap(gpt_token, embeddings, gpt_mask)
        logits = outputs.logits[:, pretrain_train_dataloader.dataset.prefix_length - 1: -1]

        task_index = torch.tensor([0] * img.shape[0], device="cpu")
        buffer.add(embeddings.detach().cpu(), vlm_token.detach().cpu(), gpt_token.detach().cpu(), gpt_mask.detach().cpu(), logits.detach().cpu(), task_index)
        caption_list.extend(caption)
        if buffer.num_seen_examples >= buffer_size:
            break

    # save caption_list as text file
    with open(f"models/{args.save_dir}/caption_list.txt", "w") as f:
        for caption in caption_list:
            f.write(caption + "\n")
    
    print("buffer size", buffer.num_seen_examples)

CL_mode = args.cl_mode

clipcap_derpp(agent.CLIP_Net, finetune_train_dataloader, finetune_test_dataloader, pretrain_test_dataloader, agent.ClipCap, f"models/{args.save_dir}", 100, lr = args.lr, warmup_steps_percent = 0.1, output_prefix = f"{args.dataset}_prefix", save_every = 1, train_mode = CL_mode, initial_epoch = args.initial_epoch, device = device, buffer = buffer, alpha=der_alpha, beta=derpp_beta)