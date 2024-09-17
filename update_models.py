import os
import sys
from tqdm import tqdm
import time
import torch
import numpy as np
from torch.optim import AdamW
from torch.nn import functional as nnf
from ProbVLM.src.losses import *
from utils import *

from transformers import get_linear_schedule_with_warmup

import clip

def clipcap_optimizer(model, lr = 2e-5, update_mode = "all"):
    if update_mode == "all":
        optimizer = AdamW(model.parameters(), lr=lr)
    elif update_mode == "last": # only update the last layer
        final_layer_params = list(model.clip_project.model[-1].parameters())
        optimizer = AdamW(final_layer_params, lr=lr)
    return optimizer

def probvlm_optimizer(model, lr = 1e-4, update_mode = "all"):
    if update_mode == "all":
        optimizer = torch.optim.Adam(
            list(model.img_BayesCap.parameters())+list(model.txt_BayesCap.parameters()), 
            lr=lr
        )
    elif update_mode == "last": # only update the last layer
        final_layer_params = list(model.img_BayesCap.block_mu.parameters()) + \
                             list(model.img_BayesCap.block_alpha.parameters()) + \
                             list(model.img_BayesCap.block_beta.parameters()) + \
                             list(model.txt_BayesCap.block_mu.parameters()) + \
                             list(model.txt_BayesCap.block_alpha.parameters()) + \
                             list(model.txt_BayesCap.block_beta.parameters())
        optimizer = torch.optim.Adam(
            final_layer_params,
            lr=lr
        )
    return optimizer

def update_clipcap(CLIP_Net, train_loader, test_loader, model, save_dir, epochs, lr = 2e-5, warmup_steps_percent = 0.1, output_prefix = "clipcap", save_every = 1, train_mode = "pretain"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()

    CLIP_Net.eval()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    optimizer = AdamW(model.parameters(), lr=lr)

    loss_list = []
    test_loss_list = []
    
    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_loader), desc=output_prefix)
        for idx, batch in enumerate(train_loader):
            model.zero_grad() 
            img, _, _, gpt_token, gpt_mask, _  = batch

            img, gpt_token, gpt_mask = img.to(device), gpt_token.to(device), gpt_mask.to(device)

            prefix = CLIP_Net.encode_image(img).to(device, dtype=torch.float32)

            outputs = model(gpt_token, prefix, gpt_mask)
            logits = outputs.logits[:, train_loader.dataset.prefix_length - 1: -1]

            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), gpt_token.flatten(), ignore_index=0)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})
            progress.update()

            loss_list.append(loss.item())

            if (idx + 1) % 10 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(save_dir, f"{output_prefix}_latest.pt"),
                )       
                np.save(os.path.join(save_dir, f"{output_prefix}_loss.npy"), loss_list)

        progress.close()
        np.save(os.path.join(save_dir, f"{output_prefix}_loss.npy"), np.array(loss_list))
        print(sum(loss_list)/len(loss_list))

        if "MHNG" in train_mode:
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, f"{output_prefix}-{train_mode}.pt"),
            )

        elif train_mode == "pretrain":
            print(f">>> Evaluating")
            test_loss = 0
            for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc="eval"):
                model.zero_grad() 
                img, _, _, gpt_token, gpt_mask,_ = batch

                img, gpt_token, gpt_mask = img.to(device), gpt_token.to(device), gpt_mask.to(device)

                prefix = CLIP_Net.encode_image(img).to(device, dtype=torch.float32)

                outputs = model(gpt_token, prefix, gpt_mask)
                logits = outputs.logits[:, test_loader.dataset.prefix_length - 1: -1]

                loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), gpt_token.flatten(), ignore_index=0)
                test_loss += loss.item()

            test_loss /= len(test_loader)
            test_loss_list.append(test_loss)
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, f"{output_prefix}-{epoch:03d}.pt"),
            )

            np.save(os.path.join(save_dir, f"{output_prefix}_test_loss.npy"), test_loss_list)

    return model, loss_list, test_loss_list

def update_clipcap_derpp(agent_z, CLIP_Net, clipcap, tokenizer, train_loader_shuffle, train_loader_fix, save_dir, epochs, save_every = 1, lr = 2e-5, warmup_steps_percent = 0.1, output_prefix = "clipcap", train_mode = "None", device = "cuda:0", buffer = None, alpha = 0.5, beta = 0.5):
    print("start training clipcap with alpha:", alpha, "beta:", beta)
    clipcap = clipcap.to(device)
    clipcap.train()

    CLIP_Net.eval()
    CLIP_Net.to(device)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    warmup_steps = int(warmup_steps_percent * epochs * len(train_loader_shuffle))
    params = list(clipcap.parameters())

    optimizer = AdamW(params, lr=lr)
    scheduler = get_linear_schedule_with_warmup(                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_loader_shuffle)
    )

    loss_list = []
    loss_der_list = []
    loss_derpp_list = []

    generated_text = generate_test(clipcap, CLIP_Net, train_loader_fix, tokenizer, 10, device = device, prefix_length = 10, temperature = 0.6)
    print("before training(finetune)", generated_text)

    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_loader_shuffle), desc=output_prefix)
        for idx, batch in enumerate(train_loader_shuffle):
            clipcap.zero_grad()
            # img, caption, vlm_token, gpt_token, gpt_mask, index  = batch
            img, vlm_token, gpt_token, gpt_mask, index = batch["image"], batch["vlm_token"], batch["gpt_token"], batch["gpt_mask"], batch["index"]
            img, gpt_token, gpt_mask = img.to(device), gpt_token.to(device), gpt_mask.to(device)

            prefix = agent_z[index].to(device)
            
            outputs = clipcap(gpt_token, prefix, gpt_mask)

            logits = outputs.logits[:, train_loader_shuffle.dataset.prefix_length - 1: -1]

            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), gpt_token.flatten(), ignore_index=0)
            loss.backward(retain_graph=True)
            
            # set task index to epoch  
            task_index = torch.tensor([epoch+1] * img.shape[0], device="cpu")
            # Experience replay (ER)
            if train_mode == "ER_RS" or train_mode == "DER" or train_mode == "DERPP":
                if idx == 0 and epoch == 0:
                    print("reservoir")
                buffer.add(prefix.detach().cpu(), vlm_token.detach().cpu(), gpt_token.detach().cpu(), gpt_mask.detach().cpu(), logits.detach().cpu(), task_index)

            # Dark experience replay (DER)
            if train_mode == "DER" or train_mode == "DERPP":
                emb, vlm_token, gpt_token, gpt_mask, logits = buffer.sample(16)
                emb, vlm_token, gpt_token, gpt_mask, logits = emb.to(device), vlm_token, gpt_token.to(device), gpt_mask.to(device), logits.to(device)

                outputs = clipcap(gpt_token, emb, gpt_mask)
                outputs_logits = outputs.logits[:, train_loader_fix.dataset.prefix_length - 1: -1]

                # Dark experience replay (DER)
                # Use Euclidean distance between the logits of the current model and the logits of the model trained on the buffer
                loss_der = alpha * nnf.mse_loss(outputs_logits, logits)
                loss_der.backward(retain_graph=True)

                loss_der_list.append(loss_der.detach().cpu().item()/alpha)
                if idx ==0 and epoch == 0:
                    print("dark knowledge loss", loss_der.item(), "alpha", alpha)

            if train_mode == "DERPP" or train_mode == "ER" or train_mode == "ER_RS":
                emb, vlm_token, gpt_token, gpt_mask, logits = buffer.sample(16)
                emb, vlm_token, gpt_token, gpt_mask, logits = emb.to(device), vlm_token, gpt_token.to(device), gpt_mask.to(device), logits.to(device)

                outputs = clipcap(gpt_token, emb, gpt_mask)
                outputs_logits = outputs.logits[:, train_loader_fix.dataset.prefix_length - 1: -1]
                
                # Dark experience replay ++ (DER++)
                # Use cross-entropy loss between the logits of the current model and correct tokens
                loss_derpp = beta * nnf.cross_entropy(outputs_logits.reshape(-1, outputs_logits.shape[-1]), gpt_token.flatten(), ignore_index=0)
                loss_derpp.backward(retain_graph=True)
                loss_derpp_list.append(loss_derpp.detach().cpu().item()/beta)
                if idx ==0 and epoch == 0:
                    print("replay loss", loss_derpp.item(), "beta", beta)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})

            progress.update()

            loss_list.append(loss.item())
        generated_text = generate_test(clipcap, CLIP_Net, train_loader_fix, tokenizer, 10, device = device, prefix_length = 10, temperature = 0.6)
        print("epoch(finetune)", epoch, generated_text)
            
        if train_mode != "None":
            print(np.bincount(buffer.get_task_ids().cpu().numpy()))
        if "DERPP" in train_mode:
            print(f"loss, der, derpp: {loss.item()}, {loss_der.item()/alpha}, {loss_derpp.item()}")
        elif "DER" in train_mode:
            print(f"loss, der: {loss.item()}, {loss_der.item()/alpha}")
        elif "ER" in train_mode or "ER_RS" in train_mode:
            print(f"loss: {loss.item()}")

        progress.close()
        np.save(os.path.join(save_dir, f"{output_prefix}_loss.npy"), np.array(loss_list))
        np.save(os.path.join(save_dir, f"{output_prefix}_loss_der.npy"), np.array(loss_der_list))
        np.save(os.path.join(save_dir, f"{output_prefix}_loss_derpp.npy"), np.array(loss_derpp_list))

        if save_every > 0 and epoch+1 % save_every == 0 or epoch == 0 or epoch == epochs-1:
            torch.save(
                clipcap.state_dict(),
                os.path.join(save_dir, f"{output_prefix}-{epoch:03d}.pt"),
            )
        s = time.time()

    return clipcap

def update_probvlm(agent_z, CLIP_Net, BayesCap_Net, train_loader, save_dir, epochs, save_every=1, lr = 1e-4, output_prefix = "probvlm", device="cuda:0", Cri = TempCombLoss(), T1=1e0, T2=5e-2, cross_modal_lambda=1e-4):
    
    BayesCap_Net = BayesCap_Net.to(device)
    BayesCap_Net.img_BayesCap.eval()
    BayesCap_Net.txt_BayesCap.train()
    CLIP_Net.to(device)
    CLIP_Net.eval()

    model_params = [{'params': BayesCap_Net.txt_BayesCap.block_mu[2].parameters()},
                    {'params': BayesCap_Net.txt_BayesCap.block_alpha[2].parameters()},
                    {'params': BayesCap_Net.txt_BayesCap.block_beta[2].parameters()}]

    # model_params = [{"params": BayesCap_Net.txt_BayesCap.parameters()}]

    optimizer = torch.optim.Adam(model_params, lr=lr)

    loss_list = []
    for eph in tqdm(range(epochs)):
        eph_loss = 0
        BayesCap_Net.train()
        # with tqdm(train_loader, unit = 'batch') as tepoch:
        for batch in train_loader:
            # tepoch.set_description('Epoch {}'.format(eph))

            # vlm_token = batch[2].to(device)
            # index = batch[5]
            vlm_token = batch["vlm_token"].to(device)
            index = batch["index"]

            z = agent_z[index].to(device)

            with torch.no_grad():
                text_emb = CLIP_Net.encode_text(vlm_token).to(device, dtype=torch.float32)
            
            # (img_mu, img_alpha, img_beta), (txt_mu, txt_alpha, txt_beta) = BayesCap_Net(xfI, xfT)
            txt_mu, txt_alpha, txt_beta = BayesCap_Net.txt_BayesCap(text_emb)

            optimizer.zero_grad()

            loss= Cri(txt_mu, txt_alpha, txt_beta, z, T1=T1, T2=T2)

            loss.backward()
            optimizer.step()

            eph_loss += loss.item()
            loss_list.append(loss.item())

        eph_loss /= len(train_loader)
        print('Epoch loss: {}'.format(eph_loss))
        
        
        np.save(os.path.join(save_dir, f"{output_prefix}_loss.npy"), np.array(loss_list)) 

        if save_every > 0 and eph+1 % save_every == 0 or eph == 0 or eph == epochs-1:
            torch.save(BayesCap_Net.state_dict(), os.path.join(save_dir, f"{output_prefix}-epoch-{eph}.pth"))

    return BayesCap_Net

def update_probvlm_derpp(agent_z, CLIP_Net, BayesCap_Net, train_loader, save_dir, epochs, save_every=1, lr = 1e-4, output_prefix = "probvlm", device="cuda:0", Cri = TempCombLoss(), T1=1e0, T2=5e-2, train_mode = "None", buffer = None, alpha = 0.5, beta = 0.5):
    
    BayesCap_Net = BayesCap_Net.to(device)
    BayesCap_Net.img_BayesCap.eval()
    BayesCap_Net.txt_BayesCap.train()
    CLIP_Net.to(device)
    CLIP_Net.eval()

    model_params = [{'params': BayesCap_Net.txt_BayesCap.block_mu[2].parameters()},
                    {'params': BayesCap_Net.txt_BayesCap.block_alpha[2].parameters()},
                    {'params': BayesCap_Net.txt_BayesCap.block_beta[2].parameters()}]

    # model_params = [{"params": BayesCap_Net.txt_BayesCap.parameters()}]

    optimizer = torch.optim.Adam(model_params, lr=lr)

    loss_list = []
    loss_der_list = []
    loss_derpp_list = []
    
    for eph in tqdm(range(epochs)):
        eph_loss = 0
        eph_der_loss = 0
        eph_derpp_loss = 0
        BayesCap_Net.train()
        # with tqdm(train_loader, unit = 'batch') as tepoch:
        for idx, batch in enumerate(train_loader):
            # tepoch.set_description('Epoch {}'.format(eph))

            # vlm_token = batch[2].to(device)
            # index = batch[5]
            vlm_token = batch["vlm_token"].to(device)
            index = batch["index"]

            z = agent_z[index].to(device)

            with torch.no_grad():
                text_emb = CLIP_Net.encode_text(vlm_token).to(device, dtype=torch.float32)
            
            # (img_mu, img_alpha, img_beta), (txt_mu, txt_alpha, txt_beta) = BayesCap_Net(xfI, xfT)
            txt_mu, txt_alpha, txt_beta = BayesCap_Net.txt_BayesCap(text_emb)

            optimizer.zero_grad()

            loss= Cri(txt_mu, txt_alpha, txt_beta, z, T1=T1, T2=T2)

            loss.backward()
            optimizer.step()

            eph_loss += loss.item()
            loss_list.append(loss.item())
            
            # set task index to epoch
            task_index = torch.tensor([eph+1] * vlm_token.shape[0], device="cpu")
            if train_mode == "ER_RS" or train_mode == "DER" or train_mode == "DERPP":
                if idx == 0 and eph == 0:
                    print("use reservoir")
                # buffer.add(prefix.detach().cpu(), vlm_token.detach().cpu(), gpt_token.detach().cpu(), gpt_mask.detach().cpu(), logits.detach().cpu(), task_index)
                buffer.add(text_emb.detach().cpu(), z.detach().cpu(), txt_mu.detach().cpu(), txt_alpha.detach().cpu(), txt_beta.detach().cpu(), task_index)

            if train_mode == "DER" or train_mode == "DERPP":
                text_emb, z, txt_mu, txt_alpha, txt_beta = buffer.sample(16)
                text_emb, z, txt_mu, txt_alpha, txt_beta = text_emb.to(device), z.to(device), txt_mu.to(device), txt_alpha.to(device), txt_beta.to(device)

                txt_mu_pred, txt_alpha_pred, txt_beta_pred = BayesCap_Net.txt_BayesCap(text_emb)

                loss_der = alpha * nnf.mse_loss(txt_mu_pred, txt_mu) + nnf.mse_loss(txt_alpha_pred, txt_alpha) + nnf.mse_loss(txt_beta_pred, txt_beta)
                loss_der.backward(retain_graph=True)

                if idx == 0 and eph == 0:
                    print("dark knowledge loss", loss_der.item(), "alpha", alpha)

                eph_der_loss += loss_der.item()                
                loss_der_list.append(loss_der.item()/alpha)
                

            if train_mode == "DERPP" or train_mode == "ER" or train_mode == "ER_RS":
                text_emb, z, txt_mu, txt_alpha, txt_beta = buffer.sample(16)
                text_emb, z, txt_mu, txt_alpha, txt_beta = text_emb.to(device), z.to(device), txt_mu.to(device), txt_alpha.to(device), txt_beta.to(device)

                txt_mu_pred, txt_alpha_pred, txt_beta_pred = BayesCap_Net.txt_BayesCap(text_emb)

                loss_derpp = beta * Cri(txt_mu_pred, txt_alpha_pred, txt_beta_pred, z, T1=T1, T2=T2)
                loss_derpp.backward(retain_graph=True)

                if idx == 0 and eph == 0:
                    print("replay loss", loss_derpp.item(), "beta", beta)
                
                eph_derpp_loss += loss_derpp.item()
                loss_derpp_list.append(loss_derpp.item()/beta)

            optimizer.step()

        eph_loss /= len(train_loader)
        eph_der_loss /= len(train_loader)
        eph_derpp_loss /= len(train_loader)
        print('Epoch loss: {}, der loss: {}, derpp loss: {}'.format(eph_loss, eph_der_loss, eph_derpp_loss))        
        
        np.save(os.path.join(save_dir, f"{output_prefix}_loss.npy"), np.array(loss_list)) 
        np.save(os.path.join(save_dir, f"{output_prefix}_loss_der.npy"), np.array(loss_der_list))
        np.save(os.path.join(save_dir, f"{output_prefix}_loss_derpp.npy"), np.array(loss_derpp_list))

        if save_every > 0 and eph+1 % save_every == 0 or eph == 0 or eph == epochs-1:
            torch.save(BayesCap_Net.state_dict(), os.path.join(save_dir, f"{output_prefix}-epoch-{eph}.pth"))

    return BayesCap_Net


def pretrain_clipcap(CLIP_Net, train_loader, test_loader, model, save_dir, epochs, lr = 2e-5, warmup_steps_percent = 0.1, output_prefix = "clipcap", save_every = 1, train_mode = "pretain", device = "cuda:0", ff_mode = "full"):
    
    model = model.to(device)
    model.train()

    CLIP_Net.eval()
    CLIP_Net.to(device)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    warmup_steps = int(warmup_steps_percent * epochs * len(train_loader))
    print("clipcap lr:", lr)

    if ff_mode == "full":
        print("full fine-tuning")
        model.gpt.train()
        params = list(model.parameters()) + list(CLIP_Net.parameters())
        total_params = sum(p.numel() for p in params if p.requires_grad)
        print(f"Total trainable parameters: {total_params}")

    elif ff_mode == "adapter":
        print("adapter fine-tuning")
        params = list(model.clip_project.parameters())
        total_params = sum(p.numel() for p in params if p.requires_grad)
        print(f"Total trainable parameters: {total_params}")

    elif ff_mode == "decoder":
        print("decoder fine-tuning")
        model.gpt.train()
        params = list(model.gpt.parameters()) + list(model.clip_project.parameters())
        total_params = sum(p.numel() for p in params if p.requires_grad)
        print(f"Total trainable parameters: {total_params}")
    
    elif ff_mode == "original":
        print("original pre-training")
        params = list(model.parameters())
        total_params = sum(p.numel() for p in params if p.requires_grad)
        print(f"Total trainable parameters: {total_params}")

    optimizer = AdamW(params, lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_loader)
    )
    # total_params = sum(p.numel() for p in params if p.requires_grad)

    # print(f"Total trainable parameters: {total_params}")
    # for name, param in model.named_parameters():
    #     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

    loss_list = []
    test_loss_list = []
    #時間計測開始
    print("時間計測開始")

    for epoch in range (epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_loader), desc=output_prefix)
        for idx, batch in enumerate(train_loader):
            model.zero_grad()

            img, _, _, gpt_token, gpt_mask, _  = batch

            img, gpt_token, gpt_mask = img.to(device), gpt_token.to(device), gpt_mask.to(device)

            start_forward = time.time()
            prefix = CLIP_Net.encode_image(img).to(device, dtype=torch.float32)
            outputs = model(gpt_token, prefix, gpt_mask)
            forward_time = time.time() - start_forward

            logits = outputs.logits[:, train_loader.dataset.prefix_length - 1: -1]

            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), gpt_token.flatten(), ignore_index=0)

            start_backward = time.time()
            loss.backward()
            backward_time = time.time() - start_backward
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})
            progress.update()

            loss_list.append(loss.item())
                    
        progress.close()
        np.save(os.path.join(save_dir, f"{output_prefix}_loss.npy"), np.array(loss_list))

        if "MHNG" in train_mode:
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, f"{output_prefix}-{train_mode}.pt"),
            )

        elif train_mode == "pretrain":
            print(f">>> Evaluating")
            test_loss = 0
            for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc="eval"):
                model.zero_grad() 
                img, _, _, gpt_token, gpt_mask,_,_ = batch

                img, gpt_token, gpt_mask = img.to(device), gpt_token.to(device), gpt_mask.to(device)

                prefix = CLIP_Net.encode_image(img).to(device, dtype=torch.float32)

                outputs = model(gpt_token, prefix, gpt_mask)
                logits = outputs.logits[:, test_loader.dataset.prefix_length - 1: -1]

                loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), gpt_token.flatten(), ignore_index=0)
                test_loss += loss.item()

            test_loss /= len(test_loader)
            print(f"Test Loss: {test_loss}")
            test_loss_list.append(test_loss)
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, f"{output_prefix}-{epoch:03d}.pt"),
            )
            
            np.save(os.path.join(save_dir, f"{output_prefix}_test_loss.npy"), test_loss_list)

    return model

def pretrain_probvlm(CLIP_Net, train_loader, test_loader, BayesCap_Net, save_dir, epochs, lr = 1e-4, output_prefix = "probvlm", save_every = 1,Cri = TempCombLoss(), T1=1e0, T2=5e-2, cross_modal_lambda=1e-4, train_mode = "pretain"):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    BayesCap_Net = BayesCap_Net.to(device)
    BayesCap_Net.img_BayesCap.train()
    BayesCap_Net.txt_BayesCap.train()
    CLIP_Net.to(device)
    CLIP_Net.eval()

    print("probvlm lr:", lr)
    optimizer = torch.optim.Adam(
        list(BayesCap_Net.img_BayesCap.parameters())+list(BayesCap_Net.txt_BayesCap.parameters()), 
        lr=lr
    )
    optim_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    loss_list = []
    test_loss_list = []
    for eph in range(epochs):
        eph_loss = 0
        BayesCap_Net.train()
        # with tqdm(train_loader, unit = 'batch') as tepoch:
        for (idx, batch) in tqdm(enumerate(train_loader)):
            # tepoch.set_description('Epoch {}'.format(eph))
            ##
            xI, xT = batch[0].to(device), batch[2].to(device)

            with torch.no_grad():
                xfI = CLIP_Net.encode_image(xI)
                xfT = CLIP_Net.encode_text(xT)
            
            (img_mu, img_alpha, img_beta), (txt_mu, txt_alpha, txt_beta) = BayesCap_Net(xfI, xfT)

            optimizer.zero_grad()
            loss_i = Cri(img_mu, img_alpha, img_beta, xfI, T1=T1, T2=T2)
            loss_t = Cri(txt_mu, txt_alpha, txt_beta, xfT, T1=T1, T2=T2)

            loss_i2t = Cri(img_mu, img_alpha, img_beta, xfT, T1=T1, T2=T2)
            loss_t2i = Cri(txt_mu, txt_alpha, txt_beta, xfI, T1=T1, T2=T2)

            loss = loss_i + loss_t + cross_modal_lambda * (loss_i2t + loss_t2i)

            loss.backward()
            optimizer.step()

            eph_loss += loss.item()
            loss_list.append(loss.item())
            # tepoch.set_postfix(loss=loss.item())
            
            if idx % 500 == 0:
                torch.save(BayesCap_Net.state_dict(), os.path.join(save_dir, f"{output_prefix}_latest.pth"))
                np.save(os.path.join(save_dir, f"{output_prefix}_loss.npy"), loss_list)
            
            eph_loss /= len(train_loader)
            optim_scheduler.step()
        print('Epoch loss: {}'.format(eph_loss))
        

        if "MHNG" in train_mode:
            torch.save(BayesCap_Net.state_dict(), os.path.join(save_dir, f"{output_prefix}-{train_mode}.pth"))

    
        elif train_mode == "pretrain":
            
            BayesCap_Net.eval()
            test_loss = 0
            for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc="eval"):
                xI, xT = batch[0].to(device), batch[2].to(device)

                with torch.no_grad():
                    xfI = CLIP_Net.encode_image(xI)
                    xfT = CLIP_Net.encode_text(xT)

                (img_mu, img_alpha, img_beta), (txt_mu, txt_alpha, txt_beta) = BayesCap_Net(xfI, xfT)

                loss_i = Cri(img_mu, img_alpha, img_beta, xfI, T1=T1, T2=T2)
                loss_t = Cri(txt_mu, txt_alpha, txt_beta, xfT, T1=T1, T2=T2)

                loss_i2t = Cri(img_mu, img_alpha, img_beta, xfT, T1=T1, T2=T2)
                loss_t2i = Cri(txt_mu, txt_alpha, txt_beta, xfI, T1=T1, T2=T2)

                loss = loss_i + loss_t + cross_modal_lambda * (loss_i2t + loss_t2i)

                test_loss += loss.item()

            test_loss /= len(test_loader)
            test_loss_list.append(test_loss)
            np.save(os.path.join(save_dir, f"{output_prefix}_test_loss.npy"), test_loss_list)
            torch.save(BayesCap_Net.state_dict(), os.path.join(save_dir, f"{output_prefix}-{eph:03d}.pth"))
        
    return BayesCap_Net