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
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


from transformers import get_linear_schedule_with_warmup


def update_clipcap_derpp(agent_z, CLIP_Net, clipcap, tokenizer, train_loader_shuffle, train_loader_fix, save_dir, epochs, save_every = 1, lr = 2e-5, warmup_steps_percent = 0.1, output_prefix = "clipcap", train_mode = "None", device = "cuda:0", buffer = None, alpha = 0.5, beta = 0.5, reserovoir = True):
    
    # Update the clipcap model with dark experience replay ++
    clipcap = clipcap.to(device)
    clipcap.train()

    CLIP_Net.eval()
    CLIP_Net.to(device)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    optimizer = torch.optim.AdamW(clipcap.parameters(), lr=lr)

    warmup_steps = int(warmup_steps_percent * epochs * len(train_loader_shuffle))
    scheduler = get_linear_schedule_with_warmup(                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_loader_shuffle)
    )

    loss_list = []
    loss_der_list = []
    loss_derpp_list = []

    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_loader_shuffle), desc=output_prefix)
        eph_loss = 0
        eph_der_loss = 0
        eph_derpp_loss = 0
        for idx, batch in enumerate(train_loader_shuffle):
            clipcap.zero_grad()
            img, vlm_token, gpt_token, gpt_mask, index = batch["image"], batch["vlm_token"], batch["gpt_token"], batch["gpt_mask"], batch["index"]
            caption = batch["caption"]
            img, gpt_token, gpt_mask = img.to(device), gpt_token.to(device), gpt_mask.to(device)

            batch_size = img.shape[0]

            prefix = agent_z[index].to(device)
            
            outputs = clipcap(gpt_token, prefix, gpt_mask)

            logits = outputs.logits[:, train_loader_shuffle.dataset.prefix_length - 1: -1]

            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), gpt_token.flatten(), ignore_index=0)
            loss.backward(retain_graph=True)

            eph_loss += loss.item()
            
            # set task index to epoch  
            task_index = torch.tensor([epoch+1] * img.shape[0], device="cpu")
            # Experience replay (ER)
            if (train_mode == "ER_RS" or train_mode == "DER" or train_mode == "DERPP") and reserovoir:
                if idx == 0 and epoch == 0:
                    print("use reservoir")
                buffer.add(prefix.detach().cpu(), vlm_token.detach().cpu(), gpt_token.detach().cpu(), gpt_mask.detach().cpu(), logits.detach().cpu(), task_index)

            # Loss for Dark experience replay (DER)
            if train_mode == "DER" or train_mode == "DERPP":
                emb, vlm_token, gpt_token, gpt_mask, logits = buffer.sample(batch_size)
                emb, vlm_token, gpt_token, gpt_mask, logits = emb.to(device), vlm_token, gpt_token.to(device), gpt_mask.to(device), logits.to(device)

                outputs = clipcap(gpt_token, emb, gpt_mask)
                outputs_logits = outputs.logits[:, train_loader_fix.dataset.prefix_length - 1: -1]

                # Dark experience replay (DER)
                # Use Euclidean distance between the logits of the current model and the logits of the model trained on the buffer

                loss_der = alpha * nnf.mse_loss(outputs_logits, logits)
                loss_der.backward(retain_graph=True)

                eph_der_loss += loss_der.item()

                loss_der_list.append(loss_der.detach().cpu().item()/alpha)
                if idx ==0 and epoch == 0:
                    print("dark knowledge loss", loss_der.item(), "alpha", alpha)

            # Loss for Dark experience replay ++ (DER++)
            if train_mode == "DERPP" or train_mode == "ER" or train_mode == "ER_RS":
                emb, vlm_token, gpt_token, gpt_mask, logits = buffer.sample(batch_size)
                emb, vlm_token, gpt_token, gpt_mask, logits = emb.to(device), vlm_token, gpt_token.to(device), gpt_mask.to(device), logits.to(device)

                outputs = clipcap(gpt_token, emb, gpt_mask)
                outputs_logits = outputs.logits[:, train_loader_fix.dataset.prefix_length - 1: -1]
                
                loss_derpp = beta * nnf.cross_entropy(outputs_logits.reshape(-1, outputs_logits.shape[-1]), gpt_token.flatten(), ignore_index=0)
                loss_derpp.backward(retain_graph=True)

                eph_derpp_loss += loss_derpp.item()
                loss_derpp_list.append(loss_derpp.detach().cpu().item()/beta)
                if idx ==0 and epoch == 0:
                    print("replay loss", loss_derpp.item(), "beta", beta)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})

            progress.update()

            loss_list.append(loss.item())

        if "DERPP" in train_mode:
            print(f"loss, der, derpp: {eph_loss/len(train_loader_shuffle)}, {eph_der_loss/len(train_loader_shuffle)/alpha}, {eph_derpp_loss/len(train_loader_shuffle)/beta}")
        elif "DER" in train_mode:
            print(f"loss, der: {eph_loss/len(train_loader_shuffle)}, {eph_der_loss/len(train_loader_shuffle)/alpha}")
        elif "ER" in train_mode or "ER_RS" in train_mode or train_mode == "None":
            print(f"loss: {eph_loss/len(train_loader_shuffle)}")

        progress.close()
        np.save(os.path.join(save_dir, f"{output_prefix}_loss.npy"), np.array(loss_list))
        np.save(os.path.join(save_dir, f"{output_prefix}_loss_der.npy"), np.array(loss_der_list))
        np.save(os.path.join(save_dir, f"{output_prefix}_loss_derpp.npy"), np.array(loss_derpp_list))

        if (save_every > 0 and (epoch + 1) % save_every == 0) or epoch == 0 or epoch == epochs - 1:
            torch.save(
                clipcap.state_dict(),
                os.path.join(save_dir, f"{output_prefix}-{epoch:03d}.pt"),
            )
        
    return clipcap

def update_distillation_merge(listener_agent, speaker_agent, train_loader_shuffle, train_loader_fix, save_dir, epochs, save_every = 1, lr = 2e-5, warmup_steps_percent = 0.1, output_prefix = "distillation_merge", device = "cuda:0", train_mode = "None", buffer = None, alpha = 0.5, beta = 0.5, reservoir = True):
    """
    Distillation merge training for clipcap model
    """

    print("start training distillation_merge with alpha:", alpha, "beta:", beta, "train_mode:", train_mode)

    listener_agent.to(device)
    speaker_agent.to(device)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    optimizer = torch.optim.AdamW(listener_agent.ClipCap.parameters(), lr=lr)

    warmup_steps = int(warmup_steps_percent * epochs * len(train_loader_shuffle))
    scheduler = get_linear_schedule_with_warmup(                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_loader_shuffle)
    )

    listener_agent.ClipCap.train()
    listener_agent.CLIP_Net.eval()
    speaker_agent.eval()

    loss_list = []
    loss_der_list = []
    loss_derpp_list = []

    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_loader_shuffle), desc=output_prefix)
        eph_loss = 0
        eph_der_loss = 0
        eph_derpp_loss = 0
        for idx, batch in enumerate(train_loader_shuffle):
            listener_agent.ClipCap.zero_grad()
            img, _, gpt_token, gpt_mask, index = batch["image"], batch["vlm_token"], batch["gpt_token"], batch["gpt_mask"], batch["index"]
            
            img, gpt_token, gpt_mask = img.to(device), gpt_token.to(device), gpt_mask.to(device)

            batch_size = img.shape[0]

            # output by speaker
            with torch.no_grad():
                _, _, _, z = speaker_agent.image_encoder(img)
                outputs = speaker_agent.ClipCap(gpt_token, z, gpt_mask)
                logits = outputs.logits[:, train_loader_shuffle.dataset.prefix_length - 1: -1]
                target_prob = nnf.softmax(logits, dim=-1)

            # output by listener
            _, _, _, z = listener_agent.image_encoder(img)
            outputs = listener_agent.ClipCap(gpt_token, z, gpt_mask)
            logits_listener = outputs.logits[:, train_loader_shuffle.dataset.prefix_length - 1: -1]
            log_probs_listener = nnf.log_softmax(logits_listener, dim=-1)

            # KL divergence loss
            loss = nnf.kl_div(log_probs_listener, target_prob, reduction="batchmean")

            loss.backward(retain_graph=True)

            eph_loss += loss.item()

            # set task index to epoch
            task_index = torch.tensor([epoch+1] * img.shape[0], device="cpu")
            # Experience replay (ER)
            if (train_mode == "ER_RS" or train_mode == "DER" or train_mode == "DERPP") and reservoir:
                if idx == 0 and epoch == 0:
                    print("use reservoir")
                buffer.add(z.detach().cpu(), gpt_token.detach().cpu(), gpt_mask.detach().cpu(), logits.detach().cpu(), task_index)

            # Dark experience replay (DER)
            if train_mode == "DER" or train_mode == "DERPP":
                emb, vlm_token, gpt_token, gpt_mask, logits = buffer.sample(batch_size)
                emb, vlm_token, gpt_token, gpt_mask, logits = emb.to(device), vlm_token, gpt_token.to(device), gpt_mask.to(device), logits.to(device)

                outputs = listener_agent.ClipCap(gpt_token, emb, gpt_mask)
                outputs_logits = outputs.logits[:, train_loader_fix.dataset.prefix_length - 1: -1]

                # Dark experience replay (DER)
                # Use Euclidean distance between the logits of the current model and the logits of the model trained on the buffer

                loss_der = alpha * nnf.mse_loss(outputs_logits, logits)
                loss_der.backward(retain_graph=True)

                eph_der_loss += loss_der.item()

                loss_der_list.append(loss_der.detach().cpu().item()/alpha)
                if idx ==0 and epoch == 0:
                    print("dark knowledge loss", loss_der.item(), "alpha", alpha)

            if train_mode == "DERPP" or train_mode == "ER" or train_mode == "ER_RS":
                emb, vlm_token, gpt_token, gpt_mask, logits = buffer.sample(batch_size)
                emb, vlm_token, gpt_token, gpt_mask, logits = emb.to(device), vlm_token, gpt_token.to(device), gpt_mask.to(device), logits.to(device)

                outputs = listener_agent.ClipCap(gpt_token, emb, gpt_mask)
                outputs_logits = outputs.logits[:, train_loader_fix.dataset.prefix_length - 1: -1]
                
                # Dark experience replay ++ (DER++)
                # Use cross-entropy loss between the logits of the current model and correct tokens
                loss_derpp = beta * nnf.cross_entropy(outputs_logits.reshape(-1, outputs_logits.shape[-1]), gpt_token.flatten(), ignore_index=0)
                loss_derpp.backward(retain_graph=True)

                eph_derpp_loss += loss_derpp.item()
                loss_derpp_list.append(loss_derpp.detach().cpu().item()/beta)
                if idx ==0 and epoch == 0:
                    print("replay loss", loss_derpp.item(), "beta", beta)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})
            progress.update()

            loss_list.append(loss.item())

        if "DERPP" in train_mode:
            print(f"loss, der, derpp: {eph_loss/len(train_loader_shuffle)}, {eph_der_loss/len(train_loader_shuffle)/alpha}, {eph_derpp_loss/len(train_loader_shuffle)/beta}")
        elif "DER" in train_mode:
            print(f"loss, der: {eph_loss/len(train_loader_shuffle)}, {eph_der_loss/len(train_loader_shuffle)/alpha}")
        elif "ER" in train_mode or "ER_RS" in train_mode or train_mode == "None":
            print(f"loss: {eph_loss/len(train_loader_shuffle)}")

        progress.close()
        np.save(os.path.join(save_dir, f"{output_prefix}_loss.npy"), np.array(loss_list))
        np.save(os.path.join(save_dir, f"{output_prefix}_loss_der.npy"), np.array(loss_der_list))
        np.save(os.path.join(save_dir, f"{output_prefix}_loss_derpp.npy"), np.array(loss_derpp_list))

        if (save_every > 0 and (epoch + 1) % save_every == 0) or epoch == 0 or epoch == epochs - 1:
            torch.save(
                listener_agent.ClipCap.state_dict(),
                os.path.join(save_dir, f"{output_prefix}-{epoch:03d}.pt"),
            )
    return listener_agent.ClipCap

def update_probvlm_derpp(agent_z, CLIP_Net, BayesCap_Net, train_loader, save_dir, epochs, save_every=1, lr = 1e-5, output_prefix = "probvlm", device="cuda:0", Cri = TempCombLoss(), T1=1, T2=1, cross_modal_lambda=0.9, train_mode = "None", buffer = None, alpha = 0.5, beta = 0.5, pretrain_test_loader = None, fine_tune_test_loader = None, reservoir = True):
    # Update the probvlm model with dark experience replay ++
    
    BayesCap_Net.txt_BayesCap.eval()
    # check the model before training
    text_emb, z, txt_mu, txt_alpha, txt_beta = buffer.sample(16)
    text_emb, z, txt_mu, txt_alpha, txt_beta = text_emb.to(device), z.to(device), txt_mu.to(device), txt_alpha.to(device), txt_beta.to(device)

    txt_mu_pred, txt_alpha_pred, txt_beta_pred = BayesCap_Net.txt_BayesCap(text_emb)

    loss_der = nnf.mse_loss(txt_mu_pred, txt_mu) + nnf.mse_loss(txt_alpha_pred, txt_alpha) + nnf.mse_loss(txt_beta_pred, txt_beta)

    BayesCap_Net = BayesCap_Net.to(device)
    BayesCap_Net.img_BayesCap.eval()
    BayesCap_Net.txt_BayesCap.train()
    CLIP_Net.to(device)
    CLIP_Net.eval()

    # use all parameters in BayesCap_Net.txt_BayesCap
    model_params = [{"params": BayesCap_Net.txt_BayesCap.parameters()}]

    optimizer = torch.optim.Adam(model_params, lr=lr)

    loss_list = []
    loss_der_list = []
    loss_derpp_list = []
    loss_pretrain_test_list = []
    loss_fine_tune_test_list = []
    
    # for eph in tqdm(range(epochs)):
    for eph in tqdm(range(epochs)):
        eph_loss = 0
        eph_der_loss = 0
        eph_derpp_loss = 0
        BayesCap_Net.train()
        for idx, batch in enumerate(train_loader):
            vlm_token = batch["vlm_token"].to(device)
            index = batch["index"]

            z = agent_z[index].to(device)

            with torch.no_grad():
                text_emb = CLIP_Net.encode_text(vlm_token).to(device, dtype=torch.float32)
            
            txt_mu, txt_alpha, txt_beta = BayesCap_Net.txt_BayesCap(text_emb)

            optimizer.zero_grad()

            loss_i = Cri(txt_mu, txt_alpha, txt_beta, z, T1=0, T2=1)
            loss_reg = Cri(txt_mu, txt_alpha, txt_beta, text_emb, T1=0, T2=1)

            loss =  loss_i + cross_modal_lambda *  loss_reg

            loss.backward()
            optimizer.step()

            eph_loss += loss.item()
            loss_list.append(loss.item())
            
            # set task index to epoch
            task_index = torch.tensor([eph+1] * vlm_token.shape[0], device="cpu")
            if (train_mode == "ER_RS" or train_mode == "DER" or train_mode == "DERPP") and reservoir:
                if idx == 0 and eph == 1:
                    print("use reservoir")
                # buffer.add(prefix.detach().cpu(), vlm_token.detach().cpu(), gpt_token.detach().cpu(), gpt_mask.detach().cpu(), logits.detach().cpu(), task_index)
                buffer.add(text_emb.detach().cpu(), z.detach().cpu(), txt_mu.detach().cpu(), txt_alpha.detach().cpu(), txt_beta.detach().cpu(), task_index)

            if train_mode == "DER" or train_mode == "DERPP":
                text_emb, z, txt_mu, txt_alpha, txt_beta = buffer.sample(16)
                text_emb, z, txt_mu, txt_alpha, txt_beta = text_emb.to(device), z.to(device), txt_mu.to(device), txt_alpha.to(device), txt_beta.to(device)

                txt_mu_pred, txt_alpha_pred, txt_beta_pred = BayesCap_Net.txt_BayesCap(text_emb)

                loss_der = alpha * nnf.mse_loss(txt_mu_pred, txt_mu) + nnf.mse_loss(txt_alpha_pred, txt_alpha) + nnf.mse_loss(txt_beta_pred, txt_beta)
                loss_der.backward(retain_graph=True)

                eph_der_loss += loss_der.item()                
                loss_der_list.append(loss_der.item()/alpha)
                

            if train_mode == "DERPP" or train_mode == "ER" or train_mode == "ER_RS":
                text_emb, z, txt_mu, txt_alpha, txt_beta = buffer.sample(16)
                text_emb, z, txt_mu, txt_alpha, txt_beta = text_emb.to(device), z.to(device), txt_mu.to(device), txt_alpha.to(device), txt_beta.to(device)

                txt_mu_pred, txt_alpha_pred, txt_beta_pred = BayesCap_Net.txt_BayesCap(text_emb)

                loss_derpp_i = Cri(txt_mu_pred, txt_alpha_pred, txt_beta_pred, z, T1=0, T2=1)
                loss_derpp_reg = Cri(txt_mu_pred, txt_alpha_pred, txt_beta_pred, text_emb, T1=0, T2=1)

                loss_derpp = beta * (loss_derpp_i + cross_modal_lambda * loss_derpp_reg)
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

        if eph == epochs - 1:
            torch.save(BayesCap_Net.state_dict(), os.path.join(save_dir, f"{output_prefix}-epoch-{eph}.pth"))

        if pretrain_test_loader is not None:
            print(f">>> Evaluating epoch {eph} <<<")
            BayesCap_Net.eval()
            test_loss = 0
            with torch.no_grad():
                for idx, batch in tqdm(enumerate(pretrain_test_loader), total=len(pretrain_test_loader), desc="eval"):
                    vlm_token = batch[2].to(device)
                    index = batch[5]
                    z = agent_z[index].to(device)
                    print("vlm_token", vlm_token.cpu().numpy().shape)
                    text_emb = CLIP_Net.encode_text(vlm_token).to(device, dtype=torch.float32)

                    txt_mu, txt_alpha, txt_beta = BayesCap_Net.txt_BayesCap(text_emb)

                    loss = Cri(txt_mu, txt_alpha, txt_beta, z, T1=T1, T2=T2)
                    test_loss += loss.item()

            test_loss /= len(pretrain_test_loader)
            loss_pretrain_test_list.append(test_loss)
            print(f"Test Loss: {test_loss}")

            np.save(os.path.join(save_dir, f"{output_prefix}_loss_pretrain_test.npy"), np.array(loss_pretrain_test_list))
        
        if fine_tune_test_loader is not None:
            print(f">>> Evaluating epoch {eph} <<<")
            BayesCap_Net.eval()
            test_loss = 0
            with torch.no_grad():
                for idx, batch in tqdm(enumerate(fine_tune_test_loader), total=len(fine_tune_test_loader), desc="eval"):
                    vlm_token = batch[2].to(device)
                    index = batch[5]
                    z = agent_z[index].to(device)

                    text_emb = CLIP_Net.encode_text(vlm_token).to(device, dtype=torch.float32)

                    txt_mu, txt_alpha, txt_beta = BayesCap_Net.txt_BayesCap(text_emb)

                    loss = Cri(txt_mu, txt_alpha, txt_beta, z, T1=T1, T2=T2)
                    test_loss += loss.item()

            test_loss /= len(fine_tune_test_loader)
            loss_fine_tune_test_list.append(test_loss)
            print(f"Test Loss: {test_loss}")

            np.save(os.path.join(save_dir, f"{output_prefix}_loss_fine_tune_test.npy"), np.array(loss_fine_tune_test_list))

    return BayesCap_Net


def pretrain_probvlm(CLIP_Net, train_loader, BayesCap_Net, save_dir, epochs, device, lr=1e-4, output_prefix="probvlm", Cri=TempCombLoss(), T1=1e0, T2=5e-2, cross_modal_lambda_init=1e-4, cross_modal_lambda_final=1.0, start_annealing_epoch=5, train_mode="pretain"):
    
    BayesCap_Net = BayesCap_Net.to(device)
    BayesCap_Net.img_BayesCap.train()
    BayesCap_Net.txt_BayesCap.train()
    CLIP_Net.to(device)
    CLIP_Net.eval()

    print("probvlm lr:", lr)
    optimizer = torch.optim.Adam(
        list(BayesCap_Net.img_BayesCap.parameters()) + list(BayesCap_Net.txt_BayesCap.parameters()), 
        lr=lr
    )
    optim_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    loss_list = []
    loss_i_list = []
    loss_t_list = []
    loss_i2t_list = []
    loss_t2i_list = []

    for eph in range(epochs):
        eph_loss = 0
        eph_loss_i = 0
        eph_loss_t = 0
        eph_loss_i2t = 0
        eph_loss_t2i = 0
        BayesCap_Net.train()

        # Anneal cross_modal_lambda linearly after start_annealing_epoch
        if eph < start_annealing_epoch:
            cross_modal_lambda = cross_modal_lambda_init
        else:
            cross_modal_lambda = cross_modal_lambda_init + (cross_modal_lambda_final - cross_modal_lambda_init) * ((eph - start_annealing_epoch) / (epochs - start_annealing_epoch))

        with tqdm(total=len(train_loader), desc=f"Epoch {eph}") as pbar:
            for (idx, batch) in enumerate(train_loader):
                # xI, xT = batch[0].to(device), batch[2].to(device)
                xI, xT = batch["image"].to(device), batch["vlm_token"].to(device)

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
                eph_loss_i += loss_i.item()
                eph_loss_t += loss_t.item()
                eph_loss_i2t += loss_i2t.item()
                eph_loss_t2i += loss_t2i.item()

                # tqdmのpostfixにloss_iとloss_tを表示
                pbar.set_postfix({'loss_i': f'{loss_i.item():.1f}', 'loss_t': f'{loss_t.item():.1f}'})

                if idx % 500 == 0:
                    torch.save(BayesCap_Net.state_dict(), os.path.join(save_dir, f"{output_prefix}_latest.pth"))
                    np.save(os.path.join(save_dir, f"{output_prefix}_loss.npy"), loss_list)
                optim_scheduler.step()
                pbar.update(1)

        print(f'Epoch {eph} - cross_modal_lambda: {cross_modal_lambda}, Epoch loss: {eph_loss}, loss_i: {eph_loss_i}, loss_t: {eph_loss_t}, loss_i2t: {eph_loss_i2t}, loss_t2i: {eph_loss_t2i}')

        eph_loss /= len(train_loader.dataset)
        eph_loss_i /= len(train_loader.dataset)
        eph_loss_t /= len(train_loader.dataset)
        eph_loss_i2t /= len(train_loader.dataset)
        eph_loss_t2i /= len(train_loader.dataset)

        loss_list.append(eph_loss)
        loss_i_list.append(eph_loss_i)
        loss_t_list.append(eph_loss_t)
        loss_i2t_list.append(eph_loss_i2t)
        loss_t2i_list.append(eph_loss_t2i)

        np.save(os.path.join(save_dir, f"{output_prefix}_loss.npy"), np.array(loss_list))
        np.save(os.path.join(save_dir, f"{output_prefix}_loss_i.npy"), np.array(loss_i_list))
        np.save(os.path.join(save_dir, f"{output_prefix}_loss_t.npy"), np.array(loss_t_list))
        np.save(os.path.join(save_dir, f"{output_prefix}_loss_i2t.npy"), np.array(loss_i2t_list))
        np.save(os.path.join(save_dir, f"{output_prefix}_loss_t2i.npy"), np.array(loss_t2i_list))

        if eph < 5 or eph % 3 == 0 or eph == epochs - 1:
            torch.save(BayesCap_Net.state_dict(), os.path.join(save_dir, f"{output_prefix}-epoch-{eph}.pth"))

    return BayesCap_Net, loss_list[-1], loss_i_list[-1], loss_t_list[-1], loss_i2t_list[-1], loss_t2i_list[-1]