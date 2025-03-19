import copy
import argparse
import numpy as np
import pandas as pd
from utils import *
from update_models import *
import clip
from transformers import GPT2Tokenizer
# from ProbVLM.src.networks import BayesCap_for_CLIP
# from CLIP_prefix_caption.train import ClipCaptionPrefix

class OneAgent(nn.Module):
    def __init__(self, agent_name='A', device='cuda:0', adapter="mlp", td_update_epochs=10, te_update_epochs=10, temperature = 0.7, te_alpha_beta=0.5, td_alpha_beta=0.5, te_train_mode="DERPP", td_train_mode="DERPP", clip_arch = "ViT-B/32", pretrain = False, td_lr=1e-4, te_lr=1e-6):
        """
        Initialize the OneAgent instance with specified network architectures and training parameters.

        Args:
            agent_name (str): Name of the agent.
            device (str): Device to be used for computations (e.g., 'cuda:0').
            adapter (str): Type of adapter to use ('mlp' or 'transformer').
            td_update_epochs (int): Number of update epochs for the text decoder.
            te_update_epochs (int): Number of update epochs for the text encoder.
            temperature (float): Temperature parameter for text generation.
            te_alpha_beta (float): Alpha-beta parameter for text encoder training.
            td_alpha_beta (float): Alpha-beta parameter for text decoder training.
            te_train_mode (str): Training mode for the text encoder.
            td_train_mode (str): Training mode for the text decoder.
            clip_arch (str): CLIP architecture name.
            pretrain (bool): Flag to indicate usage of pre-trained weights.
            td_lr (float): Learning rate for text decoder updates.
            te_lr (float): Learning rate for text encoder updates.
        """
        super().__init__()
        from ProbVLM.src.networks import BayesCap_for_CLIP # import ProbVLM network
        from CLIP_prefix_caption.train import ClipCaptionModel # import ClipCaption network
        self.agent_name = agent_name
        self.device = device
        self.ProbVLM_Net = BayesCap_for_CLIP(inp_dim=512, out_dim=512, hid_dim=256, num_layers=3, p_drop=0.01,)
        self.ProbVLM_Net.eval()
        self.CLIP_Net, self.preprocess = clip.load(clip_arch, device = self.device)
        self.CLIP_Net = self.CLIP_Net.float()
        self.CLIP_Net.eval()
        self.adapter = adapter
        if self.adapter == "mlp":
            self.prefix_length = 10
        elif self.adapter == "transformer":
            self.prefix_length = 40
        self.prefix_length_clip = 40
        self.prefix_dim = 512
        self.num_layers = 8
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        
        self.ClipCap = ClipCaptionModel(self.prefix_length, self.prefix_length_clip, self.prefix_dim, self.num_layers, self.adapter)
        
        self.GGL = GenGaussLoss(reduction='batchsum')
        self.GGLogLikelihood = GenGaussLogLikelihood(reduction='batchsum')
        self.clipcap_loss_list = []
        self.clipcap_loss_list_test = []
        self.probvlm_loss_list = []
        self.probvlm_loss_list_test = []
        self.td_buffer = None
        self.te_buffer = None

        self.td_update_epochs = td_update_epochs
        self.te_update_epochs = te_update_epochs
        self.temperature = temperature
        self.te_alpha_beta = te_alpha_beta
        self.td_alpha_beta = td_alpha_beta
        self.te_train_mode = te_train_mode
        self.td_train_mode = td_train_mode
        self.td_lr = td_lr
        self.te_lr = te_lr
    
    def initialize_sign(self):
        """
        Initialize signs for the MHCG process.

        This function iterates over the fixed MHNG dataloader, encodes each image,
        generates text using the text decoder, tokenizes the generated text, and
        updates the dataset with the new caption and corresponding tokens.
        """
        print("Agent", self.agent_name, " initialize sign")
        with torch.no_grad():
            i = 0
            for batch in tqdm(self.dataloader_MHNG_fix, desc="initialize sign"):
                img = batch["image"].to(self.device)
                mu, _, _, _ = self.image_encoder(img)
                texts = self.text_decoder(mu)
                for b, text in enumerate(texts):
                    caption = text
                    vlm_token = tokenize(text)[0]
                    gpt_token = torch.tensor(self.tokenizer.encode(text))
                    
                    datatype = "coco"
                    self.dataloader_MHNG_fix.dataset.dataset[i]["caption"] = caption
                    self.dataloader_MHNG_fix.dataset.dataset[i]["vlm_token"] = vlm_token
                    self.dataloader_MHNG_fix.dataset.dataset[i]["gpt_token"] = gpt_token
                    self.dataloader_MHNG_fix.dataset.dataset[i]["dataset"] = datatype
                    i += 1
    
    def save_sign(self, status):
        """
        Save the current signs (captions) to a CSV file.

        Args:
            status: A status sentence used to store the captions in the DataFrame.
        """   
        captions = [self.dataloader_MHNG_fix.dataset.dataset[i]["caption"] for i in range(len(self.dataloader_MHNG_fix.dataset))]
        self.df_sign[status] = captions
        self.df_sign.to_csv(f"{self.save_dir}/agent_{self.agent_name}_sign.csv")
    
    def save_proposed_w(self, proposed_w, status):
        """
        Save the proposed captions to a CSV file.

        Args:
            proposed_w: Proposed tokenized captions.
            status: A status sentence used to store the captions in the DataFrame.
        """
        proposed_captions = [tokenizer_decode(proposed_w[i]) for i in range(len(proposed_w))]
        self.df_proposed_w[status] = proposed_captions
        # save proposed caption in csv
        self.df_proposed_w.to_csv(f"{self.save_dir}/agent_{self.agent_name}_proposed_w.csv")

    def load_pretrain(self, probvlm_path, clipcap_path, strict_clipcap=True):
        """
        Load pre-trained weights for the ProbVLM and ClipCap networks.

        Args:
            probvlm_path (str): File path for the ProbVLM pre-trained model.
            clipcap_path (str): File path for the ClipCap pre-trained model.
            strict_clipcap (bool): Whether to enforce strict matching when loading ClipCap weights.
        """
        self.probvlm_path = probvlm_path
        self.clipcap_path = clipcap_path
        self.ProbVLM_Net.load_state_dict(torch.load(probvlm_path))
        clipcap_weights = torch.load(clipcap_path, map_location="cpu")
        self.ClipCap.load_state_dict(clipcap_weights, strict=strict_clipcap)
        self.ClipCap.to(self.device)
    
    def reset_pretrain(self):
        """
        Reset the ProbVLM and ClipCap networks to the pre-trained weights.
        """
        self.ProbVLM_Net.load_state_dict(torch.load(self.probvlm_path))
        clipcap_weights = torch.load(self.clipcap_path, map_location="cpu")
        self.ClipCap.load_state_dict(clipcap_weights)
        self.ClipCap.to(self.device)

    def lora_setting(self, r=2, alpha=32, dropout=0.1, clipcap=True, probvlm=True):
        """
        Configure LoRA settings for ProbVLM and ClipCap networks by freezing parameters
        and applying LoRA to specific layers.

        Args:
            r (int): Rank for LoRA.
            alpha (int): Scaling factor for LoRA.
            dropout (float): Dropout rate for LoRA.
            clipcap (bool): Whether to apply LoRA to the ClipCap network.
            probvlm (bool): Whether to apply LoRA to the ProbVLM network.
        """
        for name, param in self.ProbVLM_Net.named_parameters():
            param.requires_grad = False
        if clipcap:
            for name, param in self.ClipCap.named_parameters():
                param.requires_grad = False
        elif clipcap == False:
            for name, param in self.ClipCap.gpt.named_parameters():
                param.requires_grad = False

        from peft import get_peft_model, LoraConfig, TaskType
        if probvlm:
            apply_lora_to_layer(self.ProbVLM_Net.txt_BayesCap._modules['mod'], "5", r=r, alpha=alpha, dropout=dropout)
            apply_lora_to_layer(self.ProbVLM_Net.txt_BayesCap._modules['block_mu'], "2", r=r, alpha=alpha, dropout=dropout)
            apply_lora_to_layer(self.ProbVLM_Net.txt_BayesCap._modules['block_alpha'], "2", r=r, alpha=alpha, dropout=dropout)
            apply_lora_to_layer(self.ProbVLM_Net.txt_BayesCap._modules['block_beta'], "2", r=r, alpha=alpha, dropout=dropout)
    
        if clipcap:
            # print the parameters of the network
            print("ClipCap")
            apply_lora_to_layer(self.ClipCap.clip_project._modules['model'], "0", r=r, alpha=alpha, dropout=dropout)
            apply_lora_to_layer(self.ClipCap.clip_project._modules['model'], "2", r=r, alpha=alpha, dropout=dropout)
            target_modules = ["attn.c_attn", "attn.c_proj", ]
            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=target_modules)
            self.ClipCap.gpt = get_peft_model(self.ClipCap.gpt, peft_config)
            # count trainable parameters
            total_params = sum(p.numel() for p in self.ClipCap.parameters())
            print(f'{total_params:,} total parameters.')
            total_trainable_params = sum(p.numel() for p in self.ClipCap.parameters() if p.requires_grad)
            print(f'{total_trainable_params:,} training parameters.')
        self = self.to(self.device)

    def communication_field_setup(self, dataloader_MHNG_fix, dataloader_MHNG_shuffle, MH_iter, EM_iter, mode="MHNG"):
        """
        Set up the communication field by initializing dataloaders, epochs, and temperature schedule.

        Args:
            dataloader_MHNG_fix: Fixed dataset dataloader for MHNG.
            dataloader_MHNG_shuffle: Shuffled dataset dataloader for MHNG.
            MH_iter (int): Number of iterations/epochs for the MH process.
            EM_iter (int): Number of iterations/epochs for the EM process.
            mode (str): Communication mode (e.g., "MHNG", "all_accept", "no_communication").
        """
        self.dataloader_MHNG_fix = dataloader_MHNG_fix
        self.dataloader_MHNG_shuffle = dataloader_MHNG_shuffle
        self.MH_epochs = MH_iter
        self.EM_epochs = EM_iter
        self.temperature_annealing_values = linear_schedule(0.7, 0.001, self.MH_epochs, self.MH_epochs)
        self.mode = mode
        self.df_proposed_w = pd.DataFrame()
        self.df_sign = pd.DataFrame()

    def image_encoder(self, o): 
        """
        Encode an image and extract features.

        Args:
            o: Input image tensor.

        Returns:
            tuple: (mu_img, alpha_img, sigma_img, z) where:
                mu_img: Mean image features.
                alpha_img: Alpha values from the image encoder.
                sigma_img: Sigma values from the image encoder.
                z: Raw image features from CLIP.
        """
        with torch.no_grad():
            z = self.CLIP_Net.encode_image(o)
            mu_img, alpha_img, sigma_img = self.ProbVLM_Net.img_BayesCap(z)
        return mu_img, alpha_img, sigma_img, z
    
    def text_encoder(self, w, return_z = False):
        """
        Encode tokenized text and extract features.

        Args:
            w: Tokenized text input (by CLIP tokenizer).
            return_z (bool): If True, also return the raw text features from CLIP.

        Returns:
            tuple: If return_z is True, returns (mu_cap, alpha_cap, sigma_cap, z);
                   otherwise, returns (mu_cap, alpha_cap, sigma_cap).
        """
        with torch.no_grad():
            z = self.CLIP_Net.encode_text(w)
            mu_cap, alpha_cap, sigma_cap = self.ProbVLM_Net.txt_BayesCap(z)
        if return_z:
            return mu_cap, alpha_cap, sigma_cap, z
        else:
            return mu_cap, alpha_cap, sigma_cap
        
    def text_decoder(self, z, temperature = "None"):
        """
        Generate text (caption) from image features.

        Args:
            z: Input image features tensor.
            temperature: Temperature parameter for generation; if "None", uses self.temperature.

        Returns:
            list: Generated text (caption) as a list of strings.
        """
        if temperature == "None":
            temperature = self.temperature
        with torch.no_grad():
            prefix_embeds = self.ClipCap.clip_project(z.float()).reshape(z.shape[0], self.prefix_length, -1)
            t = generate_batch(self.ClipCap, self.tokenizer, embed=prefix_embeds, temperature=temperature)
        return t

    def text_decoder_with_token(self, z, temperature = "None", caption = None):
        """
        Generate text (caption) using provided caption tokens for additional context.

        Args:
            z: Input image features tensor.
            temperature: Temperature parameter for generation; if "None", uses self.temperature.
            caption (str): A caption string used for tokenizing and appending to the generated sequence.

        Returns:
            list: Generated text (caption) as a list of strings.
        """
        tokens = self.tokenizer.encode(caption)
        prefix_token_embed = self.ClipCap.gpt.transformer.wte(torch.tensor(tokens).to(self.device)).unsqueeze(0)
        # copy the prefix_token_embed to the length of the batch
        prefix_token_embeds = prefix_token_embed.repeat(z.shape[0], 1, 1)
        if temperature == "None":
            temperature = self.temperature
        with torch.no_grad():
            prefix_embeds = self.ClipCap.clip_project(z.float()).reshape(z.shape[0], self.prefix_length, -1)

            prefix_embeds = torch.cat([prefix_embeds, prefix_token_embeds], dim=1)
            t = generate_batch(self.ClipCap, self.tokenizer, embed=prefix_embeds, temperature=temperature)
        return t
    
    def perception(self): 
        """
        Perform the perception step by encoding images from the fixed MHNG dataloader.

        This function collects the image features for all images and stores them in self.z.
        """
        print("Agent", self.agent_name, " perception")
        z = []
        for batch in tqdm(self.dataloader_MHNG_fix):
            o = batch["image"].to(self.device)
            mu_img, alpha_img, sigma_img, _ = self.image_encoder(o)
            z.append(mu_img)
        self.z = torch.cat(z, dim=0)
    
    def propose(self, return_caption=False, mh_epoch=-1):
        """
        Propose new captions by decoding from image features.

        Args:
            return_caption (bool): If True, returns both tokenized proposals and caption strings.
            mh_epoch (int): Index used for temperature annealing values.

        Returns:
            Depending on return_caption, either:
                - Tokenized proposals (torch.Tensor), or
                - Tuple (proposed_w, proposed_caption) with tokenized and decoded captions.
        """

        print("Agent", self.agent_name)
        with torch.no_grad():
            proposed_w = []
            proposed_caption = []
            max_index = len(self.dataloader_MHNG_fix.dataset)
            batch_size = 100
            indices = torch.arange(max_index)
            for i in range(0, max_index, batch_size):
                index = indices[i:i + batch_size]
                z = self.z[index]
                w = self.text_decoder(z, temperature=self.temperature_annealing_values[mh_epoch])
                proposed_caption.extend(w)
                proposed_w.append(tokenize(w))
            proposed_w = torch.cat(proposed_w, dim=0).to(self.device)
            
        if return_caption:
            return proposed_w, proposed_caption
        else:
            return proposed_w

    def judge(self, proposed_w):
        """
        Evaluate and judge the proposed captions based on acceptance probabilities.

        This function compares the likelihoods of the proposed captions against the current captions,
        calculates an acceptance rate, and updates the dataset accordingly.

        Args:
            proposed_w: Proposed tokenized captions.

        Returns:
            float: Acceptance rate (ratio of updated captions).
        """
        with torch.no_grad():
            
            vlm_token = [self.dataloader_MHNG_fix.dataset.dataset[i]["vlm_token"] for i in range(len(self.dataloader_MHNG_fix.dataset))]
            before_self_w = torch.cat(vlm_token, dim=0).reshape(len(self.dataloader_MHNG_fix.dataset), -1).clone().to(self.device)
            updated_index_list = []

            max_index = len(self.dataloader_MHNG_fix.dataset)
            batch_size = 100
            indices = torch.arange(max_index)

            for i in range(0, max_index, batch_size):
                index = indices[i:i + batch_size]
                mu_Sp, alpha_Sp, beta_Sp = self.text_encoder(proposed_w[index])
                mu_Li, alpha_Li, beta_Li = self.text_encoder(before_self_w[index])

                p_li = self.GGLogLikelihood(mu_Li, alpha_Li, beta_Li, self.z[index])
                p_sp = self.GGLogLikelihood(mu_Sp, alpha_Sp, beta_Sp, self.z[index])

                # calculate acceptance rate r
                
                if self.mode == "MHNG":
                    r = np.exp(np.where((p_sp-p_li).detach().cpu().numpy()<0,(p_sp-p_li).detach().cpu().numpy(),0))
                elif self.mode == "all_accept":
                    r = np.ones(len(p_sp))
                elif self.mode == "no_communication":
                    r = np.zeros(len(p_sp))
                u = np.random.rand(len(r),)            

                # update w
                global_update_index = index[np.where(u < r)[0]]
                updated_index_list = updated_index_list + list(global_update_index)
                j = 0
                for index in global_update_index:
                    self.dataloader_MHNG_fix.dataset.dataset[index]["caption"] = tokenizer_decode(proposed_w[index])
                    self.dataloader_MHNG_fix.dataset.dataset[index]["vlm_token"] = proposed_w[index].cpu()
                    self.dataloader_MHNG_fix.dataset.dataset[index]["gpt_token"] = torch.tensor(self.tokenizer.encode(self.dataloader_MHNG_fix.dataset.dataset[index]["caption"]))
                    j += 1
                
            print("acceptance rate:", len(updated_index_list)/len(self.dataloader_MHNG_fix.dataset))
            acceptance_rate = len(updated_index_list)/len(self.dataloader_MHNG_fix.dataset)
            return acceptance_rate

    def initialize_td_buffer(self, dataloader, buffer_size):
        """
        Initialize the text decoder buffer with data from a dataloader.

        Args:
            dataloader: Data loader providing image and caption data.
            buffer_size (int): Maximum number of examples to store in the buffer.
        """

        print("Agent", self.agent_name, " initialize buffer")

        self.td_buffer = ClipCapBuffer(buffer_size=buffer_size)
        mse_loss = nn.MSELoss()
        mu_z_mse = 0
        with torch.no_grad():
            for batch in dataloader:
                img = batch["image"]
                vlm_token = batch["vlm_token"]
                gpt_token = batch["gpt_token"]
                gpt_mask = batch["gpt_mask"]
                img, gpt_token, gpt_mask = img.to(self.device), gpt_token.to(self.device), gpt_mask.to(self.device)
                mu_img, alpha_img, sigma_img, z = self.image_encoder(img)
                mu_z_mse += mse_loss(mu_img, z)
                img = img.cpu()
                outputs = self.ClipCap(gpt_token, z, gpt_mask)
                logits = outputs.logits[:, dataloader.dataset.prefix_length - 1: -1]
                task_index = torch.tensor([0] * img.shape[0], device="cpu")
                self.td_buffer.add(z.detach().cpu(), vlm_token.cpu(), gpt_token.cpu(), gpt_mask.cpu(), logits.detach().cpu(), task_index)
                if self.td_buffer.num_seen_examples > buffer_size:
                    break
    
    def initialize_te_buffer(self, dataloader, buffer_size):
        """
        Initialize the text encoder buffer with data from a dataloader.

        Args:
            dataloader: Data loader providing image and caption data.
            buffer_size (int): Maximum number of examples to store in the buffer.
        """

        self.ProbVLM_Net.txt_BayesCap.eval()
        self.ProbVLM_Net.img_BayesCap.eval()
        print("Agent", self.agent_name, " initialize buffer")
        self.te_buffer = ProbVLMBuffer(buffer_size=buffer_size)
        with torch.no_grad():
            for batch in dataloader:
                img = batch["image"]
                vlm_token = batch["vlm_token"]
                img, vlm_token = img.to(self.device), vlm_token.to(self.device)
                mu_cap, alpha_cap, sigma_cap, text_emb = self.text_encoder(vlm_token, return_z=True)
                mu_img, _, _, _ = self.image_encoder(img)

                self.te_buffer.add(text_emb.detach().cpu(), mu_img.detach().cpu(), mu_cap.detach().cpu(), alpha_cap.detach().cpu(), sigma_cap.detach().cpu(), torch.tensor([0] * img.shape[0], device="cpu"))
                if self.te_buffer.num_seen_examples > buffer_size:
                    break

    def update_text_decoder(self, em_epoch):
        """
        Update the text decoder network using the current image features and dataloader.

        Args:
            em_epoch: Current epoch or iteration index used for output naming.
        """

        print("Agent", self.agent_name, " update text decoder")
        self.ClipCap.train()
        updated_clipcap = update_clipcap_derpp(self.z, self.CLIP_Net, self.ClipCap, self.tokenizer, self.dataloader_MHNG_shuffle, self.dataloader_MHNG_fix, self.save_dir, epochs = self.td_update_epochs, lr=self.td_lr, train_mode=self.td_train_mode, device=self.device, buffer=self.td_buffer, output_prefix="clipcap_"+self.agent_name+f"_{em_epoch}", save_every=5, alpha=self.td_alpha_beta, beta=self.td_alpha_beta, reserovoir=False)
        self.ClipCap = updated_clipcap.eval()

    def update_text_decoder_distillation(self, speaker_agent, em_epoch):
        """
        Update the text decoder network via distillation.

        Args:
            speaker_agent: Another agent whose output is used for distillation.
            em_epoch: Current epoch or iteration index used for output naming.
        """
        print("Agent", self.agent_name, " update text decoder distillation")
        self.ClipCap.train()
        updated_clipcap = update_distillation_merge(self, speaker_agent, self.dataloader_MHNG_shuffle, self.dataloader_MHNG_fix, self.save_dir, epochs = self.td_update_epochs, lr=self.td_lr, train_mode=self.td_train_mode, device=self.device, output_prefix="clipcap_"+self.agent_name+f"_{em_epoch}", save_every=5, buffer=self.td_buffer, alpha=self.td_alpha_beta, beta=self.td_alpha_beta, reservoir=False)
        self.ClipCap = updated_clipcap.eval()
    
    def update_text_encoder(self, em_epoch):
        """
        Update the text encoder network using the current image features and dataloader.

        Args:
            em_epoch: Current epoch or iteration index used for output naming.
        """

        print("Agent", self.agent_name, " update text encoder")
        self.ProbVLM_Net.train()
        updated_probvlm = update_probvlm_derpp(self.z, self.CLIP_Net, self.ProbVLM_Net, self.dataloader_MHNG_shuffle, self.save_dir, epochs = self.te_update_epochs, lr=self.te_lr, device=self.device, output_prefix="probvlm_"+self.agent_name+f"_{em_epoch}", save_every=5, buffer=self.te_buffer, train_mode = self.te_train_mode, cross_modal_lambda=1, alpha=self.te_alpha_beta, beta=self.te_alpha_beta, reservoir=False)
        self.ProbVLM_Net = updated_probvlm.eval()
    
    def calculate_p_z_w(self, image, caption):
        """
        Calculate the likelihood (negative loss) of the image and caption pairing.

        This function tokenizes the caption, encodes the image and text, and computes the negative
        generalized Gaussian loss between the image features and the text features.

        Args:
            image: Input image tensor.
            caption (str): Caption corresponding to the image.

        Returns:
            torch.Tensor: Computed likelihood (p_z_w).
        """
        caption = tokenize(caption).to(self.device)
        mu_img, alpha_img, sigma_img, z = self.image_encoder(image)
        mu_cap, alpha_cap, sigma_cap = self.text_encoder(caption)
        p_z_w = -self.GGL(mu_cap, alpha_cap, sigma_cap, mu_img)
        return p_z_w
