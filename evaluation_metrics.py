from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
import sacrebleu
from utils import *
import clip
import torch
import evaluate
from torchmetrics.text import SacreBLEUScore
from nltk.translate.meteor_score import meteor_score
from torch.nn import functional as nnf
from ProbVLM.src.losses import *
from bert_score import score as bert_score
import tqdm

# Reference based metrics
# BLEU, METEOR, BERT_score
def calculate_bleu_score(generated, references):
    sacre_bleu = SacreBLEUScore(lowercase=True)
    bleu_score = sacre_bleu(generated, references).item() * 100
    return bleu_score

def calculate_bleu_score_coco(generated, image_ids, dataset_mode):
    if dataset_mode == "train":
        with open("dataset/dataset_cache/coco_train_dataset_10000_imageid.pkl", "rb") as f:
            coco_dataset = pickle.load(f)
    else:
        with open("dataset/dataset_cache/coco_test_dataset_5000_imageid.pkl", "rb") as f:
            coco_dataset = pickle.load(f)
    references = []
    for image_id in image_ids:
        ref = []
        for ann in coco_dataset.coco.imgToAnns[image_id]:
            ref.append(ann['caption'])
        references.append(ref)
    bleu_score = calculate_bleu_score(generated, references)
    return bleu_score

def calculate_meteor_score(generated, references):
    flattened_references = [ref for ref_list in references for ref in ref_list]
    flattened_hypotheses = [hyp for hyp, ref_list in zip(generated, references) for _ in ref_list]

    all_refs_split = [ref.split() for ref in flattened_references]
    all_hyps_split = [hyp.split() for hyp in flattened_hypotheses]
    total_score = 0
    for ref, hyp in zip(all_refs_split, all_hyps_split):
        score = meteor_score([ref], hyp, alpha=0.9, beta=3, gamma=0.5)
        total_score += score

    # 全体の平均METEORスコアを計算
    coco_metereor_score = total_score / len(all_hyps_split) 
    return coco_metereor_score     

def calculate_bert_score(generated, references):
    bscore = bert_score(generated, references, lang="en")
    return bscore[2].mean().item()

# Reference free metrics
# CLIP_score, p(zA, zB | c)

def calculate_clip_score(image_embeddings, text_embeddings):
    clip_score = nnf.cosine_similarity(image_embeddings, text_embeddings).cpu().detach().numpy()
    return np.mean(clip_score)

def calculate_p_zA_zB_c(z_A, z_B, mu_A, alpha_A, beta_A, mu_B, alpha_B, beta_B):
    
    # calculate p(zA | c)
    zA_c = -GenGaussLoss(mu_A, alpha_A, beta_A, z_A)

    # calculate p(zB | c)
    zB_c = -GenGaussLoss(mu_B, alpha_B, beta_B, z_B)

    # calculate p(zA, zB | c)
    zA_zB_c = zA_c + zB_c

    return zA_zB_c, zA_c, zB_c

class Evaluater():
    def __init__(self, agent, dataset, dataset_mode, dataset_name, device):
        self.agent = agent
        self.dataset = dataset
        self.dataset_mode = dataset_mode
        self.dataset_name = dataset_name # "coco" or "conceptual"
        self.device = device

        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=64, shuffle=False, num_workers=1)

    def generate(self, gen_mode, temperature=0.6):
        refs = []
        gens = []
        image_features = []
        text_features = []
        image_index = 0
        for batch in tqdm.tqdm(self.dataloader):
            img = batch[0].to(self.device)
            caption = batch[1]

            prefix = self.agent.CLIP_Net.encode_image(img).to(self.device, dtype=torch.float32)
            prefix_embeds = self.agent.clip_project(prefix).reshape(prefix.shape[0], self.dataset.prefix_length, -1)

            image_features.append(prefix.cpu().detach())

            if gen_mode == "beam":
                for idx, prefix_embed in enumerate(prefix_embeds):
                    prefix_embed = prefix_embed.unsqueeze(0)
                    generated_text = generate_beam(self.agent, self.agent.tokenizer, embed=prefix_embed, temperature=temperature)[0]
                    gens.append(generated_text)
                    text_tokens = tokenize(generated_text).to(self.device)
                    text_embed = self.agent.CLIP_Net.encode_text(text_tokens).to(self.device, dtype=torch.float32)
                    text_features.append(text_embed.cpu().detach())
                    print(f"Image {image_index}:", generated_text)
                    image_index += 1
            if gen_mode == "sample":
                generated_texts = generate_batch(self.agent, self.agent.tokenizer, embed=prefix_embeds, temperature=temperature)
                gens.extend(generated_texts)
                for text in generated_texts:
                    text_tokens = tokenize(text).to(self.device)
                    text_embed = self.agent.CLIP_Net.encode_text(text_tokens).to(self.device, dtype=torch.float32)
                    text_features.append(text_embed.cpu().detach())
                for text in generated_texts:
                    print(f"Image {image_index}:", text)
                    image_index += 1

            refs.extend(list(caption))
        
        image_features = torch.cat(image_features, dim=0)
        text_features = torch.cat(text_features, dim=0)

        self.generated_captions = gens
        self.references = refs
        self.image_features = image_features
        self.text_features = text_features

    def get_image_ids(self):
        image_ids = []
        for batch in self.dataloader:
            image_id = batch[-1]
            image_ids.extend([id.item() for id in image_id])
        self.image_ids = image_ids

    def evaluate(self):
        if self.dataset_name == "coco":
            self.get_image_ids()
            bleu_score = calculate_bleu_score_coco(self.generated_captions, self.image_ids, self.dataset_mode)
        else:
            bleu_score = calculate_bleu_score(self.generated_captions, self.references)
        meteor_score = calculate_meteor_score(self.generated_captions, self.references)
        bert_score = calculate_bert_score(self.generated_captions, self.references)
        clip_score = calculate_clip_score(self.image_features, self.text_features)

        return bleu_score, meteor_score, bert_score, clip_score