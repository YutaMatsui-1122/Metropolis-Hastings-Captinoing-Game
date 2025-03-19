import os
import argparse
import torch
import json
import evaluation
import numpy as np

from models.clip import clip

from evaluation.pac_score import PACScore, RefPACScore, clip_score , bert_score
from models import open_clip

_MODELS = {
    "ViT-B/32": "pacscore/checkpoints/clip_ViT-B-32.pth",
    "open_clip_ViT-L/14": "pacscore/checkpoints/openClip_ViT-L-14.pth"
}


def compute_scores(model, preprocess, image_ids, candidates, references, args):
    gen = {}
    gts = {}
    ims_cs = list()
    gen_cs = list()
    gts_cs = list()
    all_scores = dict()
    model.eval()


    for i, (im_i, gts_i, gen_i) in enumerate(zip(image_ids, references, candidates)):
        gen['%d' % (i)] = [gen_i, ]
        gts['%d' % (i)] = gts_i

        ims_cs.append(im_i)
        gen_cs.append(gen_i)
        gts_cs.append(gts_i)

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)

    all_scores_metrics = evaluation.get_all_metrics(gts, gen)

    bert_scores = bert_score(gen_cs, gts_cs)
    all_scores['BERT-S'] = bert_scores['f1']

    for k, v in all_scores_metrics.items():
        if k == 'BLEU':
            all_scores['BLEU-1'] = v[0]
            all_scores['BLEU-4'] = v[-1]
        else:
            all_scores[k] = v

    # PAC-S
    _, pac_scores, candidate_feats, len_candidates = PACScore(
        model, preprocess, ims_cs, gen_cs, device, w=2.0)
    all_scores['PAC-S'] = np.mean(pac_scores)
    num = 10
    # 値が小さい順にインデックスを取得
    min_index = np.argsort(pac_scores)
    max_index = np.argsort(pac_scores)[::-1]
    print("min_index",min_index[:num])
    print("min pac_scores",pac_scores[min_index[:num]])
    print("max_index",max_index[:num])
    print("max pac_scores",pac_scores[max_index[:num]])

    # RefPAC-S
    if args.compute_refpac:
        _, per_instance_text_text = RefPACScore(
            model, gts_cs, candidate_feats, device, torch.tensor(len_candidates))
        refpac_scores = 2 * pac_scores * per_instance_text_text / \
            (pac_scores + per_instance_text_text)
        all_scores['RefPAC-S'] = np.mean(refpac_scores)

    # CLIP-S
    # def clip_score(model, transform, images, captions, device, w=2.5):
    # clip_scores = clip_score(model, preprocess, ims_cs, gen_cs, device, w=model.logit_scale.item())
    clip_s, clip_scores = clip_score(model, preprocess, ims_cs, gen_cs, device, w=2.5)
    all_scores['CLIP-S'] = clip_s
    # print(clip_scores[:3])
    # 値が小さい順にインデックスを取得
    min_index = np.argsort(clip_scores)
    max_index = np.argsort(clip_scores)[::-1]
    return all_scores


if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description='PAC-S evaluation')
    parser.add_argument('--clip_model', type=str, default='ViT-B/32',
                        choices=['ViT-B/32', 'open_clip_ViT-L/14'])
    # parser.add_argument('--image_dir', type=str, default='../dataset/coco/val2014')
    parser.add_argument('--candidates_json', type=str,
                        default='example/good_captions.json')
    # parser.add_argument('--references_json', type=str, default='../exp_eval/refs/coco_refs.json')
    parser.add_argument('--dataset', type=str, default='coco', choices=['coco', 'cc3m', 'nocaps'])
    parser.add_argument('--dataset_mode', type=str, default='eval', choices=['train', 'eval'])
    parser.add_argument('--compute_refpac', action='store_true')
    parser.add_argument('--device', type=str, default='cuda:3')

    args = parser.parse_args()

    if args.dataset == "coco":
        if args.dataset_mode == "train":
            image_dir = "dataset/coco/train2014"
            references_json = "exp_eval/refs/coco_train_refs.json"
        elif args.dataset_mode == "eval":
            image_dir = "dataset/coco/val2014"
            references_json = "exp_eval/refs/coco_refs.json"
    elif args.dataset == "cc3m":
        if args.dataset_mode == "train":
            image_dir = "DownloadConceptualCaptions/training"
            references_json = "exp_eval/refs/cc3m_train_refs.json"
        elif args.dataset_mode == "eval":
            image_dir = "DownloadConceptualCaptions/validation_for_comupte_metrics"
            references_json = "exp_eval/refs/cc3m_refs.json"
    elif args.dataset == "nocaps":
        image_dir = "dataset/nocaps/validation"
        references_json = "exp_eval/refs/nocaps_refs.json"
    else:
        print("Error")
        exit()

    # device = "cuda:3" if torch.cuda.is_available() else "cpu"
    device = args.device if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # image_ids = [img_id for img_id in os.listdir(image_dir)]


    with open(args.candidates_json) as f:
        candidates = json.load(f)

    with open(references_json) as f:
        references = json.load(f)

    reference_keys = list(references.keys())
    references = list(references.values())
    image_ids = [os.path.join(image_dir, img_id) for img_id in reference_keys]
    candidates = [candidates[cid] for cid in reference_keys]
    # candidates_keys = list(candidates.keys())


    # if "." in image_ids[0]:
    #     references = [references[cid.split('.')[0]] for cid in image_ids if cid.split('.')[0] in references.keys()]
    # else:
    #     references = [references[cid] for cid in image_ids if cid in references]

    # if "." in image_ids[0]:
    #     candidates = [candidates[cid.split('.')[0]] for cid in image_ids if cid.split('.')[0] in candidates.keys()]
    # else:
    #     candidates = [candidates[cid] for cid in image_ids if cid in candidates]
    
    # image_ids = [os.path.join(image_dir, img_id) for img_id in reference_keys]
    print("Image ids:", image_ids[:3])
    print("Candidates:", candidates[:3])
    print("References:", references[:3])

    print("Number of images:", len(image_ids))
    print("Number of candidates:", len(candidates))
    print("Number of references:", len(references))

    if args.clip_model.startswith('open_clip'):
        print("Using Open CLIP Model: " + args.clip_model)
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-L-14', pretrained='laion2b_s32b_b82k')
    else:
        print("Using CLIP Model: " + args.clip_model)
        model, preprocess = clip.load(args.clip_model, device=device)

    model = model.to(device)
    model = model.float()

    # checkpoint = torch.load(_MODELS[args.clip_model])
    checkpoint = torch.load(_MODELS[args.clip_model], map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    all_scores = compute_scores(
        model, preprocess, image_ids, candidates, references, args)
    
    print(all_scores)

    # Get the output file name by replacing .json with _score.txt
    output_dir = os.path.dirname(args.candidates_json)
    output_dir = os.path.join(output_dir, 'scores')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir, os.path.basename(args.candidates_json).replace('.json', '_bert_score.txt'))

    # Save the scores to a text file
    with open(output_file, 'w') as f:
        for k, v in all_scores.items():
            f.write('%s: %.4f\n' % (k, v))

    print(f"Scores saved to {output_file}")