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

    # RefPAC-S
    if args.compute_refpac:
        _, per_instance_text_text = RefPACScore(
            model, gts_cs, candidate_feats, device, torch.tensor(len_candidates))
        refpac_scores = 2 * pac_scores * per_instance_text_text / \
            (pac_scores + per_instance_text_text)
        all_scores['RefPAC-S'] = np.mean(refpac_scores)

    # CLIP-S
    clip_s, clip_scores = clip_score(model, preprocess, ims_cs, gen_cs, device, w=2.5)
    all_scores['CLIP-S'] = clip_s
    # print(clip_scores[:3])
    # 値が小さい順にインデックスを取得
    return all_scores


if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description='PAC-S evaluation')
    parser.add_argument('--clip_model', type=str, default='open_clip_ViT-L/14',
                        choices=['ViT-B/32', 'open_clip_ViT-L/14'])
    # parser.add_argument('--image_dir', type=str, default='../dataset/coco/val2014')
    parser.add_argument('--candidates_json', type=str,
                        default='example/good_captions.json')
    # parser.add_argument('--references_json', type=str, default='../exp_eval/refs/coco_refs.json')
    parser.add_argument('--dataset', type=str, default='coco_all', choices=['coco_a', 'coco_b', 'coco_all'])
    parser.add_argument('--dataset_mode', type=str, default='eval', choices=['train', 'eval'])
    parser.add_argument('--compute_refpac', action='store_true')
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--dataset_prefix', type=str, default='coco_2017_common_person_only')

    args = parser.parse_args()

    if args.dataset_mode == "train":
        image_dir = "dataset/coco/train2017"
        if args.dataset == "coco_a":
            # references_json = "exp_eval/refs/coco_a_train_refs.json"
            references_json = f"exp_eval/refs/{args.dataset_prefix}_train_split_dataset_a_refs.json"
        elif args.dataset == "coco_b":
            # references_json = "exp_eval/refs/coco_b_train_refs.json"
            references_json = f"exp_eval/refs/{args.dataset_prefix}_train_split_dataset_b_refs.json"
        elif args.dataset == "coco_all":
            # references_json = "exp_eval/refs/coco_whole_train_refs.json"
            references_json = f"exp_eval/refs/{args.dataset_prefix}_train_split_dataset_all_refs.json"

    elif args.dataset_mode == "eval":
        image_dir = "dataset/coco/val2017"
        if args.dataset == "coco_a":
            # references_json = "exp_eval/refs/coco_a_val_refs.json"
            references_json = f"exp_eval/refs/{args.dataset_prefix}_val_split_dataset_a_refs.json"
        elif args.dataset == "coco_b":
            # references_json = "exp_eval/refs/coco_b_val_refs.json"
            references_json = f"exp_eval/refs/{args.dataset_prefix}_val_split_dataset_b_refs.json"
        elif args.dataset == "coco_all":
            # references_json = "exp_eval/refs/coco_whole_val_refs.json"
            references_json = f"exp_eval/refs/{args.dataset_prefix}_val_split_dataset_all_refs.json"

    else:
        print("Invalid dataset name")
        exit()

    # device = "cuda:3" if torch.cuda.is_available() else "cpu"
    device = args.device if torch.cuda.is_available() else "cpu"

    with open(args.candidates_json) as f:
        candidates = json.load(f)

    with open(references_json) as f:
        references = json.load(f)

    reference_keys = list(references.keys())
    references = list(references.values())
    image_ids = [os.path.join(image_dir, img_id) for img_id in reference_keys]
    candidates = [candidates[cid] for cid in reference_keys]

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
    
    # Get the output file name by replacing .json with _score.txt
    output_dir = os.path.dirname(args.candidates_json)
    output_dir = os.path.join(output_dir, 'scores')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir, os.path.basename(args.candidates_json).replace('.json', '_score.txt'))

    # Save the scores to a text file
    with open(output_file, 'w') as f:
        for k, v in all_scores.items():
            f.write('%s: %.4f\n' % (k, v))

    print(f"Scores saved to {output_file}")