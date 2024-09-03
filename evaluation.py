from pycocoevalcap.eval import COCOEvalCap
from utils import *
from torchmetrics.text import SacreBLEUScore
from nltk.translate.meteor_score import meteor_score


data_path = 'dataset/'
prefix_length = 10
# test_dataset = CocoDataset(root = data_path, transform=preprocess,data_mode="test", prefix_length=prefix_length, normalize_prefix=True, datasize="full_10000", use_imageid=True)
with open("dataset/dataset_cache/coco_test_dataset.pkl", "rb") as f:
    test_dataset = pickle.load(f)
    test_dataset.prefix_length = 10
print("test_dataset:", len(test_dataset))

coco = test_dataset.coco

file_pathes = [
    "models/derpp_coco2conceptual_lr5e-6_test/generated_sentences_t_0.6_sample",
    # "exp/GEMCG_first_1/A/generated_sentences_t_0.6_sample_em_iter_3",
    # "exp/GEMCG_first_1/A/generated_sentences_t_0.6_sample_em_iter_9",
    # "exp/GEMCG_first_1/B/generated_sentences_t_0.6_sample_em_iter_3",
    # "models/lora_coco2conceptual_lr5e-6/generated_sentences_006_beam",
    # "models/lora_coco2conceptual_lr5e-6/generated_sentences_007_beam",


]

model_names = [
    "DERPP_coco2conceptual_t_0.6_sample",
    # "GEMCG_first_1_A_em_iter_3",
    # "GEMCG_first_1_A_em_iter_9",
    # "GEMCG_first_1_B_em_iter_3",
    # "GEMCG_first_1_B_em_iter_9",
]

# dataframe for all metrics
score_df = pd.DataFrame(columns=["Model", "CC3M BLEU", "CC3M METEOR", "CC3M BERT", "CC3M CLIP", "COCO BLEU", "COCO METEOR", "COCO BERT", "COCO CLIP"])

for file_path, model_name in zip(file_pathes, model_names):
    csv_filename = f"{file_path}.csv"
    json_filename = f"{file_path}.json"

    # create Res json file
    with open(json_filename, "w") as f:
        df = pd.read_csv(csv_filename)
        coco_df = df[df['ImageID'] != 'conceptual']
        gens = coco_df['Generated'].tolist()
        image_ids = coco_df['ImageID'].tolist()
        res = []
        for i, gen in enumerate(gens):
            res.append({"image_id": int(image_ids[i]), "caption": gen})
        json.dump(res, f)

    with open(json_filename,"r") as f:
        generated_captions = json.load(f)
    
    # check if all image ids in generated captions are in coco dataset
    # for image_id in res_image_ids:
    #     if image_id not in coco_image_ids:
    #         print("Image ID not in COCO dataset:", image_id)

    cocoRes = coco.loadRes(generated_captions)
    
    # create cocoEval object by taking coco and cocoRes
    cocoEval = COCOEvalCap(coco, cocoRes)

    cocoEval.params['image_id'] = cocoRes.getImgIds()
    refs = []
    gens = []
    for imgId in cocoRes.getImgIds():
        ref =[]
        gen = []
        for ann in coco.imgToAnns[imgId]:
            ref.append(ann['caption'])
        refs.append(ref)
        gens.append(cocoRes.imgToAnns[imgId][0]['caption'])

    sacre_bleu = SacreBLEUScore(lowercase=True)
    # bleu_score = sacre_bleu(gens, refs) 
    coco_bleu_score = sacre_bleu(gens, refs).item() 
    print("sacre BLEU:", coco_bleu_score)

    # meteor = evaluate.load('meteor')
    flattened_references = [ref for ref_list in refs for ref in ref_list]
    flattened_hypotheses = [hyp for hyp, ref_list in zip(gens, refs) for _ in ref_list]
    # result = meteor.compute(predictions=flattened_hypotheses, references=flattened_references)
    # print("METEOR:", result)

    # 全ての参照文と生成文を一つの評価として処理
    # all_refs_split = [[ref.split() for ref in ref_group] for ref_group in refs]
    # all_hyps_split = [hyp.split() for hyp in gens]
    all_refs_split = [ref.split() for ref in flattened_references]
    all_hyps_split = [hyp.split() for hyp in flattened_hypotheses]

    # 全体のMETEORスコアの計算
    # 全ての生成文とその対応する参照文群についてスコアを計算し、その平均を取る
    total_score = 0
    for ref, hyp in zip(all_refs_split, all_hyps_split):
        score = meteor_score([ref], hyp, alpha=0.9, beta=3, gamma=0.5)
        total_score += score

    # 全体の平均METEORスコアを計算
    # average_total_score = total_score / len(all_hyps_split)
    coco_metereor_score = total_score / len(all_hyps_split) 

    # print(f"Total METEOR Score for all data: {average_total_score:.4f}")
    print("METEOR Score:", coco_metereor_score)

    # bleu = sacrebleu.raw_corpus_bleu(gens, refs, .01)
    # print("sacre BLEU:", bleu.score)
    # evaluate results
    # cocoEval.evaluate()

    # print(file_path)

    # # print output evaluation scores
    # for metric, score in cocoEval.eval.items():
    #     print('%s: %.3f'%(metric, score))

    # calculate clip score
    coco_clip_scores = coco_df['CLIP_Score'].tolist()
    print("CLIP Score:", np.mean(coco_clip_scores))

    # calculate bert score
    coco_bert_score = coco_df["BERT_Score"].tolist()
    print("BERT Score:", np.mean(coco_bert_score))

    # save scores with text file

    with open(f"{file_path}_scores.txt", "w") as f:
        f.write("COCO Evaluation Scores\n")
        f.write(f"BLEU Score: {coco_bleu_score}\n")
        f.write(f"METEOR Score: {coco_metereor_score}\n")
        f.write(f"CLIP Score: {np.mean(coco_clip_scores)}\n")
        f.write(f"BERT Score: {np.mean(coco_bert_score)}\n")
        f.write("\n")

    conceptual_df = df[df['ImageID'] == 'conceptual']

    conceptual_gens = conceptual_df['Generated'].tolist()
    conceptual_refs = conceptual_df['Reference'].tolist()
    conceptual_refs_bleu = [[ref] for ref in conceptual_refs]
    sacre_bleu = SacreBLEUScore(lowercase=True)
    # bleu_score = sacre_bleu(conceptual_gens,conceptual_refs_bleu)
    cc3m_bleu_score = sacre_bleu(conceptual_gens,conceptual_refs_bleu).item()
    print("sacre BLEU:", cc3m_bleu_score)

    all_refs_split = [ref.split() for ref in conceptual_refs]
    all_hyps_split = [hyp.split() for hyp in conceptual_gens]

    # 全体のMETEORスコアの計算
    # 全ての生成文とその対応する参照文群についてスコアを計算し、その平均を取る
    total_score = 0
    for ref, hyp in zip(all_refs_split, all_hyps_split):
        score = meteor_score([ref], hyp, alpha=0.9, beta=3, gamma=0.5)
        total_score += score

    # 全体の平均METEORスコアを計算
    cc3m_metereor_score = total_score / len(all_hyps_split)

    print("METEOR Score:", cc3m_metereor_score)

    # calculate clip score
    # clip_scores = conceptual_df['CLIP_Score'].tolist()
    cc3m_clip_scores = conceptual_df['CLIP_Score'].tolist() 
    print("CLIP Score:", np.mean(cc3m_clip_scores))

    # calculate bert score
    cc3m_bert_score = conceptual_df["BERT_Score"].tolist()
    print("BERT Score:", np.mean(cc3m_bert_score))

    # adding scores to text file
    with open(f"{file_path}_scores.txt", "a") as f:
        f.write("Conceptual Evaluation Scores\n")
        f.write(f"BLEU Score: {cc3m_bleu_score}\n")
        f.write(f"METEOR Score: {cc3m_metereor_score}\n")
        f.write(f"CLIP Score: {np.mean(cc3m_clip_scores)}\n")
        f.write(f"BERT Score: {np.mean(cc3m_bert_score)}\n")
        f.write("\n")

    # add scores to dataframe
    score_df = score_df.append(
        {
            "Model": model_name,
            "CC3M BLEU": round(cc3m_bleu_score, 4) * 100,
            "CC3M METEOR": round(cc3m_metereor_score, 4) * 100,
            "CC3M BERT": round(np.mean(cc3m_bert_score), 4),
            "CC3M CLIP": round(np.mean(cc3m_clip_scores), 4),
            "COCO BLEU": round(coco_bleu_score, 4) * 100,
            "COCO METEOR": round(coco_metereor_score, 4) * 100,
            "COCO BERT": round(np.mean(coco_bert_score), 4),
            "COCO CLIP": round(np.mean(coco_clip_scores), 4)
        },
        ignore_index=True
    
    )


# save dataframe to csv
pd.set_option('display.float_format', lambda x: f'{x:.4f}')
score_df.to_csv("scores_matrix.csv")