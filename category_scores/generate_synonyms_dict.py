import os
import json
import time
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import numpy as np
import spacy
import nltk
import torch
import inflection

# 必要な NLTK データをダウンロード
nltk.download("punkt")

# COCOデータセットのパスを設定
data_dir = "../dataset/coco"  # COCO2017データセットのルートディレクトリ
caption_ann_file = os.path.join(data_dir, "annotations/captions_train2017.json")
category_ann_file = os.path.join(data_dir, "annotations/instances_train2017.json")

# Sentence-BERT モデルの設定
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# spaCyの英語モデルをロード
nlp = spacy.load("en_core_web_sm")


def encode_texts(texts):
    """
    Sentence-BERTを用いてテキストを埋め込みに変換
    """
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs)
        sentence_embeddings = embeddings.last_hidden_state.mean(dim=1)
    return sentence_embeddings.numpy()


def extract_phrases_by_ngram(captions, ngram):
    """
    単語ベースでキャプションから n-gram フレーズを抽出
    """
    phrases = []
    for caption in captions:
        tokens = word_tokenize(caption.lower())
        phrases.extend([' '.join(gram) for gram in ngrams(tokens, ngram)])
    phrases = list(set(phrases))  # 重複を削除
    print(f"Extracted {len(phrases)} unique {ngram}-gram phrases.")
    return phrases


def filter_nouns(phrases):
    """
    フレーズや単語から名詞のみを抽出し、語順の正しさを確認
    """
    nouns = []
    for phrase in tqdm(phrases):
        doc = nlp(phrase)
        if len(doc) > 1:
            is_valid = all(
                (token.pos_ in ["NOUN", "PROPN"] or token.pos_ == "ADJ") for token in doc
            ) and doc[-1].pos_ in ["NOUN", "PROPN"]
            if is_valid:
                nouns.append(phrase)
        elif len(doc) == 1 and doc[0].pos_ in ["NOUN", "PROPN"]:
            nouns.append(phrase)
    print(f"Filtered {len(nouns)} nouns from {len(phrases)} phrases.")
    return nouns


def get_all_categories_with_context(categories_data):
    """
    カテゴリ名を "category (super_cat)" の形式に変換
    """
    categories_with_context = [
        # f"{cat['name']} ({cat['supercategory']})"
        # f"{cat['name']} as {cat['supercategory']}"
        cat["name"]
        
        for cat in categories_data["categories"]
    ]

    supercategories = list(set(cat["supercategory"] for cat in categories_data["categories"]))
    return categories_with_context + supercategories


def extract_top_synonyms(category_name, category_with_context, phrases, all_categories_and_supercats, top_n=100):
    """
    指定されたカテゴリに対する類義語の上位N個を抽出
    """
    print(f"Calculating synonyms for category: {category_with_context}")

    # カテゴリリストには "category (super_cat)" の形式のみを使用
    category_list = list(set(all_categories_and_supercats))
    if category_with_context not in category_list:
        category_list.append(category_with_context)

    # カテゴリ名の複数形を生成
    category_plural = inflection.pluralize(category_name)

    # 埋め込み計算
    category_embeddings = encode_texts(category_list)
    phrase_embeddings = encode_texts(phrases)

    # 類似度計算
    similarities = cosine_similarity(phrase_embeddings, category_embeddings)

    print(similarities.shape)

    # 類義語抽出
    results = []
    target_index = category_list.index(category_with_context)

    # カテゴリ名とその複数形を類似度1.0で追加
    if len(phrases[0].split()) == len(category_name.split()):
        results.append((category_name, 1.0))  # 元のカテゴリ名を追加
        if category_plural != category_name:  # 複数形が元のカテゴリ名と異なる場合のみ追加
            results.append((category_plural, 1.0))  # 複数形を追加

    print(category_list)

    for i, phrase in enumerate(phrases):
        phrase_sim_scores = similarities[i]
        max_sim_idx = np.argmax(phrase_sim_scores)
        max_sim_score = phrase_sim_scores[max_sim_idx]

        # カテゴリ名とその複数形は追加しない
        if phrase in [category_name, category_plural]:
            continue
        elif max_sim_idx == target_index:
            results.append((phrase, float(max_sim_score)))

    # 類似度が高い順にソート
    results = sorted(results, key=lambda x: x[1], reverse=True)

    # トップN件を返す
    top_synonyms = results[:top_n]

    print(f"Top {top_n} synonyms extracted for category: {category_with_context}")
    return top_synonyms


def get_captions_for_category(category_id, caption_ann_file, category_ann_file):
    """
    指定されたカテゴリIDに関連するキャプションを取得
    """
    with open(caption_ann_file, "r") as f:
        captions_data = json.load(f)
    with open(category_ann_file, "r") as f:
        categories_data = json.load(f)

    categories = {cat["id"]: cat["name"] for cat in categories_data["categories"]}
    supercategories = {cat["id"]: cat["supercategory"] for cat in categories_data["categories"]}
    category_name = categories.get(category_id, None)
    supercategory_name = supercategories.get(category_id, None)

    if category_name is None or supercategory_name is None:
        raise ValueError(f"Category ID '{category_id}' not found in the dataset.")

    print(f"Processing Category ID: {category_id}, Name: {category_name}")

    image_ids = [
        ann["image_id"]
        for ann in categories_data["annotations"]
        if ann["category_id"] == category_id
    ]
    print(f"Number of images for category '{category_name}': {len(image_ids)}")
    captions = [
        ann["caption"]
        for ann in captions_data["annotations"]
        if ann["image_id"] in image_ids
    ]
    # captions = []
    # for ann in tqdm(captions_data["annotations"]):
    #     if ann["image_id"] in image_ids:
    #         captions.append(ann["caption"])
    print(f"Number of captions for category '{category_name}': {len(captions)}")
    return category_name, supercategory_name, captions, categories_data

def process_category(category_id, all_categories_and_supercats, top_n=100):
    """
    カテゴリごとに n-gram フレーズと類義語を抽出
    """
    category_name, supercategory_name, captions, categories_data = get_captions_for_category(
        category_id, caption_ann_file, category_ann_file
    )

    # "category (super_cat)"形式の対象カテゴリ名を作成
    # category_with_context = f"{category_name} ({supercategory_name})"
    category_with_context = category_name

    print("Extracting 1-gram phrases...")
    one_gram_phrases = extract_phrases_by_ngram(captions, ngram=1)
    print("Extracting 2-gram phrases...")
    two_gram_phrases = extract_phrases_by_ngram(captions, ngram=2)

    print("Filtering nouns for 1-gram...")
    one_gram_phrases = filter_nouns(one_gram_phrases)
    print("Filtering nouns for 2-gram...")
    two_gram_phrases = filter_nouns(two_gram_phrases)

    print("Extracting top synonyms for 1-gram...")
    one_gram_synonyms = extract_top_synonyms(category_name, category_with_context, one_gram_phrases, all_categories_and_supercats, top_n=top_n)
    print("Extracting top synonyms for 2-gram...")
    two_gram_synonyms = extract_top_synonyms(category_name, category_with_context, two_gram_phrases, all_categories_and_supercats, top_n=top_n)

    return {
        "category_name": category_name,
        "one_gram_synonyms": one_gram_synonyms,
        "two_gram_synonyms": two_gram_synonyms,
    }


if __name__ == "__main__":
    top_n = 100
    output_path = f"all_categories_synonyms_top{top_n}.json"

    with open(category_ann_file, "r") as f:
        categories_data = json.load(f)

    # すべてのカテゴリとスーパーカテゴリを "category (super_cat)" 形式で取得
    all_categories_and_supercats = get_all_categories_with_context(categories_data)

    all_category_ids = [cat["id"] for cat in categories_data["categories"] if cat["id"]]

    results = {}
    for category_id in all_category_ids:
        print(f"Processing category ID: {category_id}")
        start_time = time.time()
        results[category_id] = process_category(category_id, all_categories_and_supercats, top_n=top_n)
        print(f"Finished processing category ID: {category_id} in {time.time() - start_time:.2f} seconds.")

        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)
    print(f"Results saved to {output_path}")