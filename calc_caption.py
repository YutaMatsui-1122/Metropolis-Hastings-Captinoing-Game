from collections import Counter
import re
from utils import *
import clip
import torch
import time

import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

device = torch.device("cuda:0")

clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
data_path = 'dataset/'
prefix_length = 10

with open("dataset/dataset_cache/conceptual_train_dataset_30000.pkl", "rb") as f:
    conceptual_train_dataset = pickle.load(f)

print("conceptual_train_dataset:", len(conceptual_train_dataset) )
# キャプションをtxtファイルに保存
with open(f"dataset/dataset_cache/conceptual_captions_train_30000.txt", "w") as f:
    for caption in conceptual_train_dataset.captions:
        f.write(caption + "\n")

# with open("dataset/dataset_cache/cc3m_train.pkl", "rb") as f:
#     conceptual_train_dataset = pickle.load(f)

# print("conceptual_train_dataset:", len(conceptual_train_dataset))

# # キャプションをtxtファイルに保存
# with open(f"dataset/dataset_cache/cc3m_train.txt", "w") as f:
#     for caption in conceptual_train_dataset.captions:
#         f.write(caption + "\n")

# with open("dataset/dataset_cache/coco_train_dataset_10000.pkl", "rb") as f:
#     conceptual_train_dataset = pickle.load(f)

# # キャプションをtxtファイルに保存
# with open(f"dataset/dataset_cache/coco_train_dataset_10000.txt", "w") as f:
#     for caption in conceptual_train_dataset.captions:
#         f.write(caption + "\n")

        

def calculate_next_word_frequency(texts, target_word):
    next_word_counts = Counter()
    target_word_indices = []
    for i, text in enumerate(texts):
        # 文章を単語に分割
        words = re.findall(r'\w+', text.lower())
        # ターゲットワードが含まれるかどうかをチェック
        if target_word in words:
            target_word_indices.append(i)
            # 対象の単語の次の単語をカウント
            for j in range(len(words) - 1):
                if words[j] == target_word:
                    next_word = words[j + 1]
                    next_word_counts[next_word] += 1
    return target_word_indices, next_word_counts

def calculate_word_frequency(texts):
    word_counts = Counter()
    for text in texts:
        # 文章を単語に分割してカウント
        words = re.findall(r'\w+', text.lower())
        word_counts.update(words)
    return word_counts

def count_words_in_text(word_counts, text):
    # 文章を単語に分割
    words = re.findall(r'\w+', text.lower())
    word_counts_in_text = {word: word_counts[word] for word in words if word in word_counts}
    return word_counts_in_text

def count_sentences_with_multiple_target_words(texts, target_word1, target_word2):
    count = 0
    for text in texts:
        # 文章を単語に分割
        words = re.findall(r'\w+', text.lower())
        # 2つのターゲットワードが同時に含まれるかどうかをチェック
        if target_word1 in words and target_word2 in words:
            count += 1
    return count

print(len(conceptual_train_dataset.captions))

# 文章の単語の生成頻度を計算
s = time.time()
word_counts = calculate_word_frequency(conceptual_train_dataset.captions[:30000])

print("time:", time.time() - s)


# 別の文章
another_text = "wooden spoon spoons fork forks knife knives utensil utensils"

# 別の文章に含まれる各単語の出現数を計算
word_counts_in_another_text = count_words_in_text(word_counts, another_text)

# 結果を出力
print("\n別の文章に含まれる各単語の出現数:")
print(word_counts_in_another_text)

# 対象の単語の次の単語の生成頻度を計算
target_word = "wooden"
target_word_indices, next_word_counts = calculate_next_word_frequency(conceptual_train_dataset.captions[:30000], target_word)
print(f"{target_word}の次の単語の生成頻度:")
#target_wordの次に"spoon"と"spoons"が出現する頻度
print(next_word_counts["spoon"])
print(next_word_counts["spoons"])

#target_wordが含まれる文章のインデックス
print(f"{target_word}が含まれる文章のインデックス:")
print(min(target_word_indices), max(target_word_indices))