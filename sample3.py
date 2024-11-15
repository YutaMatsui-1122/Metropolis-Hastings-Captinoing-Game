import json
import re

# ファイルパスの設定
coco_file_path = 'dataset/coco/annotations/captions_train2017.json'  # or captions_train2014.json
captions_file_path = 'exp_eval/mhcg_vit_32_16_probvlm_lora_traindata_1/cc3m/cc3m_candidate_B_epoch_29_temperature_0.7_eval.json'

# COCOキャプションデータの読み込み
with open(coco_file_path, 'r') as coco_file:
    coco_data = json.load(coco_file)

# COCOデータのすべてのキャプションに含まれる単語を収集
coco_words = set()
for annotation in coco_data['annotations']:
    caption = annotation['caption']
    words = re.findall(r'\b\w+\b', caption.lower())  # 単語のみを抽出し小文字化
    coco_words.update(words)

# 調査するキャプションファイルの読み込み
with open(captions_file_path, 'r') as captions_file:
    captions_data = json.load(captions_file)

# 含まれていない単語と対応するキャプションのキーを収集
missing_words_with_keys = {}

for key, caption in captions_data.items():
    words = re.findall(r'\b\w+\b', caption.lower())  # 単語のみを抽出し小文字化
    for word in words:
        if word not in coco_words:
            if word not in missing_words_with_keys:
                missing_words_with_keys[word] = []
            missing_words_with_keys[word].append(key)

# 含まれていない単語とそのキャプションのキーを表示
print("COCOデータに含まれていない単語と、その単語が含まれていたキャプションのキー:")
for word, keys in missing_words_with_keys.items():
    print(f"単語: {word} - キー: {keys}")

print(f"\nCOCOデータに含まれていない単語数: {len(missing_words_with_keys)}")
