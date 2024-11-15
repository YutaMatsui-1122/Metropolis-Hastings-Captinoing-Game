import json
from collections import Counter
import numpy as np

# ファイルのパス
file_path = 'dataset/coco/annotations/captions_train2017.json'

# JSONファイルを読み込む
with open(file_path, 'r') as file:
    data = json.load(file)

# 画像数とキャプション数を表示
num_images = len(data['images'])
num_captions = len(data['annotations'])
print(f"画像数: {num_images}")
print(f"キャプション数: {num_captions}")

# キャプションの長さ（単語数）の分布を調べる
caption_lengths = [len(annotation['caption'].split()) for annotation in data['annotations']]
avg_caption_length = np.mean(caption_lengths)
max_caption_length = np.max(caption_lengths)
min_caption_length = np.min(caption_lengths)

print(f"キャプションの平均長さ（単語数）: {avg_caption_length:.2f}")
print(f"キャプションの最大長さ（単語数）: {max_caption_length}")
print(f"キャプションの最小長さ（単語数）: {min_caption_length}")

# 単語頻度のカウント
all_captions = " ".join([annotation['caption'] for annotation in data['annotations']])
words = all_captions.split()
word_counts = Counter(words)

# 最も頻出する単語トップ10を表示
common_words = word_counts.most_common(10)
print("最も頻出する単語トップ10:")
for word, count in common_words:
    print(f"{word}: {count}")

# 特定の画像に対するすべてのキャプションを表示する例
example_image_id = data['images'][0]['id']
captions_for_image = [annotation['caption'] for annotation in data['annotations'] if annotation['image_id'] == example_image_id]

print(f"\n画像ID {example_image_id} のキャプション:")
for caption in captions_for_image:
    print(f"- {caption}")
