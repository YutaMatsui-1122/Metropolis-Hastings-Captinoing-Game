from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import time

# モデルとトークナイザーのロード
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# パディングトークンと左側パディングを設定
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = GPT2LMHeadModel.from_pretrained(model_name)

# モデルをGPUに移動（もし使用可能なら）
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 100個の適当な入力テキスト
input_texts = [f"Sample sentence {i}" for i in range(1, 101)]  # "Sample sentence 1" から "Sample sentence 100"まで生成

# 各テキストをトークン化し、バッチ化
encoding = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True)
input_ids = encoding.input_ids.to(device)
attention_mask = encoding.attention_mask.to(device)

# 生成の時間を計測
start_generate_time = time.time()
# バッチとして生成を実行
output = model.generate(
    input_ids,
    attention_mask=attention_mask,  # attention_maskを追加
    max_length=40,       # 各シーケンスの最大トークン数
    num_return_sequences=1,  # 各入力ごとに1つのシーケンスを生成
    no_repeat_ngram_size=2,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.7
)
end_generate_time = time.time()

# 生成時間の表示
generate_time = end_generate_time - start_generate_time
print(f"Text generation time: {generate_time:.2f} seconds")

# デコードの時間を計測
start_decode_time = time.time()
# 各生成結果をデコード
generated_texts = [tokenizer.decode(output[i], skip_special_tokens=True) for i in range(len(input_texts))]
end_decode_time = time.time()

# デコード時間の表示
decode_time = end_decode_time - start_decode_time
print(f"Text decoding time: {decode_time:.2f} seconds")

# 結果を表示（最初の10個のみ表示）
for idx, text in enumerate(generated_texts[:10]):
    print(f"Input: {input_texts[idx]}\nGenerated: {text}\n")
