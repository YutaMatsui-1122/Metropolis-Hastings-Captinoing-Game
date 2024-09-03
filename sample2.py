import torch

def extract_top_k(logits, top_k):
    # top_k 値とそのインデックスを抽出
    logits_value, logits_index = torch.topk(logits, k=top_k, dim=2, largest=True, sorted=True)
    return logits_value, logits_index

def reconstruct_logits(logits_shape, logits_value, logits_index, fill_value=-1e6):
    # テンソルを作成し、小さい値で初期化（float型で指定）
    reconstructed_logits = torch.full(logits_shape, fill_value, dtype=torch.float)
    
    # バッチサイズとnum_tokenの範囲を生成
    batch_size, num_tokens = logits_shape[0], logits_shape[1]
    b, t = torch.meshgrid(torch.arange(batch_size), torch.arange(num_tokens), indexing='ij')
    
    # scatter_を使用して一括でログを復元
    reconstructed_logits[b, t] = reconstructed_logits[b, t].scatter_(2, logits_index, logits_value.float())
    
    return reconstructed_logits

# 元の logits テンソルを定義
logits_input = torch.tensor([
    [[1, 3, 4, 5, 0], [6, 9, 2, 8, 7], [4, 2, 9, 1, 6]],
    [[10, 3, 2, 0, 8], [4, 5, 3, 7, 1], [6, 8, 7, 5, 9]],
    [[7, 2, 5, 1, 9], [8, 0, 3, 6, 4], [2, 1, 0, 7, 8]]
])

# top_k を 3 に設定
top_k_setting = 2

# top_k 値とインデックスを抽出
values, indices = extract_top_k(logits_input, top_k_setting)

# 復元された logits を生成
reconstructed_logits = reconstruct_logits(logits_input.shape, values, indices)


# 結果の表示
print("元の logits テンソル:\n", logits_input)
print("top_k 値:\n", values)
print("top_k インデックス:\n", indices)
print("復元された logits テンソル:\n", reconstructed_logits)