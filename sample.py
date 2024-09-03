import torch 
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
for i in range(10):
    path = f"exp/gemcg_unified_dataset_coco5000_cc3m5000_alpha03_beta03_1/A/gpt_token_buffer_{i}.pt"
    buffer = torch.load(path)
    print(buffer[1])
    print(tokenizer.decode(buffer[1]))