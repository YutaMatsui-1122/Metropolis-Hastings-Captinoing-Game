import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time

# モデルとトークナイザーのロード
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# 100個の適当な入力テキスト
input_texts = [f"Sample sentence {i}" for i in range(1, 1001)]
encoding = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True)
input_ids = encoding.input_ids
attention_mask = encoding.attention_mask

# シングルGPUバージョンの実行
def run_single_gpu(input_ids, attention_mask):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # 生成の時間を計測
    start_generate_time = time.time()

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=40,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

    end_generate_time = time.time()
    generate_time = end_generate_time - start_generate_time
    print(f"Single GPU - Text generation time: {generate_time:.2f} seconds")

# DDPの初期化関数
def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# DDPの終了処理
def cleanup():
    dist.destroy_process_group()

# DDPメイン処理
def main_ddp(input_ids, attention_mask):
    world_size = 2  # GPUの数
    batch_size = 256  # 全体のバッチサイズ
    per_process_batch_size = batch_size // world_size  # 各プロセスごとのバッチサイズ
    
    # DDP実行
    mp.spawn(run_ddp, args=(world_size, input_ids, attention_mask, per_process_batch_size), nprocs=world_size, join=True)

# DDPメイン処理の中で呼び出すrun_ddp関数
def run_ddp(rank, world_size, input_ids, attention_mask, per_process_batch_size):
    setup(rank, world_size)

    # 使用するGPUをcuda:0, cuda:1に限定
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # モデルをロードしてDDPにラップ
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # データを分割
    num_samples = input_ids.size(0) // world_size
    start_index = rank * num_samples
    end_index = start_index + num_samples if rank != world_size - 1 else input_ids.size(0)

    input_ids_split = input_ids[start_index:end_index].to(device)
    attention_mask_split = attention_mask[start_index:end_index].to(device)

    # 生成の時間を計測
    start_generate_time = time.time()

    output = model.module.generate(
        input_ids_split,
        attention_mask=attention_mask_split,
        max_length=40,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

    end_generate_time = time.time()
    generate_time = end_generate_time - start_generate_time
    print(f"Rank {rank} - DDP Text generation time: {generate_time:.2f} seconds")

    # 終了処理
    cleanup()


# 比較実行
if __name__ == "__main__":
    # Single GPU 実行
    print("Running Single GPU version")
    run_single_gpu(input_ids, attention_mask)

    # DDP 実行
    print("\nRunning DDP version")
    main_ddp(input_ids, attention_mask)
