import torch
import torch.multiprocessing as mp

def light_worker(rank):
    print(f"Worker {rank} is running a light task.")
    # 簡単な計算を実行してみる
    result = rank * 2
    print(f"Worker {rank} completed the light task with result {result}.")

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # 2つの軽いワーカーを起動
    mp.spawn(light_worker, args=(), nprocs=10, join=True)
