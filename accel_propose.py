import torch, time
from one_agent import OneAgent
import pickle
from utils import *

for workers in [1, 2, 4, 8]:
    batch_size = 100
    num_workers = 8
    device = 'cuda:3'

    agent = OneAgent(device=device, temperature=0.7)
    agent = agent.to(device)
    agent.load_pretrain(probvlm_path="models/official_model/probvlm/COCO/probvlm_0.2_0.3_20-epoch-99.pth", clipcap_path="models/official_model/clipcap_coco_weights.pt", strict_clipcap=False)

    observation_file = f"communication_coco_500_cc3m_500"

    with open(f"dataset/dataset_cache/{observation_file}.pkl", "rb") as f:
        observationA_dataset = pickle.load(f)
        observationA_dataset.prefix_length = agent.prefix_length

    observation_loader_fix = torch.utils.data.DataLoader(observationA_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    observation_loader_shuffle = torch.utils.data.DataLoader(observationA_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    agent.communication_field_setup(observation_loader_fix, observation_loader_shuffle, MH_iter=10)

    agent.perception()
    start = time.time()
    agent.propose()
    print(f"workers: {workers}, time: {time.time()-start}")
    continue

    exit()
    # DDPによる提案実行
    world_size = 2  # cuda:0 と cuda:1 のみを使用
    available_gpus = [0, 1]  # 使用するGPUのインデックス

    # CUDA_VISIBLE_DEVICES で使用するGPUを制限
    torch.cuda.set_device(available_gpus[0])

    proposed_captions = agent.propose_ddp(world_size=world_size)

    if proposed_captions is not None:
        print("Generated captions:")
        for caption in proposed_captions:
            print(caption)
