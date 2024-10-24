import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import time

class CommunicationField():
    def __init__(self, exp_name, agentA, agentB, EM_iter=10, MH_iter=10, Whole_iter=10):
        self.agentA = agentA
        self.agentB = agentB
        self.EM_iter = EM_iter
        self.MH_iter = MH_iter
        self.Whole_iter = Whole_iter
        self.experiment_setup(exp_name)

    def experiment_setup(self, exp_name):
        # set experiment name
        self.exp_name = exp_name
        self.agentA.exp_name = exp_name
        self.agentB.exp_name = exp_name
        self.agentA.save_dir = f"exp/{self.exp_name}/{self.agentA.agent_name}"
        self.agentB.save_dir = f"exp/{self.exp_name}/{self.agentB.agent_name}"

        # make directory for each agent
        os.makedirs(f"exp/{self.exp_name}", exist_ok=True)
        os.makedirs(f"exp/{self.exp_name}/{self.agentA.agent_name}", exist_ok=True)
        os.makedirs(f"exp/{self.exp_name}/{self.agentB.agent_name}", exist_ok=True)
    
    def communication_field_setup(self, dataloader_MHNG_fix_A, dataloader_MHNG_shuffle_A, dataloader_MHNG_fix_B, dataloader_MHNG_shuffle_B, annealing="None", mode="MHNG"):
        self.agentA.communication_field_setup(dataloader_MHNG_fix_A, dataloader_MHNG_shuffle_A, self.MH_iter, annealing, mode)
        self.agentB.communication_field_setup(dataloader_MHNG_fix_B, dataloader_MHNG_shuffle_B, self.MH_iter, annealing, mode)
        self.initialize_agent()

    def initialize_agent(self):
        self.agentA.initialize_sign()
        self.agentA.lora_setting()
        self.agentB.initialize_sign()
        self.agentB.lora_setting()

    def save_proposed(self, agent, proposed_w):
        """提案された結果をエージェントのディレクトリに保存"""
        file_path = os.path.join(agent.save_dir, f'proposed_w_mh_iter.pt')
        torch.save(proposed_w, file_path)

    def load_proposed(self, agent):
        """保存された提案結果をロード"""
        file_path = os.path.join(agent.save_dir, f'proposed_w_mh_iter.pt')
        return torch.load(file_path)

    def propose_worker(self, rank, mh_iters):
        dist.init_process_group(backend='nccl', rank=rank, world_size=2)
        """各プロセスで提案を実行し、結果をファイルに保存。エージェントごとにデバイスを分ける"""
        if rank == 0:
            proposed_w_As = []
            for mh_iter in range(mh_iters):
                s = time.time()
                device = torch.device(self.agentA.device)
                self.agentA.to(device)
                proposed_w_A = self.agentA.propose()
                proposed_w_As.append(proposed_w_A)
                print("Agent A propose time:", time.time() - s)
            
            proposed_w_As = torch.stack(proposed_w_As, dim=0)            
            print("proposed_w_As:", proposed_w_As.shape)
            self.save_proposed(self.agentA, proposed_w_As)
        elif rank == 1:
            proposed_w_Bs = []
            for mh_iter in range(mh_iters):
                s = time.time()
                device = torch.device(self.agentB.device)
                self.agentB.to(device)
                proposed_w_B = self.agentB.propose()
                proposed_w_Bs.append(proposed_w_B)
                print("Agent B propose time:", time.time() - s)
            
            proposed_w_Bs = torch.stack(proposed_w_Bs, dim=0)
            print("proposed_w_Bs:", proposed_w_Bs.shape) 
            self.save_proposed(self.agentB, proposed_w_Bs)

    def mhcg(self):
        # Generalized EM Captioning Game
        self.agentA.perception()
        self.agentB.perception()
        self.agentA.save_sign("initial")
        self.agentB.save_sign("initial")
        for em_iter in range(self.EM_iter):
            print("EM iteration: ", em_iter)
            # E step with MHCG
            # Each agent updates the sign proposed by the other agent based on the MH communication
            acceptance_rate_A = []
            acceptance_rate_B = []
            # 並列にプロセスを実行
            start = time.time()
            mp.spawn(self.propose_worker, args=(self.MH_iter,), nprocs=2, join=True)
            print("Propose time:", time.time() - start)

            # 提案された結果をロード
            proposed_w_As = self.load_proposed(self.agentA)
            proposed_w_Bs = self.load_proposed(self.agentB)
            print("proposed_w_A:", proposed_w_As.shape)
            print("proposed_w_B:", proposed_w_Bs.shape)

            # make empty dataframe for proposed sign
            for mh_iter in range(self.MH_iter):
                status = f"EM_{em_iter}_MH_{mh_iter}"

                print("MHCG iteration: ", mh_iter)
                proposed_w_A = proposed_w_As[mh_iter]
                proposed_w_B = proposed_w_Bs[mh_iter]

                # 各エージェントが相手の提案を評価
                ar_B = self.agentB.judge(proposed_w_A.to(self.agentB.device), mh_iter)
                acceptance_rate_B.append(ar_B)

                ar_A = self.agentA.judge(proposed_w_B.to(self.agentA.device), mh_iter)
                acceptance_rate_A.append(ar_A)

                # サインを保存（提案を受けた側が保存）
                self.agentA.save_sign(status)
                self.agentB.save_sign(status)

                # 提案されたwを保存（提案を受けた側が保存）
                self.agentA.save_proposed_w(proposed_w_B, status)
                self.agentB.save_proposed_w(proposed_w_A, status)

            # save the acceptance rate with txt file
            with open(f"{self.agentA.save_dir}/acceptance_rate_{em_iter}.txt", "w") as f:
                f.write("\n".join(map(str, acceptance_rate_A)))
            with open(f"{self.agentB.save_dir}/acceptance_rate_{em_iter}.txt", "w") as f:
                f.write("\n".join(map(str, acceptance_rate_B)))
                            
            # Generalized M step
            # Each agent updates the text encoder based on the sign generated by MH communication (今はMH系列の最後のサインだけを使っている)
            self.agentA.update_text_encoder(em_iter)
            self.agentB.update_text_encoder(em_iter)
            
            # Each agent updates text decoder based on the sign generated by MH communication (今はMH系列の最後のサインだけを使っている)
            self.agentA.update_text_decoder(em_iter)
            self.agentB.update_text_decoder(em_iter)


if __name__ == '__main__':
    import pickle, argparse, copy
    from utils import * 
    from one_agent import OneAgent

    parser = argparse.ArgumentParser()
    parser.add_argument('--agentA_clip_arch', default="ViT-B/32", choices=('ViT-B/32', 'ViT-B/16', ))
    parser.add_argument('--agentB_clip_arch', default="ViT-B/32", choices=('ViT-B/32', 'ViT-B/16', ))
    parser.add_argument('--exp_name', default="debug")
    parser.add_argument('--exp_num', default=1, type=int)
    parser.add_argument('--MH_iter', default=10, type=int)
    parser.add_argument('--EM_iter', default=10, type=int)
    parser.add_argument('--Whole_iter', default=10, type=int)
    parser.add_argument('--td_update_epochs', default=10, type=int)
    parser.add_argument('--te_update_epochs', default=10, type=int)
    parser.add_argument('--buffer_size', default=10000, type=int)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--agentA_device', default="cuda:0")  # agentA のデバイス
    parser.add_argument('--agentB_device', default="cuda:1")  # agentB のデバイス
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--temperature', default=0.62, type=float)
    parser.add_argument('--agentA_te_alpha_beta', default=0.5, type=float)
    parser.add_argument('--agentA_td_alpha_beta', default=0.5, type=float)
    parser.add_argument('--agentB_te_alpha_beta', default=0.5, type=float)
    parser.add_argument('--agentB_td_alpha_beta', default=0.5, type=float)
    parser.add_argument('--pin_memory', default=True)
    parser.add_argument('--annealing', default="None")
    parser.add_argument('--mode', default="MHNG")
    parser.add_argument('--import_args', default="None")  # import args from saved json file for same configuration
    args = parser.parse_args()

    exp_name = args.exp_name

    if args.import_args != "None":
        with open(f"exp/{args.import_args}/args.json", "r") as f:
            import_args = json.load(f)
        args = argparse.Namespace(**import_args)
        args.exp_name = exp_name  # overwrite exp_name
    
    print("experiment name:", exp_name)

    for i in range(1, args.exp_num + 1):
        print("exp:", i)
        exp_name = args.exp_name + f"_{i}"

        agentA = OneAgent(agent_name='A', device=args.agentA_device, td_update_epochs=args.td_update_epochs, te_update_epochs=args.te_update_epochs, temperature=args.temperature, te_alpha_beta=args.agentA_te_alpha_beta, td_alpha_beta=args.agentA_td_alpha_beta, clip_arch=args.agentA_clip_arch)
        agentB = OneAgent(agent_name='B', device=args.agentB_device, td_update_epochs=args.td_update_epochs, te_update_epochs=args.te_update_epochs, temperature=args.temperature, te_alpha_beta=args.agentB_te_alpha_beta, td_alpha_beta=args.agentB_td_alpha_beta, clip_arch=args.agentB_clip_arch)
        agentA = agentA.to(args.agentA_device)
        agentB = agentB.to(args.agentB_device)

        # agentA.load_pretrain(probvlm_path="models/official_model/probvlm/CC3M/probvlm_0.2_0.2_20_arch_ViT-B-16-epoch-45.pth", clipcap_path="models/clipcap_vit16_cc3m/clipcap_009.pt", strict_clipcap=False)
        agentA.load_pretrain(probvlm_path="models/official_model/probvlm/CC3M/probvlm_0.2_0.2_20-epoch-69.pth", clipcap_path="models/official_model/clipcap_conceptual_weights.pt", strict_clipcap=False)
        agentB.load_pretrain(probvlm_path="models/official_model/probvlm/COCO/probvlm_0.2_0.3_20-epoch-99.pth", clipcap_path="models/official_model/clipcap_coco_weights.pt", strict_clipcap=False)

        if "debug" in exp_name:
            observation_file = "communication_coco_500_cc3m_500"
            args.buffer_size = 50
        else:
            observation_file = f"communication_coco_5000_cc3m_5000"
        
        with open(f"dataset/dataset_cache/{observation_file}.pkl", "rb") as f:
            observationA_dataset = pickle.load(f)
            observationA_dataset.prefix_length = agentA.prefix_length
        
        with open(f"dataset/dataset_cache/conceptual_train_dataset_30000.pkl", "rb") as f:
            conceptual_pretrain_dataset = pickle.load(f)
            conceptual_pretrain_dataset.prefix_length = agentA.prefix_length
        
        with open(f"dataset/dataset_cache/coco_train_dataset_30000.pkl", "rb") as f:
            coco_pretrain_dataset = pickle.load(f)
            coco_pretrain_dataset.prefix_length = agentB.prefix_length
        
        observationB_dataset = copy.deepcopy(observationA_dataset)

        conceptual_pretrain_loader = torch.utils.data.DataLoader(conceptual_pretrain_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
        coco_pretrain_loader = torch.utils.data.DataLoader(coco_pretrain_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

        coco_test_loader_fix_A = torch.utils.data.DataLoader(observationA_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
        coco_test_loader_shuffle_A = torch.utils.data.DataLoader(observationA_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=args.pin_memory)
        coco_test_loader_fix_B = torch.utils.data.DataLoader(observationB_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
        coco_test_loader_shuffle_B = torch.utils.data.DataLoader(observationB_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=args.pin_memory)

        communication_field = CommunicationField(exp_name, agentA, agentB, EM_iter=args.EM_iter, MH_iter=args.MH_iter, Whole_iter=args.Whole_iter)
        communication_field.communication_field_setup(coco_test_loader_fix_A, coco_test_loader_shuffle_A, coco_test_loader_fix_B, coco_test_loader_shuffle_B, annealing=args.annealing, mode=args.mode)
        save_args_to_json(args, filename="args.json", save_dir=f"exp/{exp_name}")

        agentA.initialize_td_buffer(conceptual_pretrain_loader, buffer_size=args.buffer_size)
        agentB.initialize_td_buffer(coco_pretrain_loader, buffer_size=args.buffer_size)
        agentA.initialize_te_buffer(conceptual_pretrain_loader, buffer_size=args.buffer_size)
        agentB.initialize_te_buffer(coco_pretrain_loader, buffer_size=args.buffer_size)

        # 並列処理のための設定
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '22332'
        mp.set_start_method('spawn', force=True)

        communication_field.mhcg()
