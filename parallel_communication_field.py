import os
import torch
import torch.multiprocessing as mp
import time
import socket
import random
# import 

class CommunicationField():
    def __init__(self, exp_name, agentA, agentB, Whole_iter=10, MH_iter=10):
        self.agentA = agentA
        self.agentB = agentB
        self.Whole_iter = Whole_iter
        self.MH_iter = MH_iter
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
    
    def communication_field_setup(self, dataloader_MHNG_fix_A, dataloader_MHNG_shuffle_A, dataloader_MHNG_fix_B, dataloader_MHNG_shuffle_B, mode="MHNG"):
        self.agentA.communication_field_setup(dataloader_MHNG_fix_A, dataloader_MHNG_shuffle_A, self.MH_iter, self.Whole_iter, mode)
        self.agentB.communication_field_setup(dataloader_MHNG_fix_B, dataloader_MHNG_shuffle_B, self.MH_iter, self.Whole_iter, mode)
            
        self.initialize_agent()

    def initialize_agent(self):
        self.agentA.initialize_sign()
        self.agentB.initialize_sign()

    def save_proposed(self, agent, proposed_w):
        """提案された結果をエージェントのディレクトリに保存"""
        file_path = os.path.join(agent.save_dir, f'proposed_w_mh_iter.pt')
        torch.save(proposed_w, file_path)

    def load_proposed(self, agent):
        """保存された提案結果をロード"""
        file_path = os.path.join(agent.save_dir, f'proposed_w_mh_iter.pt')
        return torch.load(file_path)

    def propose_parallel(self):
        # プロセスを使って提案を行う
        agentA_te_buffer, agentA_td_buffer = self.agentA.te_buffer, self.agentA.td_buffer
        agentB_te_buffer, agentB_td_buffer = self.agentB.te_buffer, self.agentB.td_buffer
        del self.agentA.te_buffer, self.agentA.td_buffer
        del self.agentB.te_buffer, self.agentB.td_buffer
        mp.spawn(propose_worker, args=(self.agentA, self.agentB, self.MH_iter,), nprocs=2, join=True)
        # バッファを復元
        self.agentA.te_buffer, self.agentA.td_buffer = agentA_te_buffer, agentA_td_buffer
        self.agentB.te_buffer, self.agentB.td_buffer = agentB_te_buffer, agentB_td_buffer
        proposed_w_As = self.load_proposed(self.agentA)
        proposed_w_Bs = self.load_proposed(self.agentB)
        return proposed_w_As, proposed_w_Bs

    def update_text_encoder_parallel(self, em_iter, file_name = "current_model"):
        # プロセスを使ってテキストエンコーダを更新
        # エージェントごとのバッファを取得
        agentA_td_buffer = self.agentA.td_buffer
        agentB_td_buffer = self.agentB.td_buffer
        # エージェントごとのバッファを一度削除する
        del self.agentA.td_buffer, self.agentB.td_buffer

        mp.spawn(update_text_encoder_worker, args=(self.agentA, self.agentB, em_iter, file_name), nprocs=2, join=True)
        # バッファを復元
        self.agentA.td_buffer = agentA_td_buffer
        self.agentB.td_buffer = agentB_td_buffer

        self.agentA.ProbVLM_Net.load_state_dict(torch.load(os.path.join(self.agentA.save_dir, f'{file_name}_A.pt')))
        self.agentB.ProbVLM_Net.load_state_dict(torch.load(os.path.join(self.agentB.save_dir, f'{file_name}_B.pt')))

    def only_update_text_decoder(self, fine_tune_agent="A"):
        print("Fine-tune agent:", fine_tune_agent)
        # テキストエンコーダのみ更新
        self.agentA.perception()
        self.agentB.perception()

        self.agentA.mode = "all_accept"
        self.agentB.mode = "all_accept"

        if fine_tune_agent == "A":
            for epoch in range(self.Whole_iter):
                proposed_w, proposed_captionB = self.agentB.propose(return_caption=True, mh_epoch=0)

                ar = self.agentA.judge(proposed_w.to(self.agentA.device), epoch)

                self.agentA.update_text_decoder(epoch)
        else:
            for epoch in range(self.Whole_iter):
                proposed_w, proposed_captionA = self.agentA.propose(return_caption=True, mh_epoch=0)

                ar = self.agentB.judge(proposed_w.to(self.agentB.device), epoch)

                self.agentB.update_text_decoder(epoch)
    
    def update_distillation(self, distillation_agent="A"):
        if distillation_agent == "A":
            for epoch in range(self.Whole_iter):
                self.agentA.update_text_decoder_distillation(self.agentB, epoch)
        else:
            for epoch in range(self.Whole_iter):
                self.agentB.update_text_decoder_distillation(self.agentA, epoch)
                
    def mhcg(self):
        # Metropolis-Hastings Captioning Game   
        time_list = []
        start = time.time()

        agentA.perception()
        agentB.perception()
        self.agentA.save_sign("initial")
        self.agentB.save_sign("initial")
        print("Time for perception:", time.time() - start)
        time_list.append(time.time() - start)

        for em_iter in range(self.Whole_iter):
            start = time.time()
            print("EM iteration: ", em_iter)
            # Each agent updates the sign proposed by the other agent based on the MH communication
            acceptance_rate_A = []
            acceptance_rate_B = []

            start = time.time()
            proposed_w_As, proposed_w_Bs = self.propose_parallel()
            print("time:", time.time() - start)

            # make empty dataframe for proposed sign
            for mh_iter in range(self.MH_iter):
                status = f"EM_{em_iter}_MH_{mh_iter}"

                print("MHCG iteration: ", mh_iter)
                proposed_w_A = proposed_w_As[mh_iter]
                proposed_w_B = proposed_w_Bs[mh_iter]

                # 各エージェントが相手の提案を評価
                ar_B = self.agentB.judge(proposed_w_A.to(self.agentB.device))
                acceptance_rate_B.append(ar_B)

                ar_A = self.agentA.judge(proposed_w_B.to(self.agentA.device))
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

            self.update_text_encoder_parallel(em_iter, file_name=f"current_model")
            
            # Each agent updates text decoder based on the sign generated by MH communication (今はMH系列の最後のサインだけを使っている)
            self.agentA.update_text_decoder(em_iter)
            self.agentB.update_text_decoder(em_iter)

            time_list.append(time.time() - start)
            print("Time List:", time_list)

def find_open_port():
    while True:
        port = random.randint(10000, 20000)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(('localhost', port)) != 0:  # ポートが使用されていないことを確認
                return port

if __name__ == '__main__':
    import pickle, argparse, copy
    from utils import * 
    from one_agent import OneAgent

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default="debug")
    parser.add_argument('--dataset_prefix', default="coco_2017_common_person_only")
    parser.add_argument('--MH_iter', default=3, type=int)
    parser.add_argument('--Whole_iter', default=30, type=int)
    parser.add_argument('--td_update_epochs', default=10, type=int)
    parser.add_argument('--te_update_epochs', default=1, type=int)
    parser.add_argument('--buffer_size', default=5000, type=int)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--agentA_device', default="cuda:0")  # agentA のデバイス
    parser.add_argument('--agentB_device', default="cuda:1")  # agentB のデバイス
    parser.add_argument('--batch_size', default=40, type=int)
    parser.add_argument('--temperature', default=0.7, type=float)
    parser.add_argument('--agentA_te_alpha_beta', default=0.5, type=float)
    parser.add_argument('--agentA_td_alpha_beta', default=0.01, type=float)
    parser.add_argument('--agentB_te_alpha_beta', default=0.5, type=float)
    parser.add_argument('--agentB_td_alpha_beta', default=0.01, type=float)
    parser.add_argument('--td_lr', default=1e-4 , type=float)
    parser.add_argument('--te_lr', default=1e-6, type=float)
    parser.add_argument('--lora_r', default=8, type=int)
    parser.add_argument('--lora_alpha', default=32, type=float)
    parser.add_argument('--lora_dropout', default=0.1, type=float)
    parser.add_argument('--pin_memory', default=False)
    parser.add_argument('--annealing', default="None")
    parser.add_argument('--mode', default="MHNG")
    parser.add_argument('--all_acceptance_agent', default=None, choices=(None, "A", "B"))
    parser.add_argument('--distillation_agent', default=None, choices=(None, "A", "B"))
    args = parser.parse_args()

    exp_name = args.exp_name
    
    print("experiment name:", exp_name)

    exp_name = args.exp_name 
    agentA = OneAgent(agent_name='A', device=args.agentA_device, td_update_epochs=args.td_update_epochs, te_update_epochs=args.te_update_epochs, temperature=args.temperature, te_alpha_beta=args.agentA_te_alpha_beta, td_alpha_beta=args.agentA_td_alpha_beta, clip_arch="ViT-B/16", td_lr=args.td_lr, te_lr=args.te_lr,)
    agentB = OneAgent(agent_name='B', device=args.agentB_device, td_update_epochs=args.td_update_epochs, te_update_epochs=args.te_update_epochs, temperature=args.temperature, te_alpha_beta=args.agentB_te_alpha_beta, td_alpha_beta=args.agentB_td_alpha_beta, clip_arch="ViT-B/32", td_lr=args.td_lr, te_lr=args.te_lr,)  
    agentA = agentA.to(args.agentA_device)
    agentB = agentB.to(args.agentB_device)
    
    agentA.load_pretrain(probvlm_path=f"pretrain_models/{args.dataset_prefix}/COCO_A/probvlm/probvlm-epoch-49.pth", 
                         clipcap_path=f"pretrain_models/{args.dataset_prefix}/COCO_A/clipcap/clipcap_019.pt", strict_clipcap=False)
    
    agentB.load_pretrain(probvlm_path=f"pretrain_models/{args.dataset_prefix}/COCO_B/probvlm/probvlm-epoch-49.pth", 
                         clipcap_path=f"pretrain_models/{args.dataset_prefix}/COCO_B/clipcap/clipcap_019.pt", strict_clipcap=False)
    
    agentA.lora_setting(r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
    agentB.lora_setting(r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)

    # Dataset preparation
    # Observation for both agents
    with open(f"dataset/dataset_cache/{args.dataset_prefix}/train_split_dataset_mhcg.pkl", "rb") as f:
        observationA_dataset = pickle.load(f)
        observationA_dataset.prefix_length = agentA.prefix_length
    
    # pretrain dataset for agent A
    with open(f"dataset/dataset_cache/{args.dataset_prefix}/train_split_dataset_a.pkl", "rb") as f:
        coco_a_pretrain_dataset = pickle.load(f)
        coco_a_pretrain_dataset.prefix_length = agentA.prefix_length
    
    # pretrain dataset for agent B
    with open(f"dataset/dataset_cache/{args.dataset_prefix}/train_split_dataset_b.pkl", "rb") as f:
        coco_b_pretrain_dataset = pickle.load(f)
        coco_b_pretrain_dataset.prefix_length = agentB.prefix_length
    
    observationB_dataset = copy.deepcopy(observationA_dataset)

    coco_a_pretrain_loader = torch.utils.data.DataLoader(coco_a_pretrain_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
    coco_b_pretrain_loader = torch.utils.data.DataLoader(coco_b_pretrain_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

    coco_test_loader_fix_A = torch.utils.data.DataLoader(observationA_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
    coco_test_loader_shuffle_A = torch.utils.data.DataLoader(observationA_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=args.pin_memory)
    coco_test_loader_fix_B = torch.utils.data.DataLoader(observationB_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
    coco_test_loader_shuffle_B = torch.utils.data.DataLoader(observationB_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=args.pin_memory)

    print("observationA_dataset:", len(observationA_dataset))

    communication_field = CommunicationField(exp_name, agentA, agentB, Whole_iter=args.Whole_iter, MH_iter=args.MH_iter)
    communication_field.communication_field_setup(coco_test_loader_fix_A, coco_test_loader_shuffle_A, coco_test_loader_fix_B, coco_test_loader_shuffle_B, mode=args.mode)
    save_args_to_json(args, filename="args.json", save_dir=f"exp/{exp_name}")


    # Initialize the buffer for DER++
    agentA.initialize_td_buffer(coco_a_pretrain_loader, buffer_size=args.buffer_size)
    agentA.initialize_te_buffer(coco_a_pretrain_loader, buffer_size=args.buffer_size)

    agentB.initialize_td_buffer(coco_b_pretrain_loader, buffer_size=args.buffer_size)
    agentB.initialize_te_buffer(coco_b_pretrain_loader, buffer_size=args.buffer_size)


    # 並列処理のための設定
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(find_open_port())
    mp.set_start_method('spawn', force=True)

    if args.all_acceptance_agent is not None:
        communication_field.only_update_text_decoder(fine_tune_agent=args.all_acceptance_agent)
    elif args.distillation_agent is not None:
        communication_field.update_distillation(distillation_agent=args.distillation_agent)
    else:
        communication_field.mhcg()
