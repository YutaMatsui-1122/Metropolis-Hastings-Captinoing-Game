class CommunicationField():
    def __init__(self, exp_name, agentA, agentB, EM_iter=10, MH_iter=100, Whole_iter=10):
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
    
    def communication_field_setup(self,dataloader_MHNG_fix_A, dataloader_MHNG_shuffle_A, dataloader_MHNG_fix_B, dataloader_MHNG_shuffle_B, annealing="None", mode="MHNG"):
        self.agentA.communication_field_setup(dataloader_MHNG_fix_A, dataloader_MHNG_shuffle_A, self.MH_iter, annealing, mode)
        self.agentB.communication_field_setup(dataloader_MHNG_fix_B, dataloader_MHNG_shuffle_B, self.MH_iter, annealing, mode)
        self.initialize_agent()
    
    def initialize_agent(self):
        self.agentA.initialize_sign()
        self.agentA.lora_setting()
        self.agentB.initialize_sign()
        self.agentB.lora_setting()
    
    def only_judge(self): # time test
        import time
        self.agentA.perception()
        self.agentB.perception()

        for em_iter in range(self.EM_iter):
            print("EM iteration: ", em_iter)
            
            for mh_iter in range(self.MH_iter):
                print("MHCG iteration: ", mh_iter)
                proposed_w_A = self.agentA.propose()
                self.agentB.judge(proposed_w_A, mh_iter)
                proposed_w_B = self.agentB.propose()
                self.agentA.judge(proposed_w_B, mh_iter)
                
    def gemcg(self):
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
            for mh_iter in range(self.MH_iter):
                print("MHCG iteration: ", mh_iter)
                proposed_w_A = self.agentA.propose()
                ar_B = self.agentB.judge(proposed_w_A, mh_iter)
                acceptance_rate_B.append(ar_B)

                proposed_w_B = self.agentB.propose()
                ar_A = self.agentA.judge(proposed_w_B, mh_iter)
                acceptance_rate_A.append(ar_A)

                # transform the sign to the text
                proposed_caption_A = []
                proposed_caption_B = []
                for i in range(len(proposed_w_A)):
                    # proposed_caption_A = tokenizer_decode(proposed_w_A[i])
                    proposed_caption_A.append(tokenizer_decode(proposed_w_A[i]))
                    proposed_caption_B.append(tokenizer_decode(proposed_w_B[i]))

                # save proposed sign with pandas
                df_A = pd.DataFrame(proposed_caption_A)
                df_B = pd.DataFrame(proposed_caption_B)
                df_A.to_csv(f"{self.agentA.save_dir}/proposed_sign_{em_iter}_{mh_iter}.csv")
                df_B.to_csv(f"{self.agentB.save_dir}/proposed_sign_{em_iter}_{mh_iter}.csv")
                self.agentA.save_sign(status=f"EM_{em_iter}_MH_{mh_iter}")
                self.agentB.save_sign(status=f"EM_{em_iter}_MH_{mh_iter}")
            
            # save the acceptance rate with txt file
            with open(f"{self.agentA.save_dir}/acceptance_rate_{em_iter}.txt", "w") as f:
                f.write("\n".join(map(str, acceptance_rate_A)))
            with open(f"{self.agentB.save_dir}/acceptance_rate_{em_iter}.txt", "w") as f:
                f.write("\n".join(map(str, acceptance_rate_B)))
                
            # save the sign generated by the last MH communication
            self.agentA.save_sign(status=f"EM_{em_iter}")
            self.agentB.save_sign(status=f"EM_{em_iter}")
            
            # Generalized M step
            # Each agent updates the text encoder based on the sign generated by MH communication (今はMH系列の最後のサインだけを使っている)
            self.agentA.update_text_encoder(em_iter)
            self.agentB.update_text_encoder(em_iter)
            
            # Each agent updates text decoder based on the sign generated by MH communication (今はMH系列の最後のサインだけを使っている)
            self.agentA.update_text_decoder(em_iter)
            self.agentB.update_text_decoder(em_iter)

if __name__ == '__main__':
    from one_agent import OneAgent
    from utils import * 
    import pickle, argparse, copy

    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    parser.add_argument('--exp_name', default="debug")
    parser.add_argument('--exp_num', default=1, type=int)
    parser.add_argument('--MH_iter', default=10, type=int)
    parser.add_argument('--EM_iter', default=10, type=int)
    parser.add_argument('--Whole_iter', default=10, type=int)
    parser.add_argument('--td_update_epochs', default=10, type=int)
    parser.add_argument('--te_update_epochs', default=3, type=int)
    parser.add_argument('--buffer_size', default=10000, type=int)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--device', default="cuda:0")
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--temperature', default=0.62, type=float)
    parser.add_argument('--agentA_alpha', default=0.5, type=float)
    parser.add_argument('--agentA_beta', default=0.5, type=float)
    parser.add_argument('--agentB_alpha', default=0.5, type=float)
    parser.add_argument('--agentB_beta', default=0.5, type=float)   
    parser.add_argument('--annealing', default="None")
    parser.add_argument('--mode', default="MHNG")
    parser.add_argument('--import_args', default="None") # import args from saved json file for same configuration
    args = parser.parse_args()

    exp_name = args.exp_name

    if args.import_args != "None":
        with open(f"exp/{args.import_args}/args.json", "r") as f:
            import_args = json.load(f)
        args = argparse.Namespace(**import_args)
        args.exp_name = exp_name # overwrite exp_name
    
    print("expertiment name:", exp_name)

    for i in range(1, args.exp_num+1):
        print("exp:", i)
        exp_name = args.exp_name + f"_{i}"
        # agentA = OneAgent(agent_name='A', device=args.device, td_update_epochs=args.td_update_epochs, te_update_epochs=args.te_update_epochs, temperature=args.temperature, der_alpha=args.alpha, derpp_beta=args.beta)
        # agentB = OneAgent(agent_name='B', device=args.device, td_update_epochs=args.td_update_epochs, te_update_epochs=args.te_update_epochs, temperature=args.temperature, der_alpha=args.alpha, derpp_beta=args.beta)   
        agentA = OneAgent(agent_name='A', device=args.device, td_update_epochs=args.td_update_epochs, te_update_epochs=args.te_update_epochs, temperature=args.temperature, der_alpha=args.agentA_alpha, derpp_beta=args.agentA_beta)
        agentB = OneAgent(agent_name='B', device=args.device, td_update_epochs=args.td_update_epochs, te_update_epochs=args.te_update_epochs, temperature=args.temperature, der_alpha=args.agentB_alpha, derpp_beta=args.agentB_beta)
        agentA = agentA.to(args.device)
        agentB = agentB.to(args.device)

        agentA.load_pretrain(probvlm_path="models/probVLM_conceptual_prefix-035.pth", clipcap_path="models/official_model/clipcap_conceptual_weights.pt", strict_clipcap=False)
        agentB.load_pretrain(probvlm_path="models/probVLM_coco_prefix-035.pth", clipcap_path="models/official_model/clipcap_coco_weights.pt", strict_clipcap=False)

        if "debug" in exp_name:
            observation_file = "communication_coco_50_cc3m_50"
            args.buffer_size = 20
        else:
            observation_file = f"communication_coco_5000_cc3m_5000"
            # observation_file = "communication_coco_0_cc3m_10000"
        with open(f"dataset/dataset_cache/{observation_file}.pkl", "rb") as f:
            observationA_dataset = pickle.load(f)
            observationA_dataset.prefix_length = agentA.prefix_length
        
        conceptual_pretrain_file = "conceptual_train_dataset_30000"
        with open(f"dataset/dataset_cache/{conceptual_pretrain_file}.pkl", "rb") as f:
            conceptual_pretrain_dataset = pickle.load(f)
            conceptual_pretrain_dataset.prefix_length = agentA.prefix_length
        
        coco_pretrain_file = "coco_train_dataset_30000"
        with open(f"dataset/dataset_cache/{coco_pretrain_file}.pkl", "rb") as f:
            coco_pretrain_dataset = pickle.load(f)
            coco_pretrain_dataset.prefix_length = agentB.prefix_length
        
        observationB_dataset = copy.deepcopy(observationA_dataset)

        print("observationA_dataset:", len(observationA_dataset))

        conceptual_pretrain_loader = torch.utils.data.DataLoader(conceptual_pretrain_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        coco_pretrain_loader = torch.utils.data.DataLoader(coco_pretrain_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        coco_test_loader_fix_A = torch.utils.data.DataLoader(observationA_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        coco_test_loader_shuffle_A = torch.utils.data.DataLoader(observationA_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        coco_test_loader_fix_B = torch.utils.data.DataLoader(observationB_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        coco_test_loader_shuffle_B = torch.utils.data.DataLoader(observationB_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        communication_field = CommunicationField(exp_name, agentA, agentB, EM_iter=args.EM_iter, MH_iter=args.MH_iter, Whole_iter=args.Whole_iter)
        communication_field.communication_field_setup(coco_test_loader_fix_A, coco_test_loader_shuffle_A, coco_test_loader_fix_B, coco_test_loader_shuffle_B, annealing=args.annealing, mode=args.mode)
        # save_args_to_json(args, filename="config.json", save_dir="save_models"):
        save_args_to_json(args, filename="args.json", save_dir=f"exp/{exp_name}")

        agentA.initialize_td_buffer(conceptual_pretrain_loader, buffer_size=args.buffer_size)
        agentB.initialize_td_buffer(coco_pretrain_loader, buffer_size=args.buffer_size)

        communication_field.gemcg()
        # communication_field.only_judge()