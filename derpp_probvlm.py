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
parser.add_argument('--mode', default="None", choices=("None", "DER", "DERPP"))
parser.add_argument('--basemodel', default="coco", choices=("coco", "cc3m"))
args = parser.parse_args()

if "debug" in args.exp_name:
    args.buffer_size = 50
    observation_file = "communication_coco_50_cc3m_50"
    probvlm_path = "models/official_model/probvlm/CC3M/probvlm_0.2_0.3_20-epoch-15.pth"
    exp_name = args.exp_name

elif args.basemodel == "coco":
    observation_file = "communication_coco_0_cc3m_10000"
    probvlm_path = "models/official_model/probvlm/COCO/probvlm_0.2_0.3_20-epoch-99.pth"
    exp_name = args.exp_name + "_coco2cc3m"
    
elif args.basemodel == "cc3m":
    observation_file = "communication_coco_10000_cc3m_0"
    probvlm_path = "models/official_model/probvlm/CC3M/probvlm_0.2_0.3_20-epoch-15.pth"
    exp_name = args.exp_name + "_cc3m2coco" 

for i in range(1, args.exp_num+1):
    print("exp:", i)    
    agentA = OneAgent(agent_name='A', device=args.device, td_update_epochs=args.td_update_epochs, te_update_epochs=args.te_update_epochs, temperature=args.temperature, der_alpha=args.agentA_alpha, derpp_beta=args.agentA_beta)
    agentA = agentA.to(args.device)

    # agentA.load_pretrain(probvlm_path="models/official_model/probvlm/CC3M/probvlm_0.2_0.3_20-epoch-15.pth", clipcap_path="models/official_model/clipcap_conceptual_weights.pt", strict_clipcap=False)
    agentA.load_pretrain(probvlm_path=probvlm_path, clipcap_path="models/official_model/clipcap_conceptual_weights.pt", strict_clipcap=False)

    with open(f"dataset/dataset_cache/{observation_file}.pkl", "rb") as f:
        observationA_dataset = pickle.load(f)
        observationA_dataset.prefix_length = agentA.prefix_length
    
    conceptual_pretrain_file = "conceptual_train_dataset_10000"
    with open(f"dataset/dataset_cache/{conceptual_pretrain_file}.pkl", "rb") as f:
        conceptual_pretrain_dataset = pickle.load(f)
        conceptual_pretrain_dataset.prefix_length = agentA.prefix_length

    coco_pretrain_file = "coco_train_dataset_10000"
    with open(f"dataset/dataset_cache/{coco_pretrain_file}.pkl", "rb") as f:
        coco_pretrain_dataset = pickle.load(f)
        coco_pretrain_dataset.prefix_length = agentA.prefix_length

    print("observationA_dataset:", len(observationA_dataset))
    print("Observation file:", observation_file)

    conceptual_pretrain_loader = torch.utils.data.DataLoader(conceptual_pretrain_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    coco_pretrain_loader = torch.utils.data.DataLoader(coco_pretrain_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    coco_test_loader_fix_A = torch.utils.data.DataLoader(observationA_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    coco_test_loader_shuffle_A = torch.utils.data.DataLoader(observationA_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    agentA.dataloader_MHNG_fix = coco_test_loader_fix_A
    agentA.dataloader_MHNG_shuffle = coco_test_loader_shuffle_A

    # print some samples
    batch = next(iter(coco_test_loader_fix_A))
    print("caption:", batch["caption"])

    if args.basemodel == "coco":
        agentA.initialize_te_buffer(coco_pretrain_loader,buffer_size=args.buffer_size)
    elif args.basemodel == "cc3m":
        agentA.initialize_te_buffer(conceptual_pretrain_loader,buffer_size=args.buffer_size)

    agentA.perception()
    agentA.save_dir = f"models/{exp_name}"
    os.makedirs(agentA.save_dir, exist_ok=True)

    from update_models import *

    learning_mode = args.mode
    # updated_probvlm = update_probvlm(self.z, self.CLIP_Net, self.ProbVLM_Net, self.dataloader_MHNG_shuffle, self.save_dir, epochs = self.te_update_epochs, lr=1e-6, device=self.device, output_prefix="probvlm_"+self.agent_name+f"_{em_epoch}", save_every=5)
    probvlm = update_probvlm_derpp(agentA.z, agentA.CLIP_Net, agentA.ProbVLM_Net, agentA.dataloader_MHNG_shuffle, agentA.save_dir, epochs = agentA.te_update_epochs, lr=1e-5, device=agentA.device, output_prefix="probvlm_"+agentA.agent_name, save_every=1, buffer=agentA.te_buffer, alpha=agentA.der_alpha, beta=agentA.derpp_beta, train_mode =learning_mode) #, pretrain_test_loader=conceptual_pretrain_loader, fine_tune_test_loader=coco_pretrain_loader)