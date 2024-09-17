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
args = parser.parse_args()

for i in range(1, args.exp_num+1):
    print("exp:", i)
    exp_name = args.exp_name + f"_{i}"
    # agentA = OneAgent(agent_name='A', device=args.device, td_update_epochs=args.td_update_epochs, te_update_epochs=args.te_update_epochs, temperature=args.temperature, der_alpha=args.alpha, derpp_beta=args.beta)
    # agentB = OneAgent(agent_name='B', device=args.device, td_update_epochs=args.td_update_epochs, te_update_epochs=args.te_update_epochs, temperature=args.temperature, der_alpha=args.alpha, derpp_beta=args.beta)   
    agentA = OneAgent(agent_name='A', device=args.device, td_update_epochs=args.td_update_epochs, te_update_epochs=args.te_update_epochs, temperature=args.temperature, der_alpha=args.agentA_alpha, derpp_beta=args.agentA_beta)
    agentA = agentA.to(args.device)

    agentA.load_pretrain(probvlm_path="models/probVLM_conceptual_prefix-035.pth", clipcap_path="models/official_model/clipcap_conceptual_weights.pt", strict_clipcap=False)
    
    if "debug" in exp_name:
        observation_file = "communication_coco_50_cc3m_50"
        args.buffer_size = 50
    else:
        observation_file = f"communication_coco_5000_cc3m_5000"

    with open(f"dataset/dataset_cache/{observation_file}.pkl", "rb") as f:
        observationA_dataset = pickle.load(f)
        observationA_dataset.prefix_length = agentA.prefix_length
    
    conceptual_pretrain_file = "conceptual_train_dataset_30000"
    with open(f"dataset/dataset_cache/{conceptual_pretrain_file}.pkl", "rb") as f:
        conceptual_pretrain_dataset = pickle.load(f)
        conceptual_pretrain_dataset.prefix_length = agentA.prefix_length

    print("observationA_dataset:", len(observationA_dataset))

    conceptual_pretrain_loader = torch.utils.data.DataLoader(conceptual_pretrain_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    coco_test_loader_fix_A = torch.utils.data.DataLoader(observationA_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    coco_test_loader_shuffle_A = torch.utils.data.DataLoader(observationA_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    agentA.initialize_te_buffer(conceptual_pretrain_loader,buffer_size=args.buffer_size)

    output = agentA.te_buffer.sample(10)

    for data in output:
        print(data[0][:10])

    from update_models import *

    probvlm = update_probvlm_derpp()