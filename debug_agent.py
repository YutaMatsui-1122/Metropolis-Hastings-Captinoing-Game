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
parser.add_argument('--te_update_epochs', default=10, type=int)
parser.add_argument('--buffer_size', default=10000, type=int)
parser.add_argument('--num_workers', default=1, type=int)
parser.add_argument('--device', default="cuda:0")
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--temperature', default=0.62, type=float)
parser.add_argument('--agentA_te_alpha_beta', default=0.5, type=float)
parser.add_argument('--agentA_td_alpha_beta', default=0.5, type=float)
parser.add_argument('--agentB_te_alpha_beta', default=0.5, type=float)
parser.add_argument('--agentB_td_alpha_beta', default=0.5, type=float)
parser.add_argument('--pin_memory', default=True)
parser.add_argument('--annealing', default="None")
parser.add_argument('--mode', default="MHNG")
parser.add_argument('--import_args', default="None") # import args from saved json file for same configuration
args = parser.parse_args()

exp_name = args.exp_name

print("expertiment name:", exp_name)

agentA = OneAgent(agent_name='A', device=args.device, td_update_epochs=args.td_update_epochs, te_update_epochs=args.te_update_epochs, temperature=args.temperature, te_alpha_beta=args.agentA_te_alpha_beta, td_alpha_beta=args.agentA_td_alpha_beta, clip_arch="ViT-B/32")
agentA = agentA.to(args.device)

agentA.load_pretrain(probvlm_path="models/official_model/probvlm/CC3M/probvlm_0.2_0.3_20-epoch-15.pth", clipcap_path="models/official_model/clipcap_conceptual_weights.pt", strict_clipcap=False)

if "debug" in exp_name:
    observation_file = "communication_coco_50_cc3m_50"
    args.buffer_size = 50
else:
    observation_file = f"communication_coco_5000_cc3m_5000"
    # observation_file = "communication_coco_0_cc3m_10000"
with open(f"dataset/dataset_cache/{observation_file}.pkl", "rb") as f:
    observationA_dataset = pickle.load(f)
    observationA_dataset.prefix_length = agentA.prefix_length

conceptual_pretrain_file = "conceptual_train_dataset_30000"
with open(f"dataset/dataset_cache/{conceptual_pretrain_file}.pkl", "rb") as f:
    conceptual_pretrain_dataset = pickle.load(f)
    conceptual_pretrain_dataset.prefix_length = 10

coco_pretrain_file = "coco_train_dataset_30000"
with open(f"dataset/dataset_cache/{coco_pretrain_file}.pkl", "rb") as f:
    coco_pretrain_dataset = pickle.load(f)
    coco_pretrain_dataset.prefix_length = 10

observationB_dataset = copy.deepcopy(observationA_dataset)

print("observationA_dataset:", len(observationA_dataset))

conceptual_pretrain_loader = torch.utils.data.DataLoader(conceptual_pretrain_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
coco_pretrain_loader = torch.utils.data.DataLoader(coco_pretrain_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

coco_test_loader_fix_A = torch.utils.data.DataLoader(observationA_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
coco_test_loader_shuffle_A = torch.utils.data.DataLoader(observationA_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=args.pin_memory)
coco_test_loader_fix_B = torch.utils.data.DataLoader(observationB_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
coco_test_loader_shuffle_B = torch.utils.data.DataLoader(observationB_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=args.pin_memory)

agentA.initialize_te_buffer(conceptual_pretrain_loader, buffer_size=args.buffer_size)

agentA.communication_field_setup(coco_test_loader_fix_A, coco_test_loader_shuffle_A, args.MH_iter, args.annealing, args.mode)
agentA.save_dir = f"exp/{exp_name}"
os.makedirs(agentA.save_dir, exist_ok=True)

agentA.perception()
agentA.update_text_encoder(0)