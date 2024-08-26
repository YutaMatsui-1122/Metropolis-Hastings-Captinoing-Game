from utils import * 
import pickle, clip, argparse
from torch.nn import functional as nnf
import numpy as np
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
parser.add_argument('--exp_name', default="debug")
parser.add_argument('--MH_iter', default=100, type=int)
parser.add_argument('--annealing', default="None")
parser.add_argument('--mode', default="MHNG")
args = parser.parse_args()

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

clip_model, preprocess = clip.load(args.clip_model_type, device=device, jit=False)
data_path = 'dataset/'
normalize_prefix = True

# coco_test_dataset_A = CocoDataset(root = data_path, transform=preprocess,data_mode="test", prefix_length=prefix_length, normalize_prefix=normalize_prefix, datasize="10000", use_imageid=True)
with open("dataset/dataset_cache/coco_test_dataset_5000_imageid.pkl", "rb") as f:
    coco_test_dataset_A = pickle.load(f)
    coco_test_dataset_A.prefix_length = 10

with open("dataset/dataset_cache/conceptual_train_dataset_200000.pkl", "rb") as f:
# with open("dataset/dataset_cache/conceptual_test_dataset_300.pkl", "rb") as f:
    conceptual_test_dataset = pickle.load(f)
    conceptual_test_dataset.prefix_length = 10


print("coco_test_dataset_A:", len(coco_test_dataset_A))
print("conceptual_test_dataset:", len(conceptual_test_dataset))
coco_test_loader = torch.utils.data.DataLoader(coco_test_dataset_A, batch_size=64, shuffle=False, num_workers=1)
conceptual_test_loader = torch.utils.data.DataLoader(conceptual_test_dataset, batch_size=64, shuffle=False, num_workers=1)

image_embeddings = []
text_embeddings = []
for batch in tqdm.tqdm(coco_test_loader):
    image = batch[0].to(device)
    caption = tokenize(batch[1]).to(device)
    image_embed = clip_model.encode_image(image)
    text_embed = clip_model.encode_text(caption)

    image_embeddings.append(image_embed.cpu().detach())
    text_embeddings.append(text_embed.cpu().detach())

image_embeddings = torch.cat(image_embeddings, dim=0)
text_embeddings = torch.cat(text_embeddings, dim=0)

# calculate clip score
clip_score = nnf.cosine_similarity(image_embeddings, text_embeddings).cpu().detach().numpy()
print("CLIP score:", np.mean(clip_score))

image_embeddings = []
text_embeddings = []

for batch in tqdm.tqdm(conceptual_test_loader):
    image = batch[0].to(device)
    caption = tokenize(batch[1]).to(device)
    image_embed = clip_model.encode_image(image)
    text_embed = clip_model.encode_text(caption)

    image_embeddings.append(image_embed.cpu().detach())
    text_embeddings.append(text_embed.cpu().detach())

image_embeddings = torch.cat(image_embeddings, dim=0)
text_embeddings = torch.cat(text_embeddings, dim=0)

# calculate clip score
clip_score = nnf.cosine_similarity(image_embeddings, text_embeddings).cpu().detach().numpy()
print("CLIP score:", np.mean(clip_score))