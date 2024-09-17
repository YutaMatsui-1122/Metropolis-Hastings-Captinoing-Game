# import torch 

# eval_loss = torch.load("models/coco_cc3m_pretrain/eval_loss_list.pt")
# train_loss = torch.load("models/coco_cc3m_pretrain/train_loss_list.pt")

# print(train_loss)
# print(eval_loss)

import json
from pycocotools.coco import COCO

# COCOアノテーションファイルと画像ディレクトリのパス
coco_annotation_file = 'dataset/coco/annotations/annotations/captions_val2014.json'
images_dir = 'dataset/coco/val2014'

# COCOアノテーションの読み込み
coco = COCO(coco_annotation_file)

# 画像ファイル名を取得するヘルパー関数
def get_image_filename(image_id):
    image_info = coco.loadImgs(image_id)[0]
    return image_info['file_name'].split('.')[0]

# candidates.jsonの生成
candidates = {}
for image_id in coco.getImgIds():
    image_filename = get_image_filename(image_id)
    
    # 各画像の最初のキャプションをcandidateとして使用
    ann_ids = coco.getAnnIds(imgIds=image_id)
    anns = coco.loadAnns(ann_ids)
    candidates[image_filename] = anns[0]['caption']

# candidate.jsonファイルに保存
with open('candidate.json', 'w') as f:
    json.dump(candidates, f, indent=4)

# refs.jsonの生成
references = {}
for image_id in coco.getImgIds():
    image_filename = get_image_filename(image_id)
    
    # 各画像の全てのキャプションをリファレンスとして使用
    ann_ids = coco.getAnnIds(imgIds=image_id)
    anns = coco.loadAnns(ann_ids)
    references[image_filename] = [ann['caption'] for ann in anns]

# refs.jsonファイルに保存
with open('refs.json', 'w') as f:
    json.dump(references, f, indent=4)

print("candidate.json と refs.json が正常に生成されました。")
