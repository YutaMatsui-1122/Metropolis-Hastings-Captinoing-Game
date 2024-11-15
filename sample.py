import torch

# train_loss_list.pt の読み込み
# train_loss_list = torch.load('train_loss_list.pt')
train_loss_list = torch.load('models/coco_dataset_a_vit16/train_loss_list.pt')

# 中身を出力
print(train_loss_list)
