import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms, models
from torchvision.models.resnet import ResNet18_Weights
import torch.distributed as dist
from time import time

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank + 2)  # Offset by 2 to use cuda:2 and cuda:3

def cleanup():
    dist.destroy_process_group()

class CIFARResNet(nn.Module):
    def __init__(self):
        super(CIFARResNet, self).__init__()
        original_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            *list(original_model.children())[1:-1]
        )
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def train_model(rank, world_size, epochs=5):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank + 2}")  # Offset by 2 to use cuda:2 and cuda:3

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler, shuffle=False)

    model = CIFARResNet().to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank + 2])
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    start_time = time()
    model.train()
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        if rank == 0:
            print(f'Rank {rank}, Epoch {epoch+1}, Loss: {loss.item():.4f}')

    total_time = time() - start_time
    cleanup()
    if rank == 0:
        print(f'Total Training Time: {total_time:.2f} seconds')

if __name__ == "__main__":
    world_size = 2  # 使用するGPU数を2に設定
    torch.multiprocessing.spawn(train_model, args=(world_size,), nprocs=world_size, join=True)