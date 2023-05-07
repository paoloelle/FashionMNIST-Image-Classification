import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

train_data = datasets.FashionMNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
train_data_reduced = torch.utils.data.Subset(train_data, list(range(0, 5000, 1)))

test_data = datasets.FashionMNIST(root='data', train=False, transform=transforms.ToTensor(), download=True)
test_data_reduced = torch.utils.data.Subset(test_data, list(range(0, 1000, 1)))
