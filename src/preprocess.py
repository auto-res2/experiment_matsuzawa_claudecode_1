import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Subset

def get_cifar10_data(batch_size=32, subset_size=1000):
    """Load and preprocess CIFAR-10 dataset with memory-efficient subset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform)
    
    train_subset = Subset(trainset, range(min(subset_size, len(trainset))))
    test_subset = Subset(testset, range(min(subset_size//2, len(testset))))
    
    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader

def normalize_tensor(x):
    """Normalize tensor to [0, 1] range."""
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

def denormalize_cifar10(tensor):
    """Denormalize CIFAR-10 tensor back to original range."""
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
    if tensor.is_cuda:
        mean = mean.cuda()
        std = std.cuda()
    return tensor * std + mean

def prepare_batch_for_gpu(batch, device):
    """Prepare batch for GPU processing with memory management."""
    if isinstance(batch, (list, tuple)):
        return [x.to(device) if torch.is_tensor(x) else x for x in batch]
    elif torch.is_tensor(batch):
        return batch.to(device)
    return batch
