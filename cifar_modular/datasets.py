from torchvision import datasets as dsets, transforms
from torch.utils.data import DataLoader

def get_cifar10(data_dir='/data', batch_size=128):
    t_tr = transforms.Compose([
        transforms.RandomCrop(32, padding=4), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    t_te = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_loader = DataLoader(dsets.CIFAR10(data_dir, train=True, download=True, transform=t_tr), batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(dsets.CIFAR10(data_dir, train=False, download=True, transform=t_te), batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, 10

def get_cifar100(data_dir='/data', batch_size=128):
    mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
    t_tr = transforms.Compose([
        transforms.RandomCrop(32, padding=4), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize(mean, std)
    ])
    t_te = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean, std)
    ])
    
    train_loader = DataLoader(dsets.CIFAR100(data_dir, train=True, download=True, transform=t_tr), batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(dsets.CIFAR100(data_dir, train=False, download=True, transform=t_te), batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, 100
