import os
import torch.nn as nn
import torch.optim as optim
import torchvision
from .noisy_cifar import NoisyCIFAR10, NoisyCIFAR100



# dataset --------------------------------------------------------------------------------------------------------------------------------------------
def build_transform(rescale_size=512, crop_size=448, s=1):
    cifar_train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(size=32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                                 std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    ])
    cifar_test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                                 std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    ])

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=rescale_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(size=crop_size),
        # RandAugment(),
        # ImageNetPolicy(),
        # Cutout(size=crop_size // 16),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                                 std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=rescale_size),
        torchvision.transforms.CenterCrop(size=crop_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                                 std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    ])
    return {'train': train_transform, 'test': test_transform,
            'cifar_train': cifar_train_transform, 'cifar_test': cifar_test_transform}


def build_cifar100n_dataset(root, train_transform, test_transform, noise_type, openset_ratio, closeset_ratio):
    train_data = NoisyCIFAR100(root, train=True, transform=train_transform, download=True, noise_type=noise_type, closeset_ratio=closeset_ratio,
                               openset_ratio=openset_ratio, verbose=True)
    test_data = NoisyCIFAR100(root, train=False, transform=test_transform, download=True, noise_type='clean', closeset_ratio=closeset_ratio,
                              openset_ratio=openset_ratio, verbose=True)
    return {'train': train_data, 'test': test_data}


# optimizer, scheduler -------------------------------------------------------------------------------------------------------------------------------
def build_sgd_optimizer(params, lr, weight_decay):
    return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)


def build_adam_optimizer(params, lr):
    return optim.Adam(params, lr=lr, betas=(0.9, 0.999))


def build_cosine_lr_scheduler(optimizer, total_epochs):
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=0)

