import torch
from torchvision import datasets, transforms
import os


def get_aug_transform(normalize_factor):
    return  transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(normalize_factor[0], normalize_factor[1]),
        ])

def get_org_transform (normalize_factor):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalize_factor[0], normalize_factor[1]),
        ])

def get_normalize_factor(dataset):
    if dataset=='cifar10':
        return [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)]
    elif dataset=='cifar100':
        return [(0.507, 0.487, 0.441), (0.267, 0.256, 0.276)]
    elif dataset=='tinyimg':
        return [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]


def get_transform(dataset, is_train=True, augmentation=True):
    normalize_factor = get_normalize_factor(dataset)
    if is_train:
        if augmentation:
            return get_aug_transform(normalize_factor)
        else:
            return get_org_transform(normalize_factor)
    else:
        return  get_org_transform(normalize_factor)


def get_dataset(dataset, data_path, is_train = True):
    if dataset=='cifar10':
        return datasets.CIFAR10(data_path, train=is_train, download=True, 
                transform=get_transform(dataset, is_train))
    elif dataset=='cifar100':
        return datasets.CIFAR100(data_path, train=is_train, download=True, 
                transform=get_transform(dataset, is_train))
    elif dataset=='tinyimg':
        if is_train:
            datadir = os.path.join(data_path, 'train')
        else:
            datadir = os.path.join(data_path, 'val/images')

        return datasets.ImageFolder(datadir, transform=get_transform(dataset, is_train))

def get_data_props(dataset):
    if dataset=='cifar10':
        return 3, 10, 32
    elif dataset=='cifar100':
        return 3, 100, 32
    elif dataset=='tinyimg':
        return 3, 200, 32

    
        





