# -*- coding: utf-8 -*-
"""
Created on Aug 24 2019
"""
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

from models import *

from optimizers import BayesBiNN as BayesBiNN
from optimizers import FenBPOpt
from optimizers import FenBPOptQuad
from optimizers import FenBPOptProx

from utils import plot_result, train_model, SquaredHingeLoss, save_train_history
import numpy as np

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import time
from torchvision import datasets, transforms
import optuna 

def timeSince(since):
    now = time.time()
    s = now - since
    return s

def main(trial):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Cifar10 Example')
    # Model parameters
    parser.add_argument('--model', type=str, default='VGGBinaryConnect', help='Model name: VGGBinaryConnect, VGGBinaryConnect_STE')
    parser.add_argument('--bnmomentum', type=float, default=0.2, help='BN layer momentum value')

    # Optimization parameters
    parser.add_argument('--optim', type=str, default='FenBP', help='Optimizer: BayesBiNN, STE, Adam')
    parser.add_argument('--val-split', type=float, default=0.1, help='Random validation set ratio')
    parser.add_argument('--criterion', type=str, default='cross-entropy', help='loss funcion: square-hinge or cross-entropy')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--train-samples', type=int,default=1, metavar='N',
                        help='number of Monte Carlo samples used in BayesBiNN (default: 1), if 0, point estimate using mean')
    parser.add_argument('--test-samples', type=int,default=0, metavar='N',
                        help='number of Monte Carlo samples used in evaluation for BayesBiNN (default: 1)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default= 3e-4, metavar='LR',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--lr-end', type=float, default= 1e-16, metavar='LR',
                        help='end learning rate (default: 0.01)')
    parser.add_argument('--lr-decay', type=float, default= 0.9, metavar='LR-decay',
                        help='learning rated decay factor for each epoch (default: 0.9)')
    parser.add_argument('--decay-steps', type=int, default=1, metavar='N',
                        help='LR rate decay steps (default: 1)')   

    parser.add_argument('--momentum', type=float, default=0.0, metavar='M',
                        help='BayesBiNN momentum (default: 0.9)')
    parser.add_argument('--data-augmentation', action='store_true', default=True, help='Enable data augmentation')
    # Logging parameters
    parser.add_argument('--log-interval', type=int, default=400, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--experiment-id', type=int, default=0, help='Experiment ID for log files (int)')
    # Computation parameters
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=100, metavar='S',
                        help='random seed (default: 10)')

    parser.add_argument('--lrschedular', type=str, default='Cosine', help='Mstep,Expo,Cosine')


    parser.add_argument('--trainset_scale', type=int, default=10, metavar='N',
                        help='scale of the training set used in data augmentation')


    parser.add_argument('--lamda', type=float, default= 10, metavar='lamda-init',
                        help='initial mean value of the natural parameter lamda(default: 10)')

    parser.add_argument('--lamda-std', type=float, default= 0, metavar='lamda-init',
                        help='linitial std value of the natural parameter lamda(default: 0)')

    parser.add_argument('--temperature', type=float, default= 1e-10, metavar='temperature',
                        help='temperature for BayesBiNN')

    parser.add_argument('--bn-affine', type=float, default= 0, metavar='bn-affine',
                        help='whether there is bn learnable parameters, 1: learnable, 0: no (default: 0)')


    args = parser.parse_args()

    #Define configs for optuna 
    cfg = {'lr': trial.suggest_loguniform('lr', 1e-6, 1e-3),
            'momentum': trial.suggest_uniform('momentum', 0.11, 0.99),
            'fenbp_alpha': trial.suggest_uniform('fenbp_alpha', 0.001, 0.1)
            }
    optuna_train_size = 5000
    # optuna_test_size = 1000

    if args.lr_decay > 1:
        raise ValueError('The end learning rate should be smaller than starting rate!! corrected')

    args.use_cuda = not args.no_cuda and torch.cuda.is_available()
    ngpus_per_node = torch.cuda.device_count()
    gpu_num = []
    print("Number of GPUs:%d", ngpus_per_node)
    gpu_devices = ','.join([str(id) for id in gpu_num])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

    if ngpus_per_node > 0:
        print("Use GPU: {} for training".format(gpu_devices))

    # torch.manual_seed(args.seed + args.experiment_id)
    # np.random.seed(args.seed + args.experiment_id)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)

    now = time.strftime("%Y_%m_%d_%H_%M_%S",time.localtime(time.time())) # to avoid overwrite
    args.out_dir = os.path.join('./outputs', 'cifar10_{}_{}_lr{}_{}_id{}'.format(args.model, args.optim,args.lr,now,args.experiment_id))
    os.makedirs(args.out_dir, exist_ok=True)

    config_save_path = os.path.join(args.out_dir, 'configs', 'config_{}.json'.format(args.experiment_id))
    os.makedirs(os.path.dirname(config_save_path), exist_ok=True)
    with open(config_save_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    args.device = torch.device("cuda" if args.use_cuda else "cpu")
    print('Running on', args.device)
    print('===========================')
    for key, val in vars(args).items():
        print('{}: {}'.format(key, val))
    print('===========================\n')


    # Data augmentation for cifar10
    if args.data_augmentation:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Defining the dataset
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.use_cuda else {}
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)

    if args.val_split > 0 and args.val_split < 1:
        val_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_test)

        num_train = len(train_dataset)

        indices = list(range(num_train))
        split = int(np.floor(args.val_split * num_train))
        np.random.shuffle(indices)

        #train_idx, val_idx = indices[split:], indices[:split]
        train_idx, val_idx = indices[num_train-optuna_train_size:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, sampler=val_sampler, **kwargs
        )
        print('{} train and {} validation datapoints.'.format(len(train_loader.sampler), len(val_loader.sampler)))
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
        )
        val_loader = None
        print('{} train and {} validation datapoints.'.format(len(train_loader.sampler), 0))

    test_dataset = datasets.CIFAR10('./data', train=False, transform=transform_test)
    num_test = len(test_dataset)
    indices = list(range(num_test))
    np.random.shuffle(indices)
    # test_idx = indices[0:optuna_test_size]
    # test_sampler = SubsetRandomSampler(test_idx)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size, #sampler=test_sampler,
        **kwargs
    )
    print('{} test datapoints.\n'.format(len(test_loader.sampler)))


    # Defining the model.
    in_channels, out_features = 3, 10
    if args.model == 'VGGBinaryConnect': #
        model = VGGBinaryConnect(in_channels, out_features, eps=1e-5, momentum=args.bnmomentum,batch_affine=(args.bn_affine==1))
    elif args.model == 'VGGBinaryConnect_STE':
        model = VGGBinaryConnect_STE(in_channels, out_features, eps=1e-5, momentum=args.bnmomentum,
                                     batch_affine=(args.bn_affine == 1))
    elif args.model == 'RESNET18':
        model = ResNet18()
    elif args.model == 'VGG16': #
        model = models.VGG16(in_channels, out_features, eps=1e-5, momentum=args.bnmomentum,batch_affine=(args.bn_affine==1))
    else:
        raise ValueError('Undefined Network')
    print(model)

    num_parameters = sum([l.nelement() for l in model.parameters()])
    print("Number of Network parameters: {}".format(num_parameters))

    # model = torch.nn.DataParallel(model,device_ids=gpu_num)
    model = model.to(args.device)

    # Initialize model
    # for n,p in model.parameters():
    for n, p in model.named_parameters():
        if len(p.size())>=2:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.normal_(p, std=0.1)
    cudnn.benchmark = True


    # Defining the optimizer
    if args.optim == 'Adam' or args.optim == 'STE':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    elif args.optim == 'BayesBiNN':
        effective_trainsize = len(train_loader.sampler) * args.trainset_scale
        optimizer = BayesBiNN(model,lamda_init = args.lamda,lamda_std = args.lamda_std,  temperature = args.temperature, train_set_size=effective_trainsize, lr=args.lr, betas=args.momentum, num_samples=args.train_samples)
    elif args.optim == 'FenBP':
        effective_trainsize = len(train_loader.sampler) * args.trainset_scale
        optimizer=FenBPOpt(model,train_set_size=effective_trainsize, 
                lr = cfg['lr'],
                use_STE = False,
                betas=cfg['momentum'], 
                fenbp_alpha=cfg['fenbp_alpha'],
                fenbp_beta = 1./cfg['fenbp_alpha'] + 100
                )

    # Defining the criterion
    if args.criterion == 'square-hinge':
        criterion = SquaredHingeLoss() # use the squared hinge loss for MNIST dataset
    elif args.criterion == 'cross-entropy':
        criterion = nn.CrossEntropyLoss() # this loss depends on the model output, remember to change the model output
    else:
        raise ValueError('Please select loss criterion in {square-hinge, cross-entropy}')

    start = time.time()

    # Training the model
    results = train_model(args, model, [train_loader, val_loader, test_loader], criterion, optimizer)
    model, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc = results
    save_train_history(args, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc)
    # plot_result(args, train_loss, train_acc, test_loss, test_acc)

    time_total=timeSince(start)

    print('Task completed in {:.0f}m {:.0f}s'.format(
        time_total // 60, time_total % 60))

    return max(test_acc)


if __name__ == '__main__':
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(sampler = sampler, direction='maximize')
    study.optimize(func = main, n_trials= 100)
    best_trial = study.best_trial
    print('Best value: {}'.format(best_trial.value))
    for key, value in best_trial.params.items():
        print('{} : {} :'.format(key, value))
    # main()
