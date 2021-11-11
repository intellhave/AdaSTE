import os
import json 
import argparse
import numpy as np 

import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data.sampler import SubsetRandomSampler

# Optimizers
from optimizers import BayesBiNN as BayesBiNN
from optimizers import FenBPOpt

# Utils 
from utils import train_model, SquaredHingeLoss, save_train_history, load_model
from utils import get_parser, get_transform, get_dataset, get_data_props
import numpy as np

from models import *

import torch.nn.parallel
import torch.backends.cudnn as cudnn 
import time 
from torchvision import datasets, transforms

def main():
    # Obtain input paramters 
    parser = get_parser()
    args = parser.parse_args()

    # CUDA stuffs
    args.use_cuda = not args.no_cuda and torch.cuda.is_available()
    ngpus_per_node = torch.cuda.device_count()
    gpu_num = []
    print("Number of GPUs:%d", ngpus_per_node)
    gpu_devices = ','.join([str(id) for id in gpu_num])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
    if ngpus_per_node > 0:
        print("Use GPU: {} for training".format(gpu_devices))

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)

    #Prepare output folder
    now = time.strftime("%Y_%m_%d_%H_%M_%S",time.localtime(time.time())) # to avoid overwrite
    args.out_dir = os.path.join('./outputs', '{}_{}_{}_lr{}_b{}_{}_id{}'.format(args.dataset, args.model, args.optim,args.lr, args.init_beta, now,args.experiment_id))
    os.makedirs(args.out_dir, exist_ok=True)

    # Save the configs for the runs 
    config_save_path = os.path.join(args.out_dir, 'configs', 'config_{}.json'.format(args.experiment_id))
    os.makedirs(os.path.dirname(config_save_path), exist_ok=True)
    with open(config_save_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

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


    # Get data transformations
    transform_train = get_transform(args.dataset, is_train=True)
    transform_test = get_transform(args.dataset, is_train = False)

    train_dataset = get_dataset(args.dataset, args.data_path, is_train=True)
    test_dataset = get_dataset(args.dataset, args.data_path, is_train=False)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
    )
    print('{} train and {} validation datapoints.'.format(len(train_loader.sampler), 0))

    test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    print('{} test datapoints.\n'.format(len(test_loader.sampler)))

    in_channels, out_features, imgsize = get_data_props(args.dataset)
    
    # Prepare models 
    if args.model=='RESNET18':
        model = models.ResNet18(input_channels = in_channels, output_dim=out_features, imsize=imgsize)
    elif args.model=='VGG16':
        model = models.VGG16(in_channels, out_features, eps=1e-5, momentum=args.bnmomentum, batch_affine=(args.bn_affine==1))
    else:
        raise ValueError('Undefined Network')
    print(model)
    num_parameters = sum([l.nelement() for l in model.parameters()])
    print("Number of Network parameters: {}".format(num_parameters))

    # Move model to GPU
    model = torch.nn.DataParallel(model,device_ids=[0])
    # model = model.to(args.device)

    # Initialize model
    for p in model.parameters():
        if len(p.size())>=2:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.normal_(p, std=0.1)

    #Prepare optimizers
    if args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim=='STE':
        effective_trainsize = len(train_loader.sampler) * args.trainset_scale
        optimizer=FenBPOpt(model,train_set_size=effective_trainsize, 
                delta = 1e-6,
                lr = args.lr,
                use_STE = True,
                betas = args.momentum,
                fenbp_beta = args.init_beta,
                fenbp_alpha = args.alpha
                )

    elif args.optim == 'BayesBiNN':
        effective_trainsize = len(train_loader.sampler) * args.trainset_scale
        optimizer = BayesBiNN(model,lamda_init = args.lamda,lamda_std = args.lamda_std,  temperature = args.temperature, train_set_size=effective_trainsize, lr=args.lr, betas=args.momentum, num_samples=args.train_samples)
    elif args.optim == 'FenBP':
        effective_trainsize = len(train_loader.sampler) * args.trainset_scale
        optimizer=FenBPOpt(model,train_set_size=effective_trainsize, 
                delta = 1e-6,
                lr = args.lr,
                use_STE = False,
                betas = args.momentum,
                fenbp_beta = args.init_beta,
                fenbp_alpha = args.alpha
                )

    # Defining the criterion
    if args.criterion == 'square-hinge':
        criterion = SquaredHingeLoss() # use the squared hinge loss for MNIST dataset
    elif args.criterion == 'cross-entropy':
        criterion = nn.CrossEntropyLoss() 
    else:
        raise ValueError('Please select loss criterion in {square-hinge, cross-entropy}')

    start = time.time()

    # Training the model
    results = train_model(args, model, [train_loader, None, test_loader], criterion, optimizer)
    model, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc = results
    save_train_history(args, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc)
    # plot_result(args, train_loss, train_acc, test_loss, test_acc)

    time_total=time.time() - start
    print('Task completed in {:.0f}m {:.0f}s'.format(
        time_total // 60, time_total % 60))

if __name__ == '__main__':
    main()














    





















