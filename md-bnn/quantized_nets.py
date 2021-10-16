""" CODE Implementation MD View of NN Quantization
"""
from datetime import datetime
import logging
import os
import shutil
import numpy as np
from tensorboardX import SummaryWriter

import methods.continuous_net as connet
import methods.binary_connect as binary_connect
import methods.mda_softmax as md_softmax
import methods.mda_tanh as md_tanh
import methods.softmax_projn as softmax_projn
import methods.tanh_projn as tanh_projn
import methods.fenbp_tanh as fenbp_tanh

import utils.utils as util
import cfgs.cfg as cfg
import torch
import torch.nn as nn
import torch.utils.data as du
from torchvision import datasets, transforms
from utils.randomness_fix import seed_torch, init_fn


seed_torch()


def main():
    # Get the CL arguments
    # args = get_arguments()
    args = cfg.args

    # pytorch setup
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    #### setup results directory and logging
    if args.save_dir == cfg.SAVE_DIR:

        if args.quant_levels == 3:
            if args.full_ste:
                args.save_dir = os.path.join(args.save_dir, args.dataset, args.architecture, args.method + '_TNN_STE', args.optimizer + '_'
                                             + 'lr'+ str(args.learning_rate) + '_' + 'bts'+ str(args.beta_scale)
                                             + '_' + 'lrs' + str(args.lr_scale) + '_' + 'bti' + str(args.beta_interval),
                                             datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

                args.resume_file = os.path.join('./out', args.dataset, args.architecture, args.method + '_TNN_STE', args.optimizer + '_'
                                    + 'lr'+ str(args.learning_rate) + '_' + 'bts'+ str(args.beta_scale)
                                    + '_' + 'lrs' + str(args.lr_scale) + '_' + 'bti' + str(args.beta_interval),
                                      'checkpoint.pth.tar')

            else:
                args.save_dir = os.path.join(args.save_dir, args.dataset, args.architecture, args.method + '_TNN', args.optimizer + '_'
                                             + 'lr'+ str(args.learning_rate) + '_' + 'bts'+ str(args.beta_scale)
                                             + '_' + 'lrs' + str(args.lr_scale) + '_' + 'bti' + str(args.beta_interval),
                                             datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
                args.resume_file = os.path.join('./out', args.dataset, args.architecture, args.method + '_TNN', args.optimizer + '_'
                                     + 'lr'+ str(args.learning_rate) + '_' + 'bts'+ str(args.beta_scale)
                                     + '_' + 'lrs' + str(args.lr_scale) + '_' + 'bti' + str(args.beta_interval),
                                      'checkpoint.pth.tar')
        else:
            if args.full_ste:
                args.save_dir = os.path.join(args.save_dir, args.dataset, args.architecture, args.method + '_STE', args.optimizer + '_'
                                             + 'lr'+ str(args.learning_rate) + '_' + 'bts'+ str(args.beta_scale)
                                             + '_' + 'lrs' + str(args.lr_scale) + '_' + 'bti' + str(args.beta_interval),
                                             datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
                args.resume_file = os.path.join('./out', args.dataset, args.architecture, args.method + '_STE', args.optimizer + '_'
                                      + 'lr'+ str(args.learning_rate) + '_' + 'bts'+ str(args.beta_scale)
                                      + '_' + 'lrs' + str(args.lr_scale) + '_' + 'bti' + str(args.beta_interval),
                                      'checkpoint.pth.tar')
            else:
                args.save_dir = os.path.join(args.save_dir, args.dataset, args.architecture, args.method, args.optimizer + '_'
                                             + 'lr'+ str(args.learning_rate) + '_' + 'bts'+ str(args.beta_scale)
                                             + '_' + 'lrs' + str(args.lr_scale) + '_' + 'bti' + str(args.beta_interval),
                                             datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
                args.resume_file = os.path.join('./out', args.dataset, args.architecture, args.method, args.optimizer + '_'
                                      + 'lr'+ str(args.learning_rate) + '_' + 'bts'+ str(args.beta_scale)
                                      + '_' + 'lrs' + str(args.lr_scale) + '_' + 'bti' + str(args.beta_interval),
                                      'checkpoint.pth.tar')

    if args.eval != cfg.EVAL:
        args.save_dir = os.path.join('dump/eval')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    args.save_name = os.path.join(args.save_dir, 'best_model.pth.tar')
    if args.resume:
        shutil.copyfile(args.resume_file, args.save_name)

    util.setup_logging(os.path.join(args.save_dir, 'log.txt'))
    results_file = os.path.join(args.save_dir, 'results.%s')
    results = util.ResultsLog(results_file % 'csv', results_file % 'html')

    logging.info("Saving to %s", args.save_dir)

    ########## load data
    args.data_path = os.path.join(args.data_path, args.dataset)
    if args.dataset == 'MNIST':
        args.input_channels = 1
        args.im_size = 28
        args.input_dim = 28*28*1
        args.output_dim = 10

        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_set = datasets.MNIST(args.data_path, train=True, download=True, transform=transform)
        test_set = datasets.MNIST(args.data_path, train=False, download=True, transform=transform)

        if args.val_set == 'TRAIN':
            train_frac = 5./6
        else:
            train_frac = 1
        num_train = len(train_set)
        indices = list(range(num_train))
        args.dataset_size = int(np.floor(train_frac * num_train))

    elif 'CIFAR' in args.dataset:
        args.input_channels = 3
        args.im_size = 32
        args.input_dim = 32*32*3
        args.output_dim = 10
        if args.dataset == 'CIFAR100':
            args.output_dim = 100

        kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
        transform_train=transforms.Compose([
            transforms.RandomCrop(args.im_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        if args.dataset == 'CIFAR10':
            train_set = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transform_train)
            test_set = datasets.CIFAR10(args.data_path, train=False, download=True, transform=transform_test)
        elif args.dataset == 'CIFAR100':
            train_set = datasets.CIFAR100(args.data_path, train=True, download=True, transform=transform_train)
            test_set = datasets.CIFAR100(args.data_path, train=False, download=True, transform=transform_test)
        else:
            print 'Dataset type "{0}" not recognized, exiting ...'.format(args.dataset)
            exit()

        if args.val_set == 'TRAIN':
            train_frac = 0.9
        else:
            train_frac = 1
        num_train = len(train_set)
        indices = list(range(num_train))
        args.dataset_size = int(np.floor(train_frac * num_train))

    elif args.dataset == 'TINYIMAGENET200':
        args.input_channels = 3
        args.output_dim = 200
        args.im_size =64
        args.input_dim = 64*64*3

        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        transform_train=transforms.Compose([
            transforms.RandomResizedCrop(args.im_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_test = transforms.Compose([
            transforms.CenterCrop(args.im_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        traindir = os.path.join(args.data_path, 'train')
        util.create_val_folder(args)

        valdir = os.path.join(args.data_path, 'val/images')
        train_set = datasets.ImageFolder(traindir, transform=transform_train)
        val_set = datasets.ImageFolder(valdir, transform=transform_test)
        test_set = val_set  # no test label available!
        args.dataset_size = len(train_set)
    else:
        print 'Dataset type "{0}" not recognized, exiting ...'.format(args.dataset)
        exit()

    if args.num_epochs == 0:
        args.num_epochs = int(np.ceil(float(args.num_iters) * args.batch_size / float(args.dataset_size)))


    if args.dataset == 'TINYIMAGENET200':
        train_sampler = None
        val_sampler = None
    else:
        train_idx, val_idx = indices[:args.dataset_size], indices[args.dataset_size:]
        train_sampler = du.sampler.SubsetRandomSampler(train_idx)
        val_sampler = du.sampler.SubsetRandomSampler(val_idx)
        val_set = train_set

    train_loader = du.DataLoader(train_set, sampler=train_sampler, shuffle=(train_sampler is None),
            batch_size=args.batch_size, worker_init_fn=init_fn, **kwargs)
    val_loader = du.DataLoader(val_set, sampler=val_sampler, batch_size=args.batch_size, worker_init_fn=init_fn, **kwargs)
    test_loader = du.DataLoader(test_set, batch_size=args.batch_size, worker_init_fn=init_fn, **kwargs)

    if args.val_set == 'TEST':
        val_loader = test_loader

    # lr-decay
    if args.lr_decay != 'MSTEP':
        args.lr_interval = int(args.lr_interval)

    if not args.eval:
        print(args)
        logging.debug("Run arguments: %s", args)
    
    # loss-function
    if args.loss_function == 'HINGE':
        criterion = nn.MultiLabelMarginLoss()
    elif args.loss_function == 'CROSSENTROPY':
        criterion = nn.CrossEntropyLoss()
    else:
        print 'Loss type "{0}" not recognized, exiting ...'.format(args.loss_function)
        exit()

    if args.use_tensorboard:
        summary_writer = SummaryWriter(os.path.join(args.save_dir, args.dataset, args.architecture, args.method,
                                                    args.optimizer + '_' + 'lr' + str(args.learning_rate) + '_' +
                                                    'bts' + str(args.beta_scale) + '_' + 'lrs' + str(args.lr_scale)
                                                    + '_' + 'bti' + str(args.beta_interval), 'runs',
                                                    datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    else:
        summary_writer = None

    if args.method == 'CONTINUOUS':
        connet.setup_and_run(args, criterion, device, train_loader, test_loader, val_loader, logging, results)
    elif args.method == 'BC':
        binary_connect.setup_and_run(args, criterion, device, train_loader, test_loader, val_loader, logging, results, summary_writer)
    elif args.method == 'FENBP_TANH':
        fenbp_tanh.setup_and_run(args, criterion, device, train_loader, test_loader, val_loader, logging, results, summary_writer)
    elif args.method == 'MDA_SOFTMAX':
        md_softmax.setup_and_run(args, criterion, device, train_loader, test_loader, val_loader, logging, results, summary_writer)
    elif args.method == 'MDA_TANH':
        md_tanh.setup_and_run(args, criterion, device, train_loader, test_loader, val_loader, logging, results, summary_writer)
    elif args.method == 'SOFTMAX_PROJECTION':
        softmax_projn.setup_and_run(args, criterion, device, train_loader, test_loader, val_loader, logging, results, summary_writer)
    elif args.method == 'TANH_PROJECTION':
        tanh_projn.setup_and_run(args, criterion, device, train_loader, test_loader, val_loader, logging, results, summary_writer)
    else:
        print 'Method "{0}" not recognized, exiting ...'.format(args.method)
        exit()


if __name__ == '__main__':
    main()
