import argparse
import os
import time 
import logging
import torch
import torch.nn as nn
import torch.nn.parallel 
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import models
import copy

import torch.nn.functional as F
from torch.autograd import Variable
from data import get_dataset 
from preprocess import get_transform
from utils import * 
from datetime import datetime 
from ast import literal_eval 
from torchvision.utils import save_image
from reg import * 
from writer import * 


# Assert errors if nan is found during computing gradients
# torch.autograd.set_detect_anomaly(True)

model_names = sorted(name for name in models.__dict__ 
        if name.islower() and not name.startswith("__") 
        and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Pytorch ConvNet Training')

parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
        help='results dir')
parser.add_argument('--save', metavar='SAVE', default='', 
        help='saved folder')
parser.add_argument('--save_all', action='store_true', 
        help='save model at every epoch')
parser.add_argument('--tb_dir', type=str, default=None, 
        help='TensorBoard directory')
parser.add_argument('--dataset', metavar='DATASET', default='mnist',
                    help='dataset name or folder')
parser.add_argument('--model', '-a', metavar='MODEL', default='alexnet',
        choices=model_names, 
        help='model architecture: ' + '| '.join(model_names) + ' default : alexnet')
parser.add_argument('--input_size', type=int, default=None, 
        help='image input size')
parser.add_argument('--model_config', default='',
        help='additional architecture configuration')
parser.add_argument('--type', default='torch.cuda.FloatTensor',
        help='type of tensor - e.g torch.cuda.HalfTensor')
parser.add_argument('--gpus', default='0',
        help='gpus used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
        help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=2500, type=int, metavar='N',
        help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
        help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
        metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--binary_reg', default=0.0, type=float,
        help='Binary regularization strength')
parser.add_argument('--reg_rate', default=0.0, type=float,
        help='Regularization rate')
parser.add_argument('--adjust_reg', action='store_true',
        help='Adjust regularization based on learning rate decay')
parser.add_argument('--projection_mode', default=None, type=str,
        help='Projection / rounding mode')
parser.add_argument('--freeze_epoch', default=-1, type=int,
        help='Epoch to freeze quantization')
parser.add_argument('--optimizer', default='SGD', type=str, metavar='OPT',
        help='optimizer function used')
parser.add_argument('--lr', '--learning_rate', default=100, type=float,
        metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
        help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
        metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
        metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
        help='path to latest checkpoint (default: none)')
parser.add_argument('--binarize', action='store_true',
        help='Load an existing model and binarize')
parser.add_argument('--binary_regime', action='store_true',
        help='Use alternative stepsize regime (for binary training)')
parser.add_argument('--ttq_regime', action='store_true',
        help='Use alternative stepsize regime (for ttq)')
parser.add_argument('--no_adjust', action='store_true',
        help='Will not adjust learning rate')
parser.add_argument('-e', '--evaluate', type=str, metavar='FILE',
        help='evaluate model FILE on validation set')

parser.add_argument('--delta_decrease_epoch', default=5, type=int, help='Number of epoch to reduce delta (delta = delta * 0.1)')
parser.add_argument('--init_eta', default=0.001, type=int, help='Initial value for eta')

def main():
    global args, best_prec1
    best_prec1 = 0 
    args = parser.parse_args()

    if args.evaluate:
        args.results_dir = '/tmp'
    if args.save == '':
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    save_path=os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    setup_logging(os.path.join(save_path, 'log.txt'))
    results_file = os.path.join(save_path,'results.%s')
    results = ResultsLog(results_file % 'csv', results_file % 'html')

    logging.info("saving to %s", save_path)
    logging.debug("run arguments: %s", args)

    writer = TensorboardWriter(args.tb_dir + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    logging.info("creating model %s", args.model)
    model = models.__dict__[args.model]
    bin_model = models.__dict__[args.model]

    model_config = {'input_size': args.input_size, 'dataset':args.dataset}

    if args.model_config != '':
        model_config = dict(model_config, **literal_eval(args.model_config))

    # Prepare model and binary model 
    model = model(**model_config) 
    bin_model = bin_model(**model_config)

    logging.info("created model with configuration: %s", model_config)

    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            parser.error('invalid checkpoint: {}'.format(args.evaluate))
        checkpoint=torch.load(args.evaluate)
        model.load_state_dict(checkpoint['state_dict'])
        bin_model.load_state_dict(checkpoint['state_dict'])
        logging.info("loaded checkpoint '%s' (epoch %s)",
                args.evaluate, checkpoint['epoch'])
    elif args.resume:
        checkpoint_file = args.resume
        if os.path.isdir(checkpoint_file):
            checkpoint_file = os.path.join(checkpoint_file, 'model_best.pth.tar')
        if os.path.isfile(checkpoint_file):
            logging.info("loading checkpoint '%s'", args.resume)
        best_prec1 = checkpoint['best_prec1']
        best_prec1 = best_prec1.cuda(args.gpus[0])
        model.load_state_dict(checkpoint['state_dict'])
        bin_model.load_state_dict(checkpoint['state_dict'])
        logging.info("loaded checkpoint '%s' (epoch %s)",
                checkpoint_file, checkpoint['epoch'])
    else:
        logging.error("no checkpoint found at '%s'", args.resume)
    
    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters :%d", num_parameters)

    #adjust batchnorm layers if in stochastic binarization mode
    if args.projection_mode=='stoch_bin':
        adjust_bn(model)

    # Data Loading Code
    default_transform = {
            'train':get_transform(args.dataset, 
                input_size=args.input_size, augment=True),
            'eval': get_transform(args.dataset, 
                input_size=args.input_size, augment=False)
    }

    transform =getattr(model, 'input_transform', default_transform)
    regime = getattr(model, 'regime', {0: {'optimizer': args.optimizer, 
                                            'lr': args.lr,
                                            'momentum': args.momentum,
                                            'weight_decay':args.weight_decay}})

    #Adjust stepsize regime for specific optimizers
    if args.optimizer == 'Adam':
        regime = {
            0 : {'optimizer': 'Adam', 'lr': args.lr},
        }

    # criterion = getattr(model, 'criterion', nn.CrossEntropyLoss)()
    criterion = nn.CrossEntropyLoss()
    criterion.type(args.type)
    model.type(args.type)
    bin_model.type(args.type)

    val_data = get_dataset(args.dataset, 'val', transform['eval'])
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, bin_model, criterion, 0)
        return

    #Get training and validation datasets
    train_data = get_dataset(args.dataset, 'train', transform['train'])
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    print('learning rate: ', args.lr)
    trainable_params = filter(lambda p: p.requires_grad,  model.parameters())
    optimizer = torch.optim.SGD(trainable_params, lr=args.lr)
    logging.info('training regime : %s', regime)

    bin_op = BinOp(model, if_binary = if_binary_tern if args.projection_mode in [ 
        'prox_ternary', 'ttq'] else if_binary,
        ttq = (args.projection_mode=='ttq'))

    delta0 = 0.000001
    delta=delta0
    epsillon = 1e-6
    eta = args.init_eta

    try:
        for epoch in range(args.start_epoch, args.epochs):
            if not(args.no_adjust):
                optimizer = adjust_optimizer(optimizer, epoch, regime)

            # if ((epoch+1) % args.delta_decrease_epoch == 0):
            #     delta = epsillon + delta0 - epoch*delta0/args.epochs
            # print('Delta changed to ', delta)

            # Training
            train_loss, train_prec1, train_prec5 = train(train_loader, 
                    model, bin_model, criterion, epoch, 
                    optimizer, delta = delta, eta = eta)

            # evaluate on validation set 
            val_loss, val_prec1, val_prec5 = validate(
                    val_loader, model, bin_model, criterion, epoch, 
                    delta = delta, eta = eta)

            if args.binary_reg > 1e-10 or args.reg_rate > 1e-10:
                is_best = val_prec1 > best_prec1
                best_prec1 = max(val_prec1, best_prec1)
            else:
                is_best = val_prec1 > best_prec1
                best_prec1 = max(val_prec1, best_prec1)

            save_checkpoint({
                'epoch': epoch + 1,
                'model': args.model,
                'config': args.model_config,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'regime': regime
            }, is_best, path=save_path, save_all=args.save_all)
            logging.info('\n Epoch: {0}\t'
                         'Training Loss {train_loss:.4f} \t'
                         'Training Prec@1 {train_prec1:.3f} \t'
                         'Training Prec@5 {train_prec5:.3f} \t'
                         'Validation Loss {val_loss:.4f} \t'
                         'Validation Prec@1 {val_prec1:.3f} \t'
                         'Validation Prec@5 {val_prec5:.3f} \n'
                         .format(epoch + 1, train_loss=train_loss, val_loss=val_loss,
                                 train_prec1=train_prec1, val_prec1=val_prec1,
                                 train_prec5=train_prec5, val_prec5=val_prec5))
            
            results.add(epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss,
                        train_error1=100 - train_prec1, val_error1=100 - val_prec1,
                        train_error5=100 - train_prec5, val_error5=100 - val_prec5)
            results.save()
            # result_dict = {'train_loss': train_loss, 'val_loss': val_loss,
            #                'train_error1': 100 - train_prec1, 'val_error1': 100 - val_prec1,
            #                'train_error5': 100 - train_prec5, 'val_error5': 100 - val_prec5,
            #                'val_error1_bin': 100 - val_prec1_bin,
            #                'val_error5_bin': 100 - val_prec5_bin}

            result_dict = {'train_loss': train_loss, 'val_loss': val_loss,
                           'train_prec1': train_prec1, 'val_prec1': val_prec1,
                           'train_prec5': train_prec5, 'val_prec5': val_prec5,
                        }
            writer.write(result_dict, epoch+1)
            # writer.write(binary_levels(model), epoch+1)
            
            # Compute general quantization error
            # mode = 'ternary' if args.projection_mode == 'prox_ternary' else 'deterministic'
            # writer.write(bin_op.quantize_error(mode=mode), epoch+1)
            # writer.write(sign_changes(bin_op), epoch+1)
            # if bin_op.ttq:
            #     writer.write(bin_op.ternary_vals, epoch+1)
            # writer.export()

            # Optionally freeze the binarization at a given epoch
            if args.freeze_epoch > 0 and epoch+1 == args.freeze_epoch:
                if args.projection_mode in ['prox', 'lazy']:
                    bin_op.quantize(mode='binary_freeze')
                elif args.projection_mode == 'prox_ternary':
                    bin_op.quantize(mode='ternary_freeze')
                args.projection_mode = None

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

   
def forward(data_loader, model, bin_model, criterion,  epoch=0, training=True, optimizer=None, 
        br=0.0, bin_op=None, projection_mode=None, binarize=False, delta = 1.0, eta = 1.0):

    if args.gpus and len(args.gpus) > 1:
        print('GPU')
        model = torch.nn.DataParallel(model, args.gpus)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()

    for i, (inputs, target) in enumerate(data_loader):

        data_time.update(time.time() - end)
        if args.gpus is not None:
            target = target.cuda()

        # input_var = Variable(inputs.type(args.type), volatile = not training)
        with torch.no_grad():
            input_var = Variable(inputs.type(args.type))

        target_var = Variable(target)

        # If the model is MLP (for MNIST), need to reshape it 
        if args.model == 'simple_mlp':
            input_var = input_var.reshape(-1, 28*28)

        #compute output 
        output = model(input_var)
        loss = criterion(output, target_var)
        if type(output) is list:
            output = output[0]

        # Compute prediction by bin_model
        bin_output = bin_model(input_var)
        bin_loss = criterion(bin_output, target_var)
        if type(bin_output) is list:
            bin_output = bin_output[0]

        # measure accuracy and record loss
        # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        bin_prec1, bin_prec5 = accuracy(bin_output.data, target, topk=(1,5))
        top1.update(bin_prec1.item(), inputs.size(0))
        top5.update(bin_prec5.item(), inputs.size(0))
        losses.update(bin_loss.data.item(), inputs.size(0))

        # eta = 0.25
        min_dt = -1.0
        max_dt = 1.0
        
        if training:

            # Dictionary to store the backup and quantized parameters
            backup_params={} 
            hardtanh_params = {}

            # Compute w* and assign them to the model before applying backward()
            for n, p in model.named_parameters():
                if if_binary(n):
                    # Store the current weights 
                    backup_params[n] = copy.deepcopy(p.data)

                    #Compute w*
                    wstar = F.hardtanh(p.data/delta, min_val=min_dt, max_val=max_dt)
                    hardtanh_params[n] = copy.deepcopy(wstar)

                    # Assign w* to the model parameters
                    p.data.copy_(wstar) 
                else:
                    # Do this to also copy non-binary 
                    # parameters to the evaluation network 
                    hardtanh_params[n] = copy.deepcopy(p.data)

            #Compute gradient w.r.t. w*
            output = model(input_var)
            loss = criterion(output, target_var)
            optimizer.zero_grad()
            loss.backward()

            #Now, restore the parameters
            with torch.no_grad():
                for n,p in model.named_parameters():
                    if if_binary(n):
                        p.data.copy_(backup_params[n])


            # With wstar and grad, we can now compute delta E and update paramteters 
            with torch.no_grad():
                for n, p in bin_model.named_parameters():
                    p.data.copy_(hardtanh_params[n])

            for n, p in model.named_parameters():
                if if_binary(n):
                    tau_vec = (1/32)*torch.ones_like(p.data)
                    y = p.data/delta
                    g = p.grad.data
                    wbar = (y - tau_vec * g)

                    # Compute tau 
                    mask_pos_grad = p.grad.data >= 0 
                    mask_neg_grad = p.grad.data < 0
                    # X
                    mask_pos_x = p.data >= delta
                    mask_neg_x = p.data <= -delta
                    
                    curr_mask = mask_neg_x & mask_neg_grad
                    tau_vec[curr_mask] = torch.min((y[curr_mask] - 1)/g[curr_mask], 
                            1/(1-eta) * ((y[curr_mask]+1)/g[curr_mask]))
                    wbar[curr_mask] = y[curr_mask] - torch.min((y[curr_mask] - 1), 
                            1/(1-eta) * ((y[curr_mask]+1)))

                    curr_mask = mask_pos_x & mask_pos_grad
                    tau_vec[curr_mask] = torch.min((y[curr_mask] + 1)/g[curr_mask], 
                            1/(1-eta) * ((y[curr_mask]-1)/g[curr_mask]))
                    wbar[curr_mask] = y[curr_mask] - torch.min((y[curr_mask] + 1), 
                            1/(1-eta) * ((y[curr_mask]-1)))

                    wbar = F.hardtanh(wbar, min_val=min_dt, max_val=max_dt)

                    # Use autograd to update theta -> This should be done in closed-form
                    tt = Variable(p.data, requires_grad=True)
                    sqrt_tau = torch.sqrt(tau_vec)
                    inftau = torch.isinf(tau_vec) | torch.isnan(tau_vec)
                    sqrt_tau[inftau] = 1
                    wbar[inftau] = tt[inftau]

                    wstar = hardtanh_params[n] 
                    dE = ((torch.norm((tt - wbar)/sqrt_tau)) 
                            - (torch.norm((tt-wstar)/sqrt_tau)))
                    deltaE = dE.mean()
                    deltaE.backward()

                    #Handle nan cases 
                    nangrad = torch.isnan(tt.grad.data)
                    tt.grad.data[nangrad] = 0
                    p.grad.data.copy_(tt.grad.data)
                    # p.grad.data.copy_(delta * tt.grad.data)
                else:
                    p.grad.data.copy_(p.grad.data)

            optimizer.step()
            
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logging.info('{phase} - Epoch: [{0}][{1}/{2}], Delta : {delta}\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                             epoch, i, len(data_loader), delta = delta,
                             phase='TRAINING' if training else 'EVALUATING',
                             batch_time=batch_time, data_time=data_time, loss=losses,  
                             top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg

        
def train(data_loader, model, bin_model, criterion, epoch, optimizer,
          br=0.0, bin_op=None, projection_mode=None, delta = 1.0, eta = 1.0):
    # switch to train mode
    model.train()
    return forward(data_loader, model, bin_model, criterion, epoch,
                   training=True, optimizer=optimizer, delta = delta, eta = eta)


def validate(data_loader, model, bin_model, criterion, epoch,
             br=0.0, bin_op=None, projection_mode=None,
             binarize=False, delta = 1.0, eta = 1.0):
    # switch to evaluate mode
    model.eval()
    return forward(data_loader, model, bin_model, criterion, epoch,
                   training=False, optimizer=None, delta = delta, eta = eta)


if __name__ == '__main__':
    main()
