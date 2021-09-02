import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function 
import numpy as np
from bst import * 
import pdb

def if_binary(n):
    return (not('bn' in n) and not('downsample' in n) 
            and not('fc' in n and 'bias' in n))

def if_binary_tern(n):
    return (not('bn' in n) and not('downsample' in n)
            and not('fc' in n) and not(n == 'conv1.weight'))

def if_binary_lm(n):
    return ('weight' in n)
    # return ('weight' in n) and ('encoder' not in n)
    # return ('weight' in n) and ('coder' not in n)

def binary_reg(net):
    '''Binary-enforcing regularization for deep nets'''
    return sum([torch.min(torch.abs(w-1), torch.abs(w+1)).mean()
                for n, w in net.named_parameters() if if_binary(n)])

def adjust_bn(model):
    '''Turn off running stats tracking in BN layers'''
    for n, m in model.named_modules():
        if 'bn' in n:
            m.track_running_stats = False

def stochastic_binarize(A):
    '''Stochastic binarization of a Float tensor'''
    return (torch.rand_like(A) < ((A+1)/2)).mul(2).float() - 1

def exponential_binarize(A):
    '''Exponential binarization of a Float tensor'''
    A_exp = A.exp() / (A.exp() + (-A).exp())
    return (torch.rand_like(A) < A_exp).mul(2).float() - 1

def ternarize(A, delta_method='mean'):
    '''Ternarize a Float tensor'''
    A_quant = A.clone()
    # inds_one, inds_zero, inds_mone = (A >= 0.5), (A.abs() < 0.5), (A <= -0.5)
    # A_quant.masked_fill_(inds_one, 1.0)
    # A_quant.masked_fill_(inds_zero, 0.0)
    # A_quant.masked_fill_(inds_mone, -1.0)
    if delta_method == 'max':
        delta = A.abs().max() * 0.05
    elif delta_method == 'mean':
        delta = A.abs().mean() * 0.7
    A_quant.masked_fill_(A.abs() < delta, 0)
    inds_p, inds_n = (A >= delta), (A <= -delta)
    A_quant.masked_fill_(inds_p, A[inds_p].mean())
    A_quant.masked_fill_(inds_n, A[inds_n].mean())
    return A_quant

def greedy_median(w, n_bits=1, by_row=False):
    '''Greedy median quantization for tensors'''
    b_list, alpha_list = [], []
    r, w_hat = w.clone(), 0.
    for i in range(n_bits):
        b = r.sign()
        # Break sign ties randomly
        b[b == 0] = torch.randn_like(b[b == 0]).sign()
        if by_row:
            alpha, _ = r.abs().median(dim=1, keepdim=True)
        else:
            alpha = r.abs().median()
        r -= b * alpha
        w_hat += b * alpha
        b_list += [b]
        alpha_list += [alpha]
    return w_hat, b_list, alpha_list

def soft_threshold(w, w_hat, reg=0.0):
    '''Soft threshold a tensor towards another tensor'''
    w_sign, w_res = (w-w_hat).sign(), (w-w_hat).abs()
    return w_hat + w_sign * F.relu(w_res - reg)

class BinOp():
    def __init__(self, model, if_binary = if_binary, ttq=False):
        self.model = model
        self.saved_params = {}
        self.init_params = {}
        self.if_binary = if_binary

        for n, p in model.named_parameters():
            if self.if_binary(n):
                self.saved_params[n] = p.data.clone()
                self.init_params[n] = p.data.clone()
        self.ttq = ttq
        if ttq:
            self.ternary_assigns = {n: ([], [])
                                    for n, p in model.named_parameters()
                                    if self.if_binary(n)}
            self.ternary_vals = {}
            # self.ternary_vals.update({n + "_pos": p.data[p.data >= 0].mean().detach()
            #                           for n, p in model.named_parameters()
            #                           if self.if_binary(n)})
            # self.ternary_vals.update({n + "_neg": p.data[p.data < 0].mean().detach()
            #                           for n, p in model.named_parameters()
            #                           if self.if_binary(n)})
            self.ternary_vals.update({n + "_pos": torch.ones([1]).mean().to(p.device)
                                      for n, p in model.named_parameters()
                                      if self.if_binary(n)})
            self.ternary_vals.update({n + "_neg": torch.ones([1]).mean().mul(-1).to(p.device)
                                      for n, p in model.named_parameters()
                                      if self.if_binary(n)})
    def prox_operator(self, reg, reg_type='binary',
            n_bits = 1, by_row = False, n_rounds = 1,
            norm_rate = 1.0):
        if reg_type == 'binary':
            for n, p in self.model.named_paramters():
                if self.if_binary(n):
                    p_sign, p_abs = p.data.sign(), p.data.abs()

        if reg_type == 'tenary':
            for n, p in self.model.named_paramters():
                if self.if_binary(n):
                    p.data.copy_((p.data + ternarize(p.data) * reg)/(1 + reg))
        elif reg_type == 'median':
            for n, p in self.model.named_parameters():
                if self.if_binary(n):
                    p_quant, _, _ = greedy_median(p.data, n_bits=n_bits, by_row=by_row)
                    p.data.copy_(soft_threshold(p, p_quant, reg))
                    # p_sign, p_abs = p.data.sign(), p.data.abs()
                    # if by_row:
                    #     p_med, _ = p_abs.median(dim=1, keepdim=True)
                    # else:
                    #     p_med = p_abs.median()
                    # p.data.copy_(p_sign * (F.relu((p_abs - p_med).abs() - reg) \
                    #                        * (p_abs - p_med).sign() + p_med))
        elif reg_type == 'mean':
            for n, p in self.model.named_parameters():
                if self.if_binary(n):
                    p_prox = p.data.clone()
                    for _ in range(n_rounds):
                        p_quant, _, _ = alt_quantize(p_prox, n_bits=n_bits,
                                                     by_row=by_row, n_rounds=3,
                                                     norm_rate=norm_rate)
                        p_prox = (p.data + p_quant * reg) / (1 + reg)
                    p.data.copy_(p_prox)
                    # p.data.copy_(soft_threshold(p, p_quant, reg))


