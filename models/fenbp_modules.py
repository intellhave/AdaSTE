import torch
import pdb
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function

from .binarized_modules import Binarize

class FenFun_Linear(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        # output = input.mm(weight.sign().t())
        output = input.mm(weight.t())
        ctx.save_for_backward(input, weight, bias, output)
        return output

    @staticmethod 
    def backward(ctx, grad_output):
        input, weight, bias, output = ctx.saved_tensors
        grad_input = grad_weight, grad_bias = None
        grad_input = grad_output.mm(weight)
        grad_weight  = grad_output.t().mm(input)

        if bias is not None:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias



class FLinear(nn.Linear):
    
    def __init__(self, *kargs, **kwargs):
        super(FLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        if input.size(1) != 784:
            input.data = Binarize(input.data)

        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        # self.weight.data = Binarize(self.weight.org)
        out = F.linear(input, self.weight, self.bias)

        return out


    




