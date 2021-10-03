import torch
from torch.optim.optimizer import Optimizer
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.nn.functional as F


import numpy as np
import pdb

required = object()

def update_input(self, input, output):
    self.input = input[0].data
    self.output = output 

class FenBPOpt(Optimizer):

    def __init__(self, model, train_set_size, lr=1e-4, betas=0.0, delta=1e-6, eta=0.9999, lamda_init=10, lamda_std=0, reweight=1):
        if train_set_size < 1:
            raise ValueError("Invalid number of datapoints: {}".format(train_set_size))

        defaults=dict(lr=lr, train_set_size=train_set_size, beta=betas)
        super(FenBPOpt, self).__init__(model.parameters(), defaults)

        self.train_modules = []
        self.set_train_modules(model)

        defaults = self.defaults
        parameters = self.param_groups[0]['params']
        self.param_groups[0]['lr'] = lr

        device = parameters[0].device

        p = parameters_to_vector(self.param_groups[0]['params'])

        mixtures_coeff = torch.randint_like(p,2)
        self.state['lamda'] =  mixtures_coeff * (lamda_init + np.sqrt(lamda_std)* torch.randn_like(p)) + (1-mixtures_coeff) * (-lamda_init + np.sqrt(lamda_std) * torch.randn_like(p))#  torch.log(1+p_value) - torch.log(1-p_value)  #torch.randn_like(p) # 100*torch.randn_like(p) # math.sqrt(train_set_size)*torch.randn_like(p)  #such initialization is empirically good, others are OK of course
        
        #Momentum term
        self.state['momentum'] = torch.zeros_like(p, device=device) # momentum

        # step initilization
        self.state['step'] = 0
        # self.state['temperature'] = temperature
        self.state['reweight'] = reweight
        self.state['delta']=delta
        self.state['eta']=eta

    def set_train_modules(self, module):
        if len(list(module.children())) == 0:
            if len(list(module.parameters())) != 0:
                self.train_modules.append(module)
        else:
            for child in list(module.children()):
                self.set_train_modules(child)

    def get_grad(self, closure, theta, delta, eta, straight_through=False):
        parameters = self.param_groups[0]['params']
        y = theta/delta
        w_vector = F.hardtanh(y, min_val=-1.0, max_val=1.0)
        vector_to_parameters(w_vector, parameters)

        # Get loss and predictions
        loss, preds = closure()

        linear_grad = torch.autograd.grad(loss, parameters)  # compute the gradidents
        grad = parameters_to_vector(linear_grad).detach()

        if straight_through:
            return loss, preds, grad


        tau_vec = (1/(1-eta)) * torch.ones_like(y)
        wbar = (y - tau_vec * grad)

        mask_pos_grad = grad >= 0
        mask_neg_grad = grad < -1e-12
        mask_pos_x = theta >= delta
        mask_neg_x = theta <= -delta

        curr_mask = mask_neg_x & mask_neg_grad
        wbar[curr_mask] = y[curr_mask] - 1/(1-eta) * ((y[curr_mask]+1))
        tau_vec[curr_mask] = torch.max(1e-12 * torch.ones_like(
            tau_vec[curr_mask]), 
            1/(1-eta) * ((y[curr_mask]+1)/grad[curr_mask]))

        curr_mask = mask_pos_x & mask_pos_grad
        wbar[curr_mask] = y[curr_mask] - 1/(1-eta) * ((y[curr_mask]-1))
        tau_vec[curr_mask] = torch.max(1e-12 * torch.ones_like(
            tau_vec[curr_mask]), 
            1/(1-eta) * ((y[curr_mask]-1)/grad[curr_mask]))

        wbar = F.hardtanh(wbar, min_val=-1.0, max_val=1.0)
        wstar = w_vector

        gr = (1/delta) * (wstar - wbar) / tau_vec
        return loss, preds, gr

    def step(self, closure):
        """Performs a single optimization step.
        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss without doing the backward pass
        """
        if closure is None:
            raise RuntimeError(
                'For now, BayesBiNN only supports that the model/loss can be reevaluated inside the step function')

        loss_list = []
        pred_list = []
        self.state['step'] += 1

        defaults = self.defaults
        # We only support a single parameter group.
        parameters = self.param_groups[0]['params']
        lr = self.param_groups[0]['lr']
        momentum_beta = defaults['beta']
        momentum = self.state['momentum']

        # mu = self.state['mu']
        lamda = self.state['lamda']
        reweight = self.state['reweight']

        grad_hat = torch.zeros_like(lamda)

        delta = self.state['delta']
        eta = self.state['eta']
        
        # Obtain gradients
        loss_soft, pred_soft, grad_soft = self.get_grad(closure, lamda, 1.0, 0.001,straight_through=True)
        loss, pred, gr = self.get_grad(closure, lamda, self.state['eta'], delta)

        loss_list.append(loss)
        pred_list.append(pred)

        gr = gr + grad_soft
        # gr = grad_soft
        # gr = grad
    

        # grad_hat = defaults['train_set_size'] * grad
        grad_hat = defaults['train_set_size'] * gr
        grad_hat = grad_hat.mul(defaults['train_set_size'])

        # Add momentum
        self.state['momentum'] = momentum_beta * self.state['momentum'] + (1-momentum_beta)*(grad_hat + reweight*(self.state['lamda'] ))

        # Get the mean loss over the number of samples
        loss = torch.mean(torch.stack(loss_list))

        # Bias correction of momentum as adam
        bias_correction1 = 1 - momentum_beta ** self.state['step']

        # Update lamda vector
        self.state['lamda'] = self.state['lamda'] - self.param_groups[0]['lr'] * self.state['momentum']/bias_correction1


        return loss, pred_list






