import torch
from torch.optim.optimizer import Optimizer
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.nn.functional as F
import copy


import numpy as np

required = object()

def update_input(self, input, output):
    self.input = input[0].data
    self.output = output 

class FenBPOpt(Optimizer):

    def __init__(self, model, train_set_size, lr=1e-4, betas=0.0, delta=1e-6, eta=0.9999, lamda_init=10, lamda_std=0, reweight=1, use_STE=False, fenbp_alpha = 0.01, fenbp_beta = 0.01):
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
        #self.state['lamda'] = p.detach()
        
        #Momentum term
        self.state['momentum'] = torch.zeros_like(p, device=device) # momentum

        # step initilization
        self.state['step'] = 0
        self.state['reweight'] = reweight
        self.state['delta']=delta
        self.state['eta']=eta
        self.state['use_STE']=use_STE

        self.alpha = fenbp_alpha
        self.beta = fenbp_beta

    def set_train_modules(self, module):
        if len(list(module.children())) == 0:
            if len(list(module.parameters())) != 0:
                self.train_modules.append(module)
        else:
            for child in list(module.children()):
                self.set_train_modules(child)

    def get_grad(self, closure, theta, delta, eta, straight_through=False):
        parameters = self.param_groups[0]['params']

        #Annealing parameter
        beta = self.beta
        alpha = self.alpha

        # y = theta
        
        # Compute w*
        if straight_through:
            w_vector = torch.sign(theta)
        else:
            w_vector = F.hardtanh((theta + beta * (1 + alpha)*theta.sign())/(1 + beta), 
                    min_val=-1.0, max_val=1.0)

        # Pass to the model 
        vector_to_parameters(w_vector, parameters)
        
        # Get loss and predictions
        loss, preds = closure()

        linear_grad = torch.autograd.grad(loss, parameters)  # compute the gradidents
        grad = parameters_to_vector(linear_grad).detach()

        # Use sign to evaluate
        vector_to_parameters(torch.sign(theta), parameters)
        loss, preds = closure()

        if straight_through:
            return loss, preds, grad

        # Scale the gradients
        scale = torch.ones_like(theta)

        mask_pos_grad = grad > 1e-3
        mask_neg_grad = grad < -1e-3

        if (beta * alpha < 1):
            scale *= 1/(1 + beta)
            mask_pos_x = theta > max(1e-6, 1 - beta * alpha)
            mask_neg_x = theta < min(-1e-6,-1 + beta * alpha)

            mask_pos_mid_x  = (1e-6 < theta) & (theta < 1 - beta * alpha)
            mask_neg_mid_x = (-1 + beta * alpha < theta ) & ( theta  < -1e-6)

            mask = (mask_pos_x & mask_neg_grad) | (mask_neg_x & mask_pos_grad)
            scale[mask] = 0.0
            
            mask = (mask_neg_grad & mask_neg_x) | (mask_pos_grad & mask_pos_x)
            scale[mask] = torch.clamp( ((1 + 2*beta + beta*alpha)/(1 + beta))* 1./theta[mask].abs(), min=0, max = 0.5 )

            mask = (mask_neg_grad & mask_neg_mid_x) | (mask_pos_grad & mask_pos_mid_x)
            scale[mask] = torch.clamp( ((2*beta*(1+alpha) - theta[mask])/(1 + beta))*1./theta[mask].abs(), min=0, max = 0.5)

        else:
            mask_pos_x = theta > 1e-3
            mask_neg_x = theta < -1e-3

            mask = (mask_pos_x & mask_pos_grad) | (mask_neg_x & mask_neg_grad)
            scale[mask] = torch.clamp(1./theta[mask].abs(), min=0, max = 0.5)

            mask = (mask_pos_x & mask_neg_grad) | (mask_neg_x & mask_pos_grad)
            scale[mask] = 0.0

        grad *= scale
        return loss, preds, grad

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
        loss, pred, gr = self.get_grad(closure, lamda, delta=delta, eta = self.state['eta'], straight_through=self.state['use_STE'] )

        loss_list.append(loss)
        pred_list.append(pred)

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

