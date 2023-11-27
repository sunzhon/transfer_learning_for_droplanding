import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np
import pdb




class LambdaSheduler(nn.Module):
    def __init__(self, gamma=1.0, max_iter=1000, **kwargs):
        super(LambdaSheduler, self).__init__()
        self.gamma = gamma
        self.max_iter = max_iter
        self.curr_iter = 0

    def lamb(self):
        p = self.curr_iter / self.max_iter
        lamb = 2. / (1. + np.exp(-self.gamma * p)) - 1
        return lamb
    
    def step(self):
        self.curr_iter = min(self.curr_iter + 1, self.max_iter)



class AdversarialLoss(nn.Module):
    '''
    Acknowledgement: The adversarial loss implementation is inspired by http://transfer.thuml.ai/
    '''
    def __init__(self, gamma=4.0, max_iter=1000, use_lambda_scheduler=True, **kwargs):
        super(AdversarialLoss, self).__init__()
        input_dim=kwargs['input_dim']; hidden_dim=kwargs['hidden_dim']
        self.domain_classifier = Discriminator(input_dim, hidden_dim) #分类器
        self.use_lambda_scheduler = use_lambda_scheduler
        if self.use_lambda_scheduler:
            self.lambda_scheduler = LambdaSheduler(gamma, max_iter)
        
    def forward(self, source, target):
        if self.use_lambda_scheduler:
            lamb = self.lambda_scheduler.lamb()
            self.lambda_scheduler.step()

        source_loss = self.get_adversarial_result(source, True, lamb)
        target_loss = self.get_adversarial_result(target, False, lamb)
        adv_loss = 0.5 * (source_loss + target_loss)
        return adv_loss
    
    def get_adversarial_result(self, x, source=True, lamb=1.0):
        x = ReverseLayerF.apply(x, lamb) # 反转层
        domain_pred = self.domain_classifier(x)
        device = domain_pred.device
        # label source and target data to discriminate the source and target data
        if source: # source domain data was labeled with 1
            domain_label = torch.ones(len(x), 1).long().to(device)
        else: # target domain data was labeled with 0
            domain_label = torch.zeros(len(x), 1).long().to(device)
        loss_fn = nn.BCELoss()
        loss_adv = loss_fn(domain_pred, domain_label.float())
        #print("adv loss:", loss_adv)
        #print('domain pred: ', domain_pred)
        #print('domain label: ', domain_label)
        #input("adv loss")
        return loss_adv
    

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class Discriminator(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        layers = [
            	# initialize first set of CONV => RELU => POOL layers
            nn.Unflatten(1,(1,input_dim)),
		    nn.Conv1d(1, out_channels=20,kernel_size=5),
		    nn.ReLU(),
		    nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten(1,-1),

            #nn.Linear(input_dim, hidden_dim),
            nn.Linear(760, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        ]
        self.layers = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)
