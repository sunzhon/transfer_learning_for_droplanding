import torch
import torch.nn as nn
from transfer_losses import TransferLoss
import backbones
import pdb




'''
A transfer network for regression


'''
class TransferNetForRegression(nn.Module):
    '''
    num_label is label numbers, which defines output number

    '''
    def __init__(self, num_label=1, base_net='mlnn', transfer_loss='mmd', use_bottleneck=False, bottleneck_width=20, max_iter=1000, src_reg_loss_weight=2.1, tgt_reg_loss_weight=.0, **kwargs):
        super(TransferNetForRegression, self).__init__()
        self.num_label = num_label
        self.base_network = backbones.get_backbone(base_net,**kwargs)
        self.use_bottleneck = use_bottleneck
        self.src_reg_loss_weight = src_reg_loss_weight
        self.tgt_reg_loss_weight = tgt_reg_loss_weight
        self.transfer_loss = transfer_loss
        if self.use_bottleneck:
            bottleneck_list = [
                nn.Linear(self.base_network.output_num(),bottleneck_width),
                nn.ReLU(),
                #nn.LSTM(input_size=bottleneck_width, hidden_size=60, num_layers=1, bidirectional=True, batch_first=True),
                #nn.Linear(2*bottleneck_width, bottleneck_width)
                nn.Linear(bottleneck_width, bottleneck_width)
            ]
            self.bottleneck_layer = nn.Sequential(*bottleneck_list)
            feature_dim = bottleneck_width # laten feature space
        else:
            feature_dim = self.base_network.output_num() # latent feature space
        
        # output layers
        #self.output_layer = nn.Linear(feature_dim, num_label)

        output_list = [
                nn.Linear(feature_dim,40),
                nn.Dropout(p=0.2),
                nn.ReLU(),
                nn.Linear(40, 100),
                nn.Dropout(p=0.2),
                nn.ReLU(),
                nn.Linear(100, num_label)
            ]
        self.output_layer = nn.Sequential(*output_list)


        transfer_loss_args = {
            "loss_type": self.transfer_loss,
            "input_dim": feature_dim*80,
            "hidden_dim": 80, # empirically set for discriminator 
            "max_iter": max_iter
        }
        self.adapt_loss = TransferLoss(**transfer_loss_args)
        self.criterion = torch.nn.MSELoss()

    def forward(self, samples):
        (source, source_label) = samples['src']
        (target_cls, target_label_cls) = samples['tcl']
        if(self.tgt_reg_loss_weight>0.): #using target reg
            (target_reg, target_label_reg) = samples['tre']

        # base_network
        target_cls = self.base_network(target_cls)
        source = self.base_network(source)
        if(self.tgt_reg_loss_weight>0.): #using target reg
            target_reg = self.base_network(target_reg)

        # bootleneck network
        if self.use_bottleneck:
            source = self.bottleneck_layer(source)
            target_cls = self.bottleneck_layer(target_cls)
            if(self.tgt_reg_loss_weight>0.): #using target reg
                target_reg = self.bottleneck_layer(target_reg)


        # regression
        source_reg = self.output_layer(source)
        if(self.tgt_reg_loss_weight>0.): #using target reg
            target_reg = self.output_layer(target_reg)

        # regression loss, normal danndo not use target loss since target data without labels
        src_reg_loss =  self.criterion(source_reg, source_label)
        if(self.tgt_reg_loss_weight>0.): #using target reg
            tgt_reg_loss =  self.criterion(target_reg, target_label_reg)
        else:
            tgt_reg_loss = 0

        reg_loss = self.src_reg_loss_weight*src_reg_loss + self.tgt_reg_loss_weight*tgt_reg_loss

        # transfer
        kwargs = {}
        transfer_loss = self.adapt_loss(torch.flatten(source,1), torch.flatten(target_cls,1), **kwargs)

        return reg_loss, transfer_loss
    
    def get_parameters(self, initial_lr=1.0):
        params = [
            {'params': self.base_network.parameters(), 'lr': 0.2 * initial_lr},
            {'params': self.output_layer.parameters(), 'lr': 1.0 * initial_lr},
        ]
        if self.use_bottleneck:
            params.append(
                {'params': self.bottleneck_layer.parameters(), 'lr': 1.0 * initial_lr}
            )
        # Loss-dependent
        if self.transfer_loss == "adv":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
        elif self.transfer_loss == "daan":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
            params.append(
                {'params': self.adapt_loss.loss_func.local_classifiers.parameters(), 'lr': 1.0 * initial_lr}
            )
        return params

    def predict(self, x):
        features = self.base_network(x)
        if self.use_bottleneck:
            x = self.bottleneck_layer(features)
            reg = self.output_layer(x)
        else:
            reg = self.output_layer(features)
        return reg



class BaselineModel(nn.Module):
    '''
    num_label is label numbers, which defines output number

    '''
    def __init__(self, num_label=1, base_net='mlnn', finetuning=False, **kwargs):
        super(BaselineModel, self).__init__()
        self.num_label = num_label
        self.base_network = backbones.get_backbone(base_net,**kwargs)
        feature_dim = self.base_network.output_num()

        #self.output_layer = nn.Linear(feature_dim, num_label)
        output_list = [
                nn.Linear(feature_dim,100),
                nn.Dropout(p=0.2),
                nn.ReLU(),
                nn.Linear(100,50),
                nn.Dropout(p=0.2),
                nn.ReLU(),
                nn.Linear(50, 20),
                nn.Dropout(p=0.2),
                nn.ReLU(),

                nn.Linear(20, num_label)
            ]
        self.output_layer = nn.Sequential(*output_list)


        self.criterion = torch.nn.MSELoss()
        self.finetuning=finetuning

    def forward(self, samples):
        target_reg, target_label_reg = samples['tre']
        target_reg = self.base_network(target_reg)
        target_reg = self.output_layer(target_reg)
        reg_loss = self.criterion(target_reg, target_label_reg)
        transfer_loss = 0*reg_loss

        return reg_loss, transfer_loss # 0 indicates transfer loss

    def predict(self, x):
        features = self.base_network(x)
        reg = self.output_layer(features)
        return reg

    def get_parameters(self, initial_lr=1.0):
        if(self.finetuning):# only train the output layer parameters
            print('Fine-tuning a trained model')
            params = [
                {'params': self.output_layer.parameters(), 'lr': 1.0 * initial_lr}
                ]
        else:
            params = [
                {'params': self.base_network.parameters(), 'lr': 0.1 * initial_lr},
                {'params': self.output_layer.parameters(), 'lr': 1.0 * initial_lr},
            ]

        return params


class DiscriminatorModel(nn.Module):

    '''




    '''
    def __init__(self, num_label=1, base_net='discriminator', **kwargs):
        super(DiscriminatorModel, self).__init__()
        self.num_label = num_label
        self.base_network = backbones.get_backbone(base_net,**kwargs)
        feature_dim = self.base_network.output_num()

        self.output_layer = nn.Linear(feature_dim, num_label)

        self.criterion = torch.nn.MSELoss()
        self.finetuning=finetuning

    def forward(self, source, target, source_label, target_label):
        target = self.base_network(target)
        target_reg = self.output_layer(target)
        reg_loss = self.criterion(target_reg, target_label)
        transfer_loss = 0*reg_loss

        return reg_loss, transfer_loss # 0 indicates transfer loss

    def predict(self, x):
        features = self.base_network(x)
        reg = self.output_layer(features)
        return reg
