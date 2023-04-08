import torch.nn as nn
import torch
from torchvision import models
import pdb
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pack_sequence,pad_packed_sequence
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout


resnet_dict = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "resnet152": models.resnet152,
        }

def get_backbone(name,**kwargs):
    if "resnet" in name.lower():
        return ResNetBackbone(name)
    elif "alexnet" == name.lower():
        return AlexNetBackbone()
    elif "dann" == name.lower():
        return DaNNBackbone()
    elif "mlnn" == name.lower(): # modular lstm neural network, defined by suntao
        if 'num_layers' in kwargs.keys():
            num_layers = kwargs['num_layers']
            return MLNNBackbone(num_layers=num_layers,n_input=kwargs['features_num'])
        else:
            return MLNNBackbone(n_input=kwargs['features_num'])
    elif "cnn" == name.lower(): # cnn, defined by suntao
        if 'num_layers' in kwargs.keys():
            num_layers = kwargs['num_layers']
            return CNNBackbone(num_layers=num_layers)
        else:
            return CNNBackbone()


class MLNNBackbone(nn.Module):
    def __init__(self, n_input=48, seq_len=80, n_output=1, hidden_size=100, num_layers=1):
        super(MLNNBackbone, self).__init__()
        if num_layers>1:
            self.lstm_layer = nn.LSTM(input_size=n_input, hidden_size = hidden_size, num_layers=num_layers, dropout=0.2,bidirectional=True, batch_first=True)
        else:
            self.lstm_layer = nn.LSTM(input_size=n_input, hidden_size = hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self._feature_dim = 2*hidden_size

    def forward(self, sequence): # input dim = [batch_size, seq_en, features_len]
        batch_size = sequence.shape[0]
        sequence = pack_padded_sequence(sequence, batch_size*[self.seq_len], batch_first=True, enforce_sorted=False)
        lstm_out, (hidden, c) = self.lstm_layer(sequence) # lstm_out dim  = [batch_size, seq_len, model_dim]
        lstm_out,_= pad_packed_sequence(lstm_out, batch_first=True)

        return lstm_out
    #return lstm_out[:,-1, :]

    def output_num(self):
        return self._feature_dim

'''
A backup of MLNN on Feb 10 2023

'''
class CNNBackbone(nn.Module):
    def __init__(self,in_channels=1,n_output=1,num_layers=1):
        super(CNNBackbone, self).__init__()
        self.cnn_layers=Sequential(
                # Defining a 2D convolution layer
                Conv2d(in_channels=in_channels,out_channels=8, kernel_size=(4,6), stride=1, padding=0),
                BatchNorm2d(8),
                ReLU(inplace=True),
                MaxPool2d(kernel_size=(4,6), stride=1),
                ## Defining another 2D convolution layer
                nn.Dropout(p=0.3),
                Conv2d(8, 8, kernel_size=(4,6), stride=1, padding=0),
                BatchNorm2d(8),
                ReLU(inplace=True),
                MaxPool2d(kernel_size=(4,6), stride=1),
                )
        self.in_channels = in_channels
        self._feature_dim=68

        # Defining the forward pass
    def forward(self, x): #input_dim = [batch_size, seq_len (80), feature_num (49)]
        (batch_size, seq_len, feature_num) =x.size()
        x = x.view(batch_size, self.in_channels, seq_len, feature_num)
        x = self.cnn_layers(x)
        x = x.view(batch_size, seq_len, -1)
        return x

    def output_num(self):
        return self._feature_dim


class DaNNBackbone(nn.Module):
    def __init__(self, n_input=224*224*3, n_hidden=256):
        super(DaNNBackbone, self).__init__()
        self.layer_input = nn.Linear(n_input, n_hidden)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self._feature_dim = n_hidden

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x

    def output_num(self):
        return self._feature_dim

# convnet without the last layer
class AlexNetBackbone(nn.Module):
    def __init__(self):
        super(AlexNetBackbone, self).__init__()
        model_alexnet = models.alexnet(pretrained=True)
        self.features = model_alexnet.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module(
                    "classifier"+str(i), model_alexnet.classifier[i])
            self._feature_dim = model_alexnet.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256*6*6)
        x = self.classifier(x)
        return x

    def output_num(self):
        return self._feature_dim

class ResNetBackbone(nn.Module):
    def __init__(self, network_type):
        super(ResNetBackbone, self).__init__()
        resnet = resnet_dict[network_type](pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self._feature_dim = resnet.fc.in_features
        del resnet

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self._feature_dim



if __name__=='__main__':
    model = MLNNBackbone(n_input=48, seq_len=80, n_output=1, hidden_size=100, num_layers=1)
    #model = CNNBackbone()
    print(model)
    #model()

