import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

#resnet50 = models.resnet50(pretrained=True)
#resnet50 = models.resnext101_32x8d(pretrained=True)
#models.resnet152(pretrained=True)
import timm

nclasses = 20 
class Net(nn.Module):
    def __init__(self, model_name):
        super(Net, self).__init__()
        if model_name == 'ig_resnext101_32x48d' or  model_name == 'ig_resnext101_32x32d' or  model_name == 'swsl_resnext101_32x8d': 
            algo = timm.create_model(model_name, pretrained=True)#
            self.net = algo
            self.net.fc = nn.Sequential(
              nn.Linear(2048, 512),
              nn.BatchNorm1d(512),
              nn.ReLU(inplace = True),
              nn.Linear(512, nclasses)
            )
        elif model_name == 'resnext101_32x8d':
            algo = models.resnext101_32x8d(pretrained=True)
            self.net = algo
            self.net.fc = nn.Sequential(
              nn.Linear(2048, 512),
              nn.BatchNorm1d(512),
              nn.ReLU(inplace = True),
              nn.Linear(512, nclasses)
            )
        elif model_name == 'tf_efficientnet_b4_ns' or model_name == 'tf_efficientnet_b5_ns':
            algo = timm.create_model(model_name, pretrained=True)#
            dict_model_tolayers = {'tf_efficientnet_b4_ns' : 1792, 'tf_efficientnet_b5_ns' : 2048} 
            self.net = algo
            self.net.classifier = nn.Sequential(
              nn.Linear(dict_model_tolayers[model_name], 512),
              nn.BatchNorm1d(512),
              nn.ReLU(inplace = True),
              nn.Linear(512, nclasses)
            )
        print(self.net)
    # def forward(self, x):
    #     x = F.relu(F.max_pool2d(self.conv1(x), 2))
    #     x = F.relu(F.max_pool2d(self.conv2(x), 2))
    #     x = F.relu(F.max_pool2d(self.conv3(x), 2))
    #     x = x.view(-1, 320)
    #     x = F.relu(self.fc1(x))
    #     return self.fc2(x)
    def forward(self, x):
          return self.net.forward(x)
