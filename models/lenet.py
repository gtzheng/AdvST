from __future__ import absolute_import, division

import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
class LeNet5(nn.Module):

    def __init__(self, num_classes, contrastive=False):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5)
        self.fc1 = nn.Linear(128 * 5 * 5, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.linear = nn.Linear(1024, num_classes)
        if 'contrastive' in contrastive:
            self.pro_head = nn.Linear(1024, 128)
            self.contrastive = True
        else:
            self.contrastive = False

    def get_proj(self, fea):
        z = self.pro_head(fea)
        z = F.normalize(z,dim=-1)
        return z
    def forward(self, x, more=False):
        end_points = {}

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.reshape(x.size(0), -1)
        

        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        
        end_points['Embedding'] = x
        if self.contrastive:
            end_points['Projection'] = self.get_proj(x)
   
        x = self.linear(x)
       
        end_points['Predictions'] = F.softmax(input=x, dim=-1)
        self.end_points = end_points
        if more:
            return x, end_points
        else:
            return x
