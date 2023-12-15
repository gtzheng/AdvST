from __future__ import absolute_import, division

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

__all__ = ['AlexNet', 'alexnet']

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000, contrastive=''):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True)
        )
        self.linear = nn.Linear(4096, num_classes)
        if 'contrastive' in contrastive:
            self.pro_head = nn.Sequential(nn.Linear(4096, 128),
                                          nn.ReLU(),
                                          nn.Linear(128, 128))
            self.contrastive = True
        else:
            self.contrastive = False

    def get_proj(self, fea):
        z = self.pro_head(fea)
        z = F.normalize(z,dim=-1)
        return z

    def forward(self, x):
        end_points = {}

        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        
        
        x = self.classifier(x)
        end_points['Embedding'] = x
        if self.contrastive:
            end_points['Projection'] = self.get_proj(x)
        x = self.linear(x)
        end_points['Predictions'] = F.softmax(input=x, dim=-1)
        
        return x, end_points


def alexnet(pretrained=False, **kwargs):
    model = AlexNet(**kwargs)

    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['alexnet'])

        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v.data for k, v in pretrained_dict.items() if
                           k in model_dict and v.shape == model_dict[k].size()}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    return model
