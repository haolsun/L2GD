'''
Adapted from kuangliu/pytorch-cifar .
'''

import torch.nn as nn
import torch.nn.functional as F
import torch

from models.conv2d_mtl import Conv2dMtl

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, normalization_layer=nn.BatchNorm2d, conv2d=nn.Conv2d):
        super(BasicBlock, self).__init__()
        self.conv1 = conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = normalization_layer(planes)
        
        self.conv2 = conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = normalization_layer(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                normalization_layer(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, normalization_layer=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = normalization_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = normalization_layer(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = normalization_layer(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                normalization_layer(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channels=1, num_classes=2, norm_type='batchnorm', mtl=False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.normalization_layer = nn.BatchNorm2d if norm_type == 'batchnorm' else FilterResponseNorm_layer
        self.conv2d = Conv2dMtl if mtl else nn.Conv2d

        self.conv1 = self.conv2d(in_channels, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = self.normalization_layer(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1, self.normalization_layer, self.conv2d)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2, self.normalization_layer, self.conv2d)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2, self.normalization_layer, self.conv2d)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 2, self.normalization_layer, self.conv2d)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.n_features = 512 * block.expansion
        # self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, normalization_layer, conv2d):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, normalization_layer=normalization_layer, conv2d=conv2d))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        # out = self.linear(out)
        return out


def ResNet18(in_channels, num_classes, norm_type='batchnorm', mtl=False):
    return ResNet(BasicBlock, [2, 2, 2, 2], in_channels=in_channels, num_classes=num_classes, norm_type=norm_type,
                  mtl=mtl)



def ResNet50(in_channels, num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], in_channels=in_channels, num_classes=num_classes)


# From: https://raw.githubusercontent.com/izmailovpavel/neurips_bdl_starter_kit/main/pytorch_models.py
class FilterResponseNorm_layer(nn.Module):
    def __init__(self, num_filters, eps=1e-6):
        super(FilterResponseNorm_layer, self).__init__()
        self.eps = eps
        par_shape = (1, num_filters, 1, 1)  # [1,C,1,1]
        self.tau = torch.nn.Parameter(torch.zeros(par_shape))
        self.beta = torch.nn.Parameter(
            torch.zeros(par_shape))
        self.gamma = torch.nn.Parameter(
            torch.ones(par_shape))

    def forward(self, x):
        nu2 = torch.mean(torch.square(x), dim=[2, 3], keepdim=True)
        x = x * 1 / torch.sqrt(nu2 + self.eps)
        y = self.gamma * x + self.beta
        z = torch.max(y, self.tau)
        return z