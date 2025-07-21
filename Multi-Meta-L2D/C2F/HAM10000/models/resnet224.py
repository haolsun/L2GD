import torch
import torchvision
import torch.nn as nn
from HAM10000.models.conv2d_mtl import Conv2dMtl

def replace_conv2d(module):

    for name, child in module.named_children():

        if isinstance(child, nn.Conv2d):
            setattr(module, name, Conv2dMtl(
                child.in_channels, child.out_channels, child.kernel_size,
                child.stride, child.padding, child.dilation, child.groups,
                child.bias is not None))
        else:
            replace_conv2d(child)




class ResNet34(torch.nn.Module):
    def __init__(self, mtl=False):
        super().__init__()

        self.resnet = torchvision.models.resnet34(pretrained=True)
        if mtl:
            print("********************")
            replace_conv2d(self.resnet)

        self.n_features = self.resnet.fc.in_features
        del self.resnet.fc

        for param in self.resnet.parameters():
            param.requires_grad = True

    def forward(self, x):
        # Hemmer et al ===
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)

        features = torch.flatten(x, 1)
        return features

