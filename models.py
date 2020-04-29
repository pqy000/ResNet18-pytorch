from torchvision import models
import torch.nn as nn
from utils import *
import torch.nn.functional as F

def model_A(num_classes, pretrained=True):
    model_resnet = models.resnet18(pretrained=pretrained)
    num_features = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_features, num_classes)
    return model_resnet

def model_B(num_classes, pretrained = False):
    model_resnet = ourModel(3, num_classes)
    print(model_resnet)

    return model_resnet

def conv3_3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=(3,3),stride=stride, padding=1, bias=False)

class residualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, same_shape=True):
        super().__init__()
        self.same_shape = same_shape
        if self.same_shape:
            stride = 1
        else:
            stride = 2
        self.conv1 = conv3_3(in_channel, out_channel, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = conv3_3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)

        if not self.same_shape:
            self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=(1,1), stride=2)
            self.bn3 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out), True)
        out = self.conv2(out)
        out = self.bn2(out)
        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(x+out, True)

class ResNet18(nn.Module):
    def __init__(self, in_channel, num_classes):
        super().__init__()

        self.conv0 = nn.Conv2d(in_channel, 64, kernel_size=[7,7], stride=2, padding=(3,3))
        self.bn0 = nn.BatchNorm2d(64)

        self.block1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
            residualBlock(64, 64, True),
            residualBlock(64, 64, True)
        )

        self.block2 = nn.Sequential(
            residualBlock(64, 128, False),
            residualBlock(128, 128, True)
        )

        self.block3 = nn.Sequential(
            residualBlock(128, 256, False),
            residualBlock(256, 256, True),
        )

        self.block4 = nn.Sequential(
            residualBlock(256, 512, False),
            residualBlock(512, 512, True)
        )

        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classfier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.bn0(self.conv0(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avg(x)
        x = x.view(x.shape[0], -1)
        x = self.classfier(x)
        return x

class ourModel(nn.Module):
    def __init__(self, in_channel, num_classes):
        super().__init__()

        self.conv0 = nn.Conv2d(in_channel, 8, kernel_size=[3,3], stride=2, padding=(1,1))
        self.conv0 = nn.Conv2d(8, 8, kernel_size=[3,3], stride=1, padding=(1, 1))
        self.conv0 = nn.Conv2d(8, 8, kernel_size=[3,3], stride=1, padding=(1, 1))
        self.bn0 = nn.BatchNorm2d(8)

        self.block1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
            residualBlock(8, 8, True),
            residualBlock(8, 8, True)
        )

        self.block2 = nn.Sequential(
            residualBlock(8, 16, False),
            residualBlock(16, 16, True)
        )

        self.block3 = nn.Sequential(
            residualBlock(16, 32, False),
            residualBlock(32, 32, True),
        )

        self.block4 = nn.Sequential(
            residualBlock(32, 64, False),
            residualBlock(64, 64, True)
        )

        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifer = nn.Linear(64, num_classes)
        #self.classfier = nn.Linear(64*7*7, 1000)
        #self.output = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.bn0(self.conv0(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avg(x)
        x = x.view(x.shape[0], -1)
        x = self.classfier(x)

        return x


def model_C(num_classes, pretrained = False):
    ## your code here
    pass