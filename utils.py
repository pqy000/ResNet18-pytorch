import torch.nn as nn

def conv3_3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=(3,3),stride=stride, padding=1, bias=False)
