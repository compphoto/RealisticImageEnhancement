import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import numpy as np
import scipy.ndimage.filters as filters
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
    return conv 

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, freeze=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.out_channels = 1

        self.output0 = self._make_output(64, readout=self.out_channels) 
        self.output1 = self._make_output(256, readout=self.out_channels) 
        self.output2 = self._make_output(512, readout=self.out_channels) 
        self.output3 = self._make_output(1024, readout=self.out_channels) 
        self.output4 = self._make_output(2048, readout=self.out_channels) 

        self.combined = self._make_output(5, sigmoid=True)  # use sigmoid for activation in the last layer.

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_output(self, planes, readout=1, sigmoid=False):
        layers = [
            nn.Conv2d(planes, readout, kernel_size=3, padding=1),
            nn.BatchNorm2d(readout),
        ]
        if sigmoid:
            layers.append(nn.Sigmoid())
        else:
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, decode=False):
        h, w = x.size(2), x.size(3)
        x = self.conv1(x)
        x = self.bn1(x)
        out0 = self.relu(x)
        x = self.maxpool(out0)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        out0 = self.output0(out0)
        r, c = out0.size(2), out0.size(3)
        out1 = self.output1(out1)
        out2 = self.output2(out2)
        out3 = self.output3(out3)
        out4 = self.output4(out4)

        if decode:
            return [out0, out1, out2, out3, out4]

        out1 = F.interpolate(out1, (r, c))
        out2 = F.interpolate(out2, (r, c))
        out3 = F.interpolate(out3, (r, c))
        out4 = F.interpolate(out4, (r, c))

        x = torch.cat([out0, out1, out2, out3, out4], dim=1)

        x = self.combined(x)
        x = F.interpolate(x, (h, w))
        return x


def resnet50(model_path, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if model_path is None:
        print ("Training from scratch.")
    else:
        model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
        model_state = model.state_dict()
        loaded_model = torch.load(model_path)
        if "state_dict" in loaded_model:
            loaded_model = loaded_model['state_dict']
        pretrained = {k[7:]:v for k, v in loaded_model.items() if k[7:] in model_state}
        if len(pretrained) == 0:
            pretrained = {k:v for k, v in loaded_model.items() if k in model_state}
        model_state.update(pretrained)
        model.load_state_dict(model_state)
        print ("Model loaded", model_path)
    return model
