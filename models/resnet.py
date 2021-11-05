import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import math

__all__ = ['resnet']

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def init_model(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        # elif isinstance(m, nn.BatchNorm2d):
        #     m.weight.data.fill_(1)
        #     m.bias.data.zero_()


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



# class ResNet(nn.Module):

#     def __init__(self):
#         super(ResNet, self).__init__()

#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         x = self.bnout(x)
#         return x


# class ResNet_imagenet(ResNet):

#     def __init__(self, num_classes=1000,
#                  block=Bottleneck, layers=[3, 4, 23, 3]):
#         super(ResNet_imagenet, self).__init__()
#         self.inplanes = 64
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#         self.avgpool = nn.AvgPool2d(7)
#         self.fc = nn.Linear(512 * block.expansion, num_classes)
#         self.bnout = nn.BatchNorm2d(num_classes, affine=False)

#         init_model(self)
#         self.regime = {
#             0: {'optimizer': 'SGD', 'lr': 1e-1,
#                 'weight_decay': 1e-4, 'momentum': 0.9},
#             30: {'lr': 1e-2},
#             60: {'lr': 1e-3, 'weight_decay': 0},
#             90: {'lr': 1e-4}
#         }


# class ResNet_cifar10(ResNet):

#     def __init__(self, num_classes=10,
#                  block=BasicBlock, depth=18):
#         super(ResNet_cifar10, self).__init__()
#         self.inplanes = 16
#         n = int((depth - 2) / 6)
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = lambda x: x
#         self.layer1 = self._make_layer(block, 16, n)
#         self.layer2 = self._make_layer(block, 32, n, stride=2)
#         self.layer3 = self._make_layer(block, 64, n, stride=2)
#         self.layer4 = lambda x: x
#         self.avgpool = nn.AvgPool2d(8)
#         self.fc = nn.Linear(64, num_classes)
#         self.bnout = nn.BatchNorm2d(num_classes, affine=False)

#         init_model(self)
#         # TTQ's setting
#         # self.regime = {
#         #     0: {'optimizer': 'SGD', 'lr': 1e-1,
#         #         'weight_decay': 2e-4, 'momentum': 0.9},
#         #     81: {'lr': 1e-2},
#         #     122: {'lr': 1e-3},
#         #     299: {'lr': 1e-4}
#         # }
#         # gluon's setting
#         self.regime = {
#             0: {'optimizer': 'SGD', 'lr': 1e-1,
#                 'weight_decay': 1e-4, 'momentum': 0.9},
#             100: {'lr': 1e-2},
#             150: {'lr': 1e-3},
#         }
#         # self.regime = {
#         #     0: {'optimizer': 'SGD', 'lr': 1e-1,
#         #         'weight_decay': 1e-4, 'momentum': 0.9},
#         #     81: {'lr': 1e-2},
#         #     122: {'lr': 1e-3, 'weight_decay': 0},
#         #     164: {'lr': 1e-4}
#         # }

# RESNET stuff
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=False, eps=1e-5, momentum=0.2)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=False, eps=1e-5, momentum=0.2)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes, affine=False, eps=1e-5, momentum=0.2)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=False, eps=1e-5, momentum=0.2)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=False, eps=1e-5, momentum=0.2)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes, affine=False, eps=1e-5, momentum=0.2)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes, affine=False, eps=1e-5, momentum=0.2)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, input_channels=3, imsize=32, output_dim=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.input_channels = input_channels
        self.imsize = imsize
        self.output_dim = output_dim
        self.stride1 = 1
        if imsize == 64:    # tinyimagenet
            self.stride1 = 2

        self.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=3, stride=self.stride1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=False, eps=1e-5, momentum=0.2)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, self.output_dim)
        self.bnout = nn.BatchNorm1d(self.output_dim, affine=False) # Rasmus: added this


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.bnout(out) # Rasmus: added this
        return out



def ResNet18(input_channels=3, imsize=32, output_dim=10):
    return ResNet(BasicBlock, [2,2,2,2], input_channels, imsize, output_dim)


def ResNet34(input_channels=3, imsize=32, output_dim=10):
    return ResNet(BasicBlock, [3,4,6,3], input_channels, imsize, output_dim)


def ResNet50(input_channels=3, imsize=32, output_dim=10):
    return ResNet(Bottleneck, [3,4,6,3], input_channels, imsize, output_dim)


def ResNet101(input_channels=3, imsize=32, output_dim=10):
    return ResNet(Bottleneck, [3,4,23,3], input_channels, imsize, output_dim)


def ResNet152(input_channels=3, imsize=32, output_dim=10):
    return ResNet(Bottleneck, [3,8,36,3], input_channels, imsize, output_dim)


def resnet(**kwargs):
        return ResNet18(input_channels=3, imsize=32, output_dim=10)
