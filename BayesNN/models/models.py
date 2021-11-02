import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPBinaryConnect(nn.Module):
    """Multi-Layer Perceptron used for MNIST. No convolution layers.
       This model is the MLP architecture used in paper "An empirical study of Binary NN optimization".
    """
    def __init__(self, in_features, out_features, num_units=2048, momentum=0.15, eps=1e-4,drop_prob=0,batch_affine=False):
        super(MLPBinaryConnect, self).__init__()
        self.in_features = in_features

        self.dropout1 = nn.Dropout(p=drop_prob)

        self.dropout2 = nn.Dropout(p=drop_prob)


        self.dropout3 = nn.Dropout(p=drop_prob)

        self.dropout4 = nn.Dropout(p=drop_prob)


        self.fc1 = nn.Linear(in_features, num_units, bias=False)
        self.bn1 = nn.BatchNorm1d(num_units, eps=eps, momentum=momentum,affine=batch_affine)

        self.fc2 = nn.Linear(num_units, num_units, bias=False)
        self.bn2 = nn.BatchNorm1d(num_units, eps=eps, momentum=momentum,affine=batch_affine)

        self.fc3 = nn.Linear(num_units, num_units, bias=False)
        self.bn3 = nn.BatchNorm1d(num_units, eps=eps, momentum=momentum,affine=batch_affine)

        self.fc4 = nn.Linear(num_units, out_features, bias=False)
        self.bn4 = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum,affine=batch_affine)


    def forward(self, x):
        x = x.view(-1, self.in_features)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(self.bn1(x))
        x = self.dropout2(x)


        x = self.fc2(x)
        x = F.relu(self.bn2(x))
        x = self.dropout3(x)


        x = self.fc3(x)
        x = F.relu(self.bn3(x))
        x = self.dropout4(x)


        x = self.fc4(x)
        x = self.bn4(x)

        return x  # if used NLL loss, the output should be changed to F.log_softmax(x,dim=1),

class MLP_CL_h100(nn.Module):
    """Multi-Layer Perceptron used for MNIST. No convolution layers.
       This model is the MLP architecture for continual learning
    """
    def __init__(self, in_features, out_features, num_units=100, momentum=0.15, eps=1e-4,batch_affine=True):
        super(MLP_CL_h100, self).__init__()
        self.in_features = in_features
        self.fc1 = nn.Linear(in_features, num_units,bias=False)
        self.bn1 = nn.BatchNorm1d(num_units, eps=eps, momentum=momentum,affine=batch_affine)

        self.fc2 = nn.Linear(num_units, num_units,bias=False)
        self.bn2 = nn.BatchNorm1d(num_units, eps=eps, momentum=momentum,affine=batch_affine)

        self.fc3 = nn.Linear(num_units, out_features,bias=False)
        self.bn3 = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum,affine=batch_affine)


    def forward(self, x):
        x = x.view(-1, self.in_features)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = self.bn3(x)

        return x  # if used NLL loss, the output should be changed to F.log_softmax(x,dim=1),

class VGGBinaryConnect(nn.Module):
    """VGG-like net used for Cifar10.
       This model is the Conv architecture used in paper "An empirical study of Binary NN optimization".
    """
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.2, batch_affine=False):
        super(VGGBinaryConnect, self).__init__()
        self.in_features = in_features
        self.conv1 = nn.Conv2d(in_features, 128, kernel_size=3, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(128, eps=eps, momentum=momentum,affine=batch_affine)


        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(128, eps=eps, momentum=momentum,affine=batch_affine)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(256, eps=eps, momentum=momentum,affine=batch_affine)

        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1,bias=False)
        self.bn4 = nn.BatchNorm2d(256, eps=eps, momentum=momentum,affine=batch_affine)


        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1,bias=False)
        self.bn5 = nn.BatchNorm2d(512, eps=eps, momentum=momentum,affine=batch_affine)

        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=False)
        self.bn6 = nn.BatchNorm2d(512, eps=eps, momentum=momentum,affine=batch_affine)


        self.fc1 = nn.Linear(512 * 4 * 4, 1024, bias=False)
        self.bn7 = nn.BatchNorm1d(1024,affine=batch_affine)

        self.fc2 = nn.Linear(1024, 1024, bias=False)
        self.bn8 = nn.BatchNorm1d(1024,affine=batch_affine)


        self.fc3 = nn.Linear(1024, out_features, bias=False)
        self.bn9 = nn.BatchNorm1d(out_features,affine=batch_affine)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn2(x))

        x = F.relu(self.bn3(self.conv3(x)))


        x = self.conv4(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn4(x))

        x = F.relu(self.bn5(self.conv5(x)))

        x = self.conv6(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn6(x))

        x = x.view(-1, 512 * 4 * 4)

        x = self.fc1(x)
        x = F.relu(self.bn7(x))

        x = self.fc2(x)
        x = F.relu(self.bn8(x))

        x = self.fc3(x)
        x = self.bn9(x)

        return x  # if used NLL loss, the output should be changed to F.log_softmax(x,dim=1),

class VGG16(nn.Module):
    """VGG16 used for Cifar10.
       This model is the Conv architecture used in paper "An empirical study of Binary NN optimization".
    """
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.2, batch_affine=False):
        super(VGG16, self).__init__()
        self.in_features = in_features
        self.conv1 = nn.Conv2d(in_features, 64, kernel_size=3, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=eps, momentum=momentum,affine=batch_affine)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(64, eps=eps, momentum=momentum,affine=batch_affine)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(128, eps=eps, momentum=momentum,affine=batch_affine)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1,bias=False)
        self.bn4 = nn.BatchNorm2d(128, eps=eps, momentum=momentum,affine=batch_affine)


        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1,bias=False)
        self.bn5 = nn.BatchNorm2d(256, eps=eps, momentum=momentum,affine=batch_affine)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1,bias=False)
        self.bn6 = nn.BatchNorm2d(256, eps=eps, momentum=momentum,affine=batch_affine)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1,bias=False)
        self.bn7 = nn.BatchNorm2d(256, eps=eps, momentum=momentum,affine=batch_affine)

        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, padding=1,bias=False)
        self.bn8 = nn.BatchNorm2d(512, eps=eps, momentum=momentum,affine=batch_affine)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=False)
        self.bn9 = nn.BatchNorm2d(512, eps=eps, momentum=momentum,affine=batch_affine)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=False)
        self.bn10 = nn.BatchNorm2d(512, eps=eps, momentum=momentum,affine=batch_affine)


        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=False)
        self.bn11 = nn.BatchNorm2d(512, eps=eps, momentum=momentum,affine=batch_affine)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=False)
        self.bn12 = nn.BatchNorm2d(512, eps=eps, momentum=momentum,affine=batch_affine)
        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=False)
        self.bn13 = nn.BatchNorm2d(512, eps=eps, momentum=momentum,affine=batch_affine)

        self.fc14 = nn.Linear(512, 512, bias=False)
        self.bn14 = nn.BatchNorm1d(512,affine=batch_affine)

        self.fc15 = nn.Linear(512, 512, bias=False)
        self.bn15 = nn.BatchNorm1d(512,affine=batch_affine)

        self.fc16 = nn.Linear(512, out_features, bias=False)
        self.bn16 = nn.BatchNorm1d(out_features,affine=batch_affine)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))) # layer 1: outsize 64
        x = F.relu(self.bn2(F.max_pool2d(self.conv2(x), 2, 2))) # layer 2: outsize 128

        x = F.relu(self.bn3(self.conv3(x))) # layer 3: outsize 128
        x = F.relu(self.bn4(F.max_pool2d(self.conv4(x), 2, 2))) # layer 4: outsize 256

        x = F.relu(self.bn5(self.conv5(x))) # layer 5: outsize 256
        x = F.relu(self.bn6(self.conv6(x))) # layer 6: outsize 256
        x = F.relu(self.bn7(F.max_pool2d(self.conv7(x), 2, 2))) # layer 7: outsize 512

        x = F.relu(self.bn8(self.conv8(x))) # layer 8: outsize 512
        x = F.relu(self.bn9(self.conv9(x))) # layer 9: outsize 512
        x = F.relu(self.bn10(F.max_pool2d(self.conv10(x), 2, 2))) # layer 10: outsize 512

        x = F.relu(self.bn11(self.conv11(x))) # layer 8: outsize 512
        x = F.relu(self.bn12(self.conv12(x))) # layer 9: outsize 512
        x = F.relu(self.bn13(F.max_pool2d(self.conv13(x), 2, 2))) # layer 10: outsize 512
        
        x = x.view(-1, 512)

        x = self.fc14(x)
        x = F.relu(self.bn14(x))

        x = self.fc15(x)
        x = F.relu(self.bn15(x))

        x = self.fc16(x)
        x = self.bn16(x)
        
        return x  # if used NLL loss, the output should be changed to F.log_softmax(x,dim=1),



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
