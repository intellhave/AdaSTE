import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG16(nn.Module):
    """VGG16 used for Cifar10.
       This model is the Conv architecture used in paper "An empirical study of Binary NN optimization".
    """
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.2, batch_affine=False, have_bias=True):
        super(VGG16, self).__init__()
        self.in_features = in_features
        self.conv1 = nn.Conv2d(in_features, 64, kernel_size=3, padding=1,bias=have_bias)
        self.bn1 = nn.BatchNorm2d(64, eps=eps, momentum=momentum,affine=batch_affine)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1,bias=have_bias)
        self.bn2 = nn.BatchNorm2d(64, eps=eps, momentum=momentum,affine=batch_affine)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1,bias=have_bias)
        self.bn3 = nn.BatchNorm2d(128, eps=eps, momentum=momentum,affine=batch_affine)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1,bias=have_bias)
        self.bn4 = nn.BatchNorm2d(128, eps=eps, momentum=momentum,affine=batch_affine)


        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1,bias=have_bias)
        self.bn5 = nn.BatchNorm2d(256, eps=eps, momentum=momentum,affine=batch_affine)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1,bias=have_bias)
        self.bn6 = nn.BatchNorm2d(256, eps=eps, momentum=momentum,affine=batch_affine)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1,bias=have_bias)
        self.bn7 = nn.BatchNorm2d(256, eps=eps, momentum=momentum,affine=batch_affine)

        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, padding=1,bias=have_bias)
        self.bn8 = nn.BatchNorm2d(512, eps=eps, momentum=momentum,affine=batch_affine)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=have_bias)
        self.bn9 = nn.BatchNorm2d(512, eps=eps, momentum=momentum,affine=batch_affine)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=have_bias)
        self.bn10 = nn.BatchNorm2d(512, eps=eps, momentum=momentum,affine=batch_affine)


        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=have_bias)
        self.bn11 = nn.BatchNorm2d(512, eps=eps, momentum=momentum,affine=batch_affine)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=have_bias)
        self.bn12 = nn.BatchNorm2d(512, eps=eps, momentum=momentum,affine=batch_affine)
        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=have_bias)
        self.bn13 = nn.BatchNorm2d(512, eps=eps, momentum=momentum,affine=batch_affine)

        self.fc14 = nn.Linear(512, 512, bias=have_bias)
        self.bn14 = nn.BatchNorm1d(512,affine=batch_affine)

        self.fc15 = nn.Linear(512, 512, bias=have_bias)
        self.bn15 = nn.BatchNorm1d(512,affine=batch_affine)

        self.fc16 = nn.Linear(512, out_features, bias=have_bias)
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


def vgg16(**kwargs):
    num_classes = getattr(kwargs,'num_classes', 10)
    dataset = kwargs['dataset']
    if dataset=='cifar100':
        num_classes=100
    return VGG16(3, num_classes)
