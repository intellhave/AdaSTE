import torch
import torch.nn as nn
import torch.nn.functional as F 

from .fenbp_modules import FLinear



class SimpleMLPFenBP(nn.Module):
    def __init__(self):
        super(SimpleMLPFenBP, self).__init__()
        in_features = 28*28
        out_features = 10
        self.in_features = in_features
        num_units=2048
        momentum=0.15
        eps=1e-4
        drop_prob=0
        batch_affine=False

        self.dropout1 = nn.Dropout(p=drop_prob)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.dropout3 = nn.Dropout(p=drop_prob)
        self.dropout4 = nn.Dropout(p=drop_prob)


        self.fc1 = FLinear(in_features, num_units, bias=False)
        self.bn1 = nn.BatchNorm1d(num_units, eps=eps, momentum=momentum,affine=batch_affine)

        self.fc2 = FLinear(num_units, num_units, bias=False)
        self.bn2 = nn.BatchNorm1d(num_units, eps=eps, momentum=momentum,affine=batch_affine)

        self.fc3 = FLinear(num_units, num_units, bias=False)
        self.bn3 = nn.BatchNorm1d(num_units, eps=eps, momentum=momentum,affine=batch_affine)

        self.fc4 = FLinear(num_units, out_features, bias=False)
        self.bn4 = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum,affine=batch_affine)

        # self.fc1 = nn.Linear(784, 512)
        # self.fc2 = nn.Linear(512, 128)
        # self.fc3 = nn.Linear(128, 10)

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
        # x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output


def simple_mlp_fenbp(**kwargs):
    return SimpleMLPFenBP()


