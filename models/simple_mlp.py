import torch
import torch.nn as nn
import torch.nn.functional as F 

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output

def simple_mlp(**kwargs):
    return SimpleMLP()


