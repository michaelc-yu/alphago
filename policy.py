
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class PolicyNetwork(nn.Module):
    def __init__(self, input_channels=48, k=192):
        super(PolicyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, k, kernel_size=5, stride=1, padding=2)
        self.convs = nn.ModuleList([nn.Conv2d(k, k, kernel_size=3, stride=1, padding=1) for _ in range(11)])
        self.conv_last = nn.Conv2d(k, 1, kernel_size=1, stride=1)

        # Initialize weights using He initialization
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv_last.weight, mode='fan_out', nonlinearity='relu')

        # Initialize biases for all conv layers to zero
        nn.init.constant_(self.conv1.bias, 0)
        for conv in self.convs:
            nn.init.constant_(conv.bias, 0)
        nn.init.constant_(self.conv_last.bias, 0)

    def forward(self, x): # x needs to be dimension [input_channels, 19, 19]
        x = F.relu(self.conv1(x))
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x))
        x = self.conv_last(x)
        x = x.view(x.size(0), -1)
        # x = F.softmax(x, dim=1)
        return x


