import torch
import torch.nn as nn


class InstanceWhitening(nn.Module):

    def __init__(self, dim):
        super(InstanceWhitening, self).__init__()
        self.instance_standardization = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, x):
        return self.instance_standardization(x)