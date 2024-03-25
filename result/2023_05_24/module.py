import torch
from torch import optim
from torch import nn


# 构建MLP感知机类，继承nn.Module，并调用了Linear的子module
class MLP(nn.Module):
    def __init__(self, OneHot_len):
        super().__init__()
        self.lin1 = nn.Linear(3 * OneHot_len, 256)
        self.lin2 = nn.Linear(256, 3)

    def forward(self, xb):
        temp = self.lin1(xb)
        return self.lin2(temp)



