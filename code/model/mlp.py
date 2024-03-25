import torch
from torch import optim
from torch import nn


# 构建MLP感知机类，继承nn.Module，并调用了Linear的子module
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super().__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, out_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = x.to(self.device)
        temp = self.lin1(x)
        return self.lin2(temp)



