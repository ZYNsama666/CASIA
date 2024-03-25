import torch
from torch import optim
from torch import nn
import torch.nn.functional as f


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, out_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = x.to(self.device)
        x = x.unsqueeze(1)
        scores, _ = self.lstm(x)
        scores = self.fc(scores)
        scores = scores.squeeze(1)
        return scores
