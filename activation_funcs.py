from turtle import forward
import torch
nn = torch.nn
F = nn.functional

# option 1 (create nn modules)
class NeuralNet(nn.Module):

    def __init__(self, input_size, hidden_size, ) -> None:
        super(NeuralNet).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        # nn.Sigmoid
        # nn.Softmax
        # nn.TanH
        # nn.LeakyReLU
        self.linear2 = nn.Linear(hidden_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out

# option 2 (use activation functions directly in forward pass)
class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size) -> None:
        super(NeuralNet2).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # F.leaky_relu()
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        # torch.Softmax
        # torch.TanH
        # torch.LeakyReLU
        return out