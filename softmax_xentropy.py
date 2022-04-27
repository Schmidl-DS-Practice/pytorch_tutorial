import numpy as np
import torch
nn = torch.nn

# softmax numpy
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

x = np.array([2 ,1, .1])
outputs = softmax(x)
print(f"softmax numpy: {outputs}")

# softmax torch
x = torch.tensor([2 ,1, .1])
outputs = torch.softmax(x, dim=0)
print(f"softmax numpy: {outputs}")

# cross entropy numpy
def cross_entropy(actual, predicted):
    loss = -np.sum(actual) * np.log(predicted)
    return loss #/float(predicted.shape[0])

# y must be one hot encoded
# if class 0: [1 0 0]
# if class 1: [0 1 0]
# if class 2: [0 0 1]
y = np.array([1,0,0])

# y_pred has probabilities
y_pred_good = np.array([.7, .2, .1])
y_pred_bad = np.array([.1, .3, .6])
l1 = cross_entropy(y, y_pred_good)
l2 = cross_entropy(y, y_pred_bad)
print(f"loss1 numpy: {l1}")
print(f"loss2 numpy: {l2}")

# cross entropy torch
loss = nn.CrossEntropyLoss()

y = torch.tensor([0])

# n_samples * n_classes = 1x3
y_pred_good = torch.tensor([[2, 1, .1]])
y_pred_bad = torch.tensor([[.5, 2, 1]])

l1 = loss(y_pred_good, y)
l2 = loss (y_pred_bad, y)
print(f"l1: {l1.item()}")
print(f"l2: {l2.item()}")

_, predictions1 = torch.max(y_pred_good, 1)
_, predictions2 = torch.max(y_pred_bad, 1)
print(f"predictions1: {predictions1}")
print(f"predictions2: {predictions2}")

# 3 samples
y = torch.tensor([2, 0, 1])

# n_samples * n_classes = 3x3
y_pred_good2 = torch.tensor([[.01, 1, 2], [2, 1, .1], [.1, 3, .1]])
y_pred_bad2 = torch.tensor([[2.1, 1, .1], [.1, 1, 2.1], [.1, 3, .1]])

l3 = loss(y_pred_good2, y)
l4 = loss(y_pred_bad2, y)
print(f"l3: {l3.item()}")
print(f"l4: {l4.item()}")

_, predictions3 = torch.max(y_pred_good2, 1)
_, predictions4 = torch.max(y_pred_bad2, 1)
print(f"predictions: {predictions3}")
print(f"predictions2: {predictions4}")

# mutli class
# Multiclass problem
class NeuralNet2(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes) -> None:
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # no softmax at the end
        return out

model = NeuralNet2(input_size=28*28, hidden_size=5, num_classes=3)
criterion = nn.CrossEntropyLoss() # (applied softmax

# binary class
# Binary classification
class NeuralNet1(nn.Module):

    def __init__(self, input_size, hidden_size) -> None:
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # sigmoid at the end
        y_pred = torch.sigmoid(out)
        return y_pred

model = NeuralNet1(input_size=28*28, hidden_size=5)
criterion = nn.BCELoss() # (applied softmax