# 1) design model(input, output, forward pass)
# 2) construct loss and optimizer
# 3) training loop
# - forward pass: compute prediction
# - backward pass: gradients
# - update weights
import torch
import torch.nn as nn
# f = w * x

# f = 2 * x
x = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)
n_samples, n_features = x.shape
print(n_samples, n_features)

# model = nn.Linear(in_features=n_features, out_features=n_features)

class LinearRegression(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

model = LinearRegression(n_features, n_features)

print(f"prediction before training: f(5)={model(X_test).item()}")

# training
lr = 0.01
n_iters = 100
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

for epoch in range(n_iters):
    # prediction = forward pass
    y_p = model(x)

    # loss
    l = loss(y, y_p)

    # gradients = backward pass
    l.backward() # dl/dw

    # update weights
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f"epoch {epoch+1}: w = {w[0][0].item()}, loss = {l}")


print(f"prediction after training: f(5)={model(X_test).item()}")