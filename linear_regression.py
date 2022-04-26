# 1) design model(input, output, forward pass)
# 2) construct loss and optimizer
# 3) training loop
# - forward pass: compute prediction
# - backward pass: gradients
# - update weights

import torch
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

nn = torch.nn

# 0) prepare data
x_np, y_np = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
x = torch.from_numpy(x_np.astype(np.float32))
y = torch.from_numpy(y_np.astype(np.float32))

y = y.view(y.shape[0], 1)

n_samples, n_features = x.shape

# 1) model
model = nn.Linear(n_features, 1)

# 2) loss and optimizer
lr = .01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# 3) training loop
n_epochs = 100
for epoch in range(n_epochs):
    # forward pass and loss
    y_pred = model(x)
    loss = criterion(y_pred, y)

    # backward pass
    loss.backward()

    # update
    optimizer.step()

    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f"epoch: {epoch+1}, loss = {loss.item()}")

# plot
predicted = model(x).detach().numpy()
plt.plot(x_np, y_np, "ro")
plt.plot(x_np, predicted, "b")
plt.show()
