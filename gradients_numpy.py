import numpy as np

# f = w * x

# f = 2 * x
x = np.array([1,2,3,4], dtype=np.float32)
y = np.array([2,4,6,8], dtype=np.float32)

w = 0.0

# model prediction
def forward(x):
    return w * x

# loss
def loss(y, y_pred):
    return ((y_pred-y)**2).mean()

# gradient
# mse = 1/n * (w*x - y )**2
# dJ/dw = 1/n 2x (w*x - y)
def gradient(x, y, y_pred):
    return np.dot(2 * x, y_pred - y).mean()

print(f"prediction before training: f(5)={forward(5)}")

# training
lr = 0.01
n_iters = 20
for epoch in range(n_iters):
    # prediction = forward pass
    y_p = forward(x)

    # loss
    l = loss(y, y_p)

    # gradients
    dw = gradient(x, y, y_p)

    # update weights
    w -= lr * dw

    print(f"epoch {epoch+1}: w = {w}, loss = {l}")


print(f"prediction after training: f(5) = {forward(5)}")