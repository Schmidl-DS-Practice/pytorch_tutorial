import numpy as np
import torch
nn = torch.nn

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

x = np.array([2 ,1, .1])
outputs = softmax(x)
print(f"softmax numpy: {outputs}")