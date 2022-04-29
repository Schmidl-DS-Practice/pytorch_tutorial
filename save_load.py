from turtle import forward
import torch
nn = torch.nn

class Model(nn.Module):
    def __init__(self, n_input_features) -> None:
        super().__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = Model(n_input_features=6)

FILE = "model.pth"

## LAZY OPTION
# torch.save(model, FILE)

# model = torch.load(FILE)
# model.eval()

# for param in model.parameters():
#     print(param)

## NOT LAZY OPTION - PREFERRED OPTION
torch.save(model.state_dict(), FILE)

loaded_model = Model(n_input_features=6)
loaded_model.load_state_dict(torch.load(FILE))
loaded_model.eval()

for param in loaded_model.parameters():
    print(param)


###########load checkpoint#####################
# learning_rate = 0.01
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# checkpoint = {
# "epoch": 90,
# "model_state": model.state_dict(),
# "optim_state": optimizer.state_dict()
# }
# print(optimizer.state_dict())
# FILE = "checkpoint.pth"
# torch.save(checkpoint, FILE)

# model = Model(n_input_features=6)
# optimizer = optimizer = torch.optim.SGD(model.parameters(), lr=0)

# checkpoint = torch.load(FILE)
# model.load_state_dict(checkpoint['model_state'])
# optimizer.load_state_dict(checkpoint['optim_state'])
# epoch = checkpoint['epoch']

# model.eval()
# # - or -
# # model.train()

# print(optimizer.state_dict())

# Remember that you must call model.eval() to set dropout and batch normalization layers
# to evaluation mode before running inference. Failing to do this will yield
# inconsistent inference results. If you wish to resuming training,
# call model.train() to ensure these layers are in training mode.

""" SAVING ON GPU/CPU
# 1) Save on GPU, Load on CPU
device = torch.device("cuda")
model.to(device)
torch.save(model.state_dict(), PATH)
device = torch.device('cpu')
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location=device))
# 2) Save on GPU, Load on GPU
device = torch.device("cuda")
model.to(device)
torch.save(model.state_dict(), PATH)
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.to(device)
# Note: Be sure to use the .to(torch.device('cuda')) function
# on all model inputs, too!
# 3) Save on CPU, Load on GPU
torch.save(model.state_dict(), PATH)
device = torch.device("cuda")
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  # Choose whatever GPU device number you want
model.to(device)
# This loads the model to a given GPU device.
# Next, be sure to call model.to(torch.device('cuda')) to convert the model’s parameter tensors to CUDA tensors
"""