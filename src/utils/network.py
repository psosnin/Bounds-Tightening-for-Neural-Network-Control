import numpy as np
import torch
import torch.nn as nn

from .helpers import relu


def getParameters(dir_name):
    """
    Load neural network parameters from file into a list of tuples (Wi, bi) and into a pytorch model.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params = list(np.load(f"{dir_name}/params.npy", allow_pickle=True))
    params = list(zip(params[::2], params[1::2]))
    shape = [params[0][0].shape[1]] + [p[0].shape[0] for p in params]
    print("shape = ", shape)
    model = NeuralNet(shape)
    model.load_state_dict(torch.load(f"{dir_name}/model.ckpt", map_location=device))
    return params, model


class NeuralNet(nn.Module):
    """
    Fully connected neural network with ReLU activation function. There is no activation function at the output layer.
    """

    def __init__(self, shape):
        super(NeuralNet, self).__init__()
        self.input_size = shape[0]
        self.relu = nn.ReLU()
        # fully connected network
        self.linear = nn.ModuleList([nn.Linear(shape[i], shape[i+1]) for i in range(len(shape) - 1)])

    def forward(self, h):
        for layer in self.linear[:-1]:
            h = layer(h)
            h = self.relu(h)
        h = self.linear[-1](h)  # no relu at the end
        return h


def forward(x, params):
    """
    Perform a forward pass through a neural network with parameters W and b.
    """
    h = x
    for Wl, bl in params[:-1]:
        h = relu(Wl @ h + bl)
    W, b = params[-1]
    h = W @ h + b  # no relu at the end
    return h
