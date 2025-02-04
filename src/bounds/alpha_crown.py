import torch
from torch import nn
import numpy as np

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm


def getBounds(params, input_bounds, log=None, method='CROWN-Optimized', iters=20):
    """
    Compute intermediate bounds on the network using the alpha-CROWN algorithm given
    the input domain L <= x <= U.
    Parameters:
        params: list of tuples (Wi, bi) for the i = 1, ..., L layers in the network
        input_bounds: tuple of vectors (L, U) such that L <= x <= U
        log: optional weights and biases logging
    """
    # define model and wrap in in auto-lirpa module
    input_size = input_bounds[0].size
    model = RescaledNN(params, input_bounds)
    lirpa_model = BoundedModule(model, torch.empty(1, input_size))

    # define perturbation where eps = 1 because of the rescaling we apply
    ptb = PerturbationLpNorm(norm=np.inf, eps=1)
    bounded_input = BoundedTensor(torch.zeros((1, input_size)), ptb)
    lirpa_model.set_bound_opts({'optimize_bound_args': {'iteration': iters, 'lr_alpha': 0.1}})
    lb, ub = lirpa_model.compute_bounds(x=(bounded_input,), method=method)
    intermediate = lirpa_model.save_intermediate()
    result = [input_bounds]
    for k, v in intermediate.items():
        if "input" in k:  # gets pre-relu bounds
            result.append((v[0].detach().cpu().numpy().flatten(), v[1].detach().cpu().numpy().flatten()))
    result.append((lb.detach().cpu().numpy().flatten(), ub.detach().cpu().numpy().flatten()))
    return result


class RescaledNN(torch.nn.Module):
    """
    Define a pytorch model with the given parameters. The input of the model lies within the interval [-1, 1] and is
    rescaled to lie in the range of x_in_bounds.
    """

    def __init__(self, params, x_in_bounds):
        """
        params: list of tuples (Wi, bi) for the i = 1, ..., L layers in the network
        input_bounds: tuple of vectors (L, U) such that L <= x <= U
        """
        super(RescaledNN, self).__init__()

        # compute scaling factors
        self.x_mu = torch.from_numpy((x_in_bounds[1] + x_in_bounds[0]) / 2).float()
        self.x_rad = torch.from_numpy((x_in_bounds[1] - x_in_bounds[0]) / 2).float()

        self.layers = []
        for W, b in params:
            linear = torch.nn.Linear(W.shape[1], W.shape[0])
            with torch.no_grad():
                linear.weight.copy_(torch.from_numpy(W))
                linear.bias.copy_(torch.from_numpy(b))
            self.layers.append(linear)
            self.layers.append(torch.nn.ReLU())
        self.layers.pop()  # pop last ReLU
        self.model = torch.nn.Sequential(*self.layers)

    def forward(self, y):
        x = self.x_rad * y + self.x_mu  # apply rescaling
        return self.model(x)


def getMultiStepBounds(params, input_bounds, A, B, K, method='CROWN-Optimized', iters=50):
    """
    Calculate the pre-activation bounds for the network with the given parameters over multiple time-steps of the linear
    system

        x = A x + B f_nn(x)

    using auto-Lirpa's implementation of IBP.
    Parameters:
        params: list of tuples (W, b) where W is the weight matrix and b is the bias vector for each layer
        input_bounds: tuple (L, U) where L is the lower bound and U is the upper bound on the input
        A: system state dynamics
        B: system control dynamics
        K: number of time-steps
        method: auto-lirpa algorithm to use
        iters: number of iterations for the optimization (only used for CROWN-Optimized method)
        log: optional weights and biases logging
    Returns:
        bounds: list of list of bounds such that bounds[k][l] is the bounds on the l-th hidden variable at timestep k
    """

    # define model and wrap in in auto-lirpa module
    input_size = input_bounds[0].size
    model = MultiStepModel(params, input_bounds, A, B, K)
    lirpa_model = BoundedModule(model, torch.empty(1, input_size))

    # define perturbation where eps = 1 because of the rescaling we apply
    ptb = PerturbationLpNorm(norm=np.inf, eps=1)
    bounded_input = BoundedTensor(torch.zeros((1, input_size)), ptb)
    lirpa_model.set_bound_opts({'optimize_bound_args': {'iteration': iters, 'lr_alpha': 0.1}})
    lb, ub = lirpa_model.compute_bounds(x=(bounded_input,), method=method)

    # get intermediate bounds
    intermediate = lirpa_model.save_intermediate()

    # ONNX model has the following structure:
    # first 2L nodes are the parameters of the network
    # next 4 are the rescaling of the input
    # the next 2L are the neural network
    # for k == 0, the next 2 are the linear dynamics constants
    # then 2 are the linear dynamics update
    # the 2L are the neural network
    # then 2 are the linear dynamics update
    # the 2L are the neural network
    # etc
    result = []
    L = len(params)
    offset = 2 * L + 4

    intermediate = list(intermediate.values())

    for k in range(K):
        # print(offset, intermediate[offset][0].shape)
        result.append([(
            intermediate[offset][0].detach().cpu().numpy().flatten(),
            intermediate[offset][1].detach().cpu().numpy().flatten()
        )])
        for i in range(L):
            # print(offset + 2 * i + 1, intermediate[offset + 2 * i + 1][0].shape)
            result[-1].append((
                intermediate[offset + 2 * i + 1][0].detach().cpu().numpy().flatten(),
                intermediate[offset + 2 * i + 1][1].detach().cpu().numpy().flatten()
            ))
        if k == 0:
            offset += 2
        offset += 2 * L + 2

    result.append([(lb.detach().cpu().numpy().flatten(), ub.detach().cpu().numpy().flatten())])
    return result


class MultiStepModel(torch.nn.Module):
    """
    The forward pass is a composition of a neural network applied to linear dynamical system 

        x = Ax + Bu

    where u = f_nn(x).
    The input of the model lies within the interval [-1, 1] and is rescaled to lie in the range of x_in_bounds. 
    """

    def __init__(self, params, x_in_bounds, A, B, K):
        """
        params: list of tuples (Wi, bi) for the i = 1, ..., L layers in the network
        x_in_bounds: tuple of vectors (L, U) such that L <= x <= U
        A: state dynamics matrix
        B: control dynamics matrix
        K: number of time steps
        """
        super(MultiStepModel, self).__init__()
        # save dynamics
        self.A = torch.from_numpy(A).float()
        self.B = torch.from_numpy(B).float()
        self.K = K

        # compute scaling factors
        self.x_mu = torch.from_numpy((x_in_bounds[1] + x_in_bounds[0]) / 2).float()
        self.x_rad = torch.from_numpy((x_in_bounds[1] - x_in_bounds[0]) / 2).float()

        self.layers = []
        for W, b in params:
            linear = torch.nn.Linear(W.shape[1], W.shape[0])
            with torch.no_grad():
                linear.weight.copy_(torch.from_numpy(W))
                linear.bias.copy_(torch.from_numpy(b))
            self.layers.append(linear)
            self.layers.append(torch.nn.ReLU())
        self.layers.pop()  # pop last ReLU
        self.model = torch.nn.Sequential(*self.layers)

    def forward(self, y):
        # get y in the correct shape
        if len(y.shape) == 1:
            y = y.unsqueeze(0)
        x = self.x_rad * y + self.x_mu  # apply rescaling
        for _ in range(self.K):
            u = self.model(x)
            x = x @ self.A.T + u @ self.B.T
        return x
