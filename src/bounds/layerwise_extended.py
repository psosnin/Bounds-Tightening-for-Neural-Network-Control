import numpy as np
from gurobipy import GRB

from ..utils.helpers import relu, getGurobiModel
from ..models.extended import addExtendedConstr


def getBounds(params, input_bounds, frac_binary):
    """
    Calculate the pre-activation bounds for the network with the given parameters by solving layerwise relaxed extended
    formulation.
    Paramters:
        params: list of tuples (W, b) where W is the weight matrix and b is the bias vector for each layer
        input_bounds: tuple (L, U) where L is the lower bound and U is the upper bound on the input
        frac_binary: fraction of binary vs continuous variables to use in the big-M model for each layer
    """
    bounds = [input_bounds]
    L, U = input_bounds
    W, b = zip(*params)
    bounds.append((b[0] + relu(W[0]) @ L - relu(-W[0]) @ U, b[0] + relu(W[0]) @ U - relu(-W[0]) @ L))

    for l in range(len(params) - 1):
        m = getGurobiModel()
        prev_bounds = [relu(b) for b in bounds[l]] if l > 0 else bounds[l]  # input can be negative
        hl_minus_1 = m.addMVar(W[l].shape[1], lb=prev_bounds[0], ub=prev_bounds[1])
        fb = frac_binary if isinstance(frac_binary, (int, float)) else frac_binary[l]
        hl = addExtendedConstr(m, hl_minus_1, W[l], b[l], bounds[-1], prev_bounds, int(b[l].size * fb))
        ML, MU = tighten(m, W[l+1] @ hl + b[l+1])
        bounds.append((ML, MU))
    return bounds


def tighten(m, obj):
    N = obj.size
    ML, MU = np.zeros(N), np.zeros(N)
    for i in range(N):
        m.setObjective(obj[i], GRB.MINIMIZE)
        m.reset(), m.optimize()
        ML[i] = m.objVal
        m.setObjective(obj[i], GRB.MAXIMIZE)
        m.reset(), m.optimize()
        MU[i] = m.objVal
    return ML, MU
