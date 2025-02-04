import matplotlib.pyplot as plt
import numpy as np
from gurobipy import GRB

from ..utils.helpers import relu


def addBigMToModel(m, in_var, out_var, params, pre_act_bounds, relax=True):
    """
    Add the big-M constraints to the gurobi model for a neural network with the given parameters.
    Parameters:
        m: gurobi model
        in_var: input variable to the network
        out_var: output variable from the network
        params: list of tuples (Wi, bi) for the i = 1, ..., L layers in the network
        pre_act_bounds: list of tuples (MLi, MUi) such that MLi <= Wi @ hl-1 + bi <= MUi for i = 0, ..., L
    """
    L = len(params)
    hl = [in_var]

    # add hidden layers
    for i in range(L-1):
        W, b = params[i]
        N_binary = 0 if relax else b.size
        hl.append(addBigMConstr(m, hl[-1], W, b, pre_act_bounds[i + 1], N_binary))

    # add output layer (no bigM because no relu)
    (W, b) = params[-1]
    m.addConstr(out_var == W @ hl[-1] + b)


def addBigMConstr(model, hl_minus_1, Wl, bl, pre_act_bounds_l, N_binary):
    """
    Add the big M constraints for hl = ReLU(Wl @ hl_minus_1 + bl).
    Parameters:
        model: gurobi model to add the constraints to
        hl_minus_1: previous layer variable
        Wl: weight matrix for the current layer
        bl: bias vector for the current layer
        pre_act_bounds_l: tuple (MLl, MUl) such that MLl <= Wl @ hl_minus_1 + bl <= MUl
        N_binary: number of binary (instead of continuous) variables to use in this layer
    """
    assert N_binary <= bl.size and N_binary >= 0
    MLl, MUl = pre_act_bounds_l
    # define next layer variable
    hl = model.addMVar(bl.size, lb=relu(MLl), ub=relu(MUl))

    # randomly allocate binary and continuous variables
    binary_indices = np.random.choice(hl.size, N_binary, replace=False)
    vtypes = np.array([GRB.CONTINUOUS] * hl.size)
    vtypes[binary_indices] = GRB.BINARY
    zl = model.addMVar(hl.size, lb=0, ub=1, vtype=vtypes)

    # add big M constraints
    model.addConstr(hl >= Wl @ hl_minus_1 + bl)
    model.addConstr(hl <= Wl @ hl_minus_1 + bl - MLl * (1 - zl))
    model.addConstr(hl <= MUl * zl)
    model.addConstr(hl >= 0)
    model.addConstr(Wl @ hl_minus_1 + bl <= MUl)  # this is not encoded by the original bigM formulation when u < 0
    model.addConstr(Wl @ hl_minus_1 + bl >= MLl)

    return hl
