import matplotlib.pyplot as plt
import numpy as np
from gurobipy import GRB

from ..utils.helpers import relu


def addExtendedToModel(m, in_var, out_var, params, pre_act_bounds, relax=False):
    """
    Add the extended formulation constraints to the gurobi model for a neural network with the given parameters.
    Parameters:
        m: gurobi model
        in_var: input variable to the network
        out_var: output variable from the network
        params: list of tuples (Wi, bi) for the i = 1, ..., L layers in the network
        pre_act_bounds: list of tuples (MLi, MUi) for i = 0, ..., L such that MLi <= Wi @ hl-1 + bi <= MUi
    """
    L = len(params)
    assert L >= 2  # network must have at least 1 hidden layer
    hl = [in_var]

    # add hidden layers
    for i in range(L-1):
        Wl, bl = params[i]
        N_binary = 0 if relax else bl.size
        post_act_bounds = [relu(bd) for bd in pre_act_bounds[i]] if i >= 1 else pre_act_bounds[i]
        hl.append(addExtendedConstr(m, hl[-1], Wl, bl, pre_act_bounds[i+1], post_act_bounds, N_binary))

    # add output layer (no relu)
    (W, b) = params[-1]
    m.addConstr(out_var == W @ hl[-1] + b)


def addExtendedConstr(model, hl_minus_1, Wl, bl, pre_act_bound_l, post_act_bound_l_minus_1, N_binary):
    """
    Add extended formulation constraints for hl = ReLU(Wl @ hl_minus_1 + bl).
    Parameters:
        model: gurobi model to add the constraints to
        hl_minus_1: previous layer variable
        Wl: weight matrix for the current layer
        bl: bias vector for the current layer
        pre_act_bounds_l: tuple (MLl, MUl) such that MLl <= Wl @ hl_minus_1 + bl <= MUl
        post_bounds_l_minus_1: tuple (Ll_minus_1, Ul_minus_1) such that Ll_minus_1 <= hl_minus_1 <= Ul_minus_1
        N_binary: number of binary (instead of continuous) variables to use in this layer
    """
    assert pre_act_bound_l[0].shape == bl.shape
    assert post_act_bound_l_minus_1[0].shape == hl_minus_1.shape
    # constants for the extended formulation
    MLl, MUl = pre_act_bound_l
    Ll_minus_1, Ul_minus_1 = post_act_bound_l_minus_1
    nl = bl.size
    # define extended variables for previous layer.
    # note that we index h^{l-1, +, i} as h_pos[i]
    # h_neg[i] is implicitly defined as hl_minus_1 - h_pos[i]
    h_pos = model.addMVar(Wl.shape, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    # define variables for next layer
    hl = model.addMVar(nl, lb=relu(MLl), ub=relu(MUl))
    # randomly allocate binary and continuous variables
    binary_indices = np.random.choice(hl.size, N_binary, replace=False)
    vtypes = np.array([GRB.CONTINUOUS] * hl.size)
    vtypes[binary_indices] = GRB.BINARY
    zl = model.addMVar(hl.size, ub=1, vtype=vtypes)

    # add bigM formulation constraints
    model.addConstr(hl >= Wl @ hl_minus_1 + bl)
    model.addConstr(hl <= Wl @ hl_minus_1 + bl - MLl * (1 - zl))
    model.addConstr(hl <= MUl * zl)
    model.addConstr(hl >= 0)
    model.addConstr(Wl @ hl_minus_1 + bl <= MUl)  # this is not encoded by the original bigM formulation when u < 0
    model.addConstr(Wl @ hl_minus_1 + bl >= MLl)

    # add extended formulation constraints
    model.addConstr(hl == (Wl * h_pos).sum(1) + bl * zl)

    model.addConstr(0 <= (Wl * h_pos).sum(1) + bl * zl)
    model.addConstr(0 >= (Wl * (hl_minus_1[None] - h_pos)).sum(1) + bl * (1 - zl))

    model.addConstr((hl_minus_1[None] - h_pos) <= Ul_minus_1[None] * (1 - zl[:, None]))
    model.addConstr((hl_minus_1[None] - h_pos) >= Ll_minus_1[None] * (1 - zl[:, None]))

    model.addConstr(h_pos <= Ul_minus_1[None] * zl[:, None])
    model.addConstr(h_pos >= Ll_minus_1[None] * zl[:, None])

    return hl
