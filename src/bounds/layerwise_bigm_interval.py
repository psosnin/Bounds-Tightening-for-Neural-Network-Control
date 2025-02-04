import matplotlib.pyplot as plt
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from tqdm import trange

from ..utils.helpers import relu, getGurobiModel
from ..models.bigm import addBigMConstr


def getBounds(params, input_bounds, frac_interval):
    """
    Calculate the pre-activation bounds for the network with the given parameters by solving layerwise relaxed big-M
    models.
    Paramters:
        params: list of tuples (W, b) where W is the weight matrix and b is the bias vector for each layer
        input_bounds: tuple (L, U) where L is the lower bound and U is the upper bound on the input
        frac_interval: fraction of neurons to use interval (instead of big M) bounds

    """
    bounds = [input_bounds]
    assert len(params) >= 2
    W, b = params[0]
    # use interval bound for first layer
    L, U = input_bounds
    bounds.append((b + relu(W) @ L - relu(-W) @ U, b + relu(W) @ U - relu(-W) @ L))
    for l in range(1, len(params)):
        Wl, bl = params[l]
        Wl_minus_1, bl_minus_1 = params[l-1]
        post_act_bounds_l_minus_2 = (relu(b) for b in bounds[-2]) if l != 1 else bounds[-2]  # layer 0 can be negative
        # handle different fraction for each layer
        fi = frac_interval if isinstance(frac_interval, (int, float)) else frac_interval[l-1]
        bounds.append(_boundsHelper(Wl, bl, Wl_minus_1, bl_minus_1, bounds[-1], post_act_bounds_l_minus_2, fi))
    return bounds


def _boundsHelper(Wl, bl, Wl_minus_1, bl_minus_1, pre_bounds_l_minus_1, post_bounds_l_minus_2, frac_interval):
    """
    Compute bounds on Wl @ hl_minus_1 + bl for a single layer using big-M to solve the following problems:

        min/maximise y[i]
        subject to: y == Wl @ hl_minus_1 + bl
                    hl_minus_1 = ReLU(Wl_minus_1 @ hl_minus_2 + bl_minus_1)
                    hl_minus_2 in [Ll_minus_2, Ul_minus_2]

    where the big M coefficients are in MLl_minus_1 <= Wl_minus_1 @ hl_minus_2 + bl_minus_1 <= MUl_minus_1
    Parameters:
        Wl: weight matrix for the current layer
        bl: bias vector for the current layer
        Wl_minus_1: weight matrix for the previous layer
        bl_minus_1: bias vector for the previous layer
        pre_bounds_l_minus_1: tuple (MLl, MUl) such that MLl <= Wl @ hl_minus_1 + bl <= MUl
        post_bounds_l_minus_2: tuple (Ll_minus_2, Ul_minus_2) such that Ll_minus_2 <= hl_minus_2 <= Ul_minus_2
        frac_interval: fraction of neurons to use interval (instead of big M) bounds
    Returns:
        MLl, MUl such that MLl <= Wl @ hl_minus_1 + bl <= MUl
    """
    assert frac_interval >= 0 and frac_interval <= 1
    # choose random subset of neurons to use big M constraints
    bigm_idx = np.random.choice(bl_minus_1.size, int((1 - frac_interval) * bl_minus_1.size), replace=False)
    int_idx = np.delete(np.arange(bl_minus_1.size), bigm_idx)

    # split parameters according to indices
    MLl_minus_1_int = pre_bounds_l_minus_1[0][int_idx]
    MUl_minus_1_int = pre_bounds_l_minus_1[1][int_idx]
    pre_bounds_l_minus_1_bigm = [b[bigm_idx] for b in pre_bounds_l_minus_1]

    # interval bounds for non-big M neurons
    MLl = relu(Wl[:, int_idx]) @ relu(MLl_minus_1_int) - relu(- Wl[:, int_idx]) @ relu(MUl_minus_1_int)
    MUl = relu(Wl[:, int_idx]) @ relu(MUl_minus_1_int) - relu(- Wl[:, int_idx]) @ relu(MLl_minus_1_int)

    # if interval vars only, return
    if bigm_idx.size == 0:
        return MLl + bl, MUl + bl

    # set up model
    m = getGurobiModel()
    hl_minus_2 = m.addMVar(Wl_minus_1.shape[1], *post_bounds_l_minus_2)

    hl_minus_1 = addBigMConstr(
        m, hl_minus_2, Wl_minus_1[bigm_idx], bl_minus_1[bigm_idx], pre_bounds_l_minus_1_bigm, bigm_idx.size
    )
    y = m.addMVar(bl.size, lb=-np.inf)
    m.addConstr(y == Wl[:, bigm_idx] @ hl_minus_1 + bl)
    m.update()

    # solve model
    for i in range(bl.size):
        m.setObjective(y[i], GRB.MINIMIZE), m.reset(), m.optimize()
        MLl[i] += y.x[i]
        m.setObjective(y[i], GRB.MAXIMIZE), m.reset(), m.optimize()
        MUl[i] += y.x[i]
    return MLl, MUl
