from math import floor, ceil

import matplotlib.pyplot as plt
import numpy as np
from gurobipy import GRB

from ..utils.helpers import relu


def addPartitionedToModel(m, in_var, out_var, params, pre_act_bounds, split_fn, P, relax=False):
    """
    Add the partitioned formulation constraints to the gurobi model for a neural network with the given parameters.
    Parameters:
        m: gurobi model
        in_var: input variable to the network
        out_var: output variable from the network
        params: list of tuples (Wi, bi) for the i = 1, ..., L layers in the network
        pre_act_bounds: list of tuples (MLi, MUi) for i = 0, ..., L such that MLi <= Wi @ hl-1 + bi <= MUi
        split_fn: function to compute partitions with
        P: number of partitions to use
        relax: whether to use the linear relaxation of the network constraints
    """
    L = len(params)
    hl = [in_var]

    # add hidden layers
    for i in range(L-1):
        Wl, bl = params[i]
        N_binary = 0 if relax else bl.size
        post_act_bounds_l_minus_1 = [relu(bd) for bd in pre_act_bounds[i]] if i >= 1 else pre_act_bounds[i]
        hl.append(addPartitionedConstr(
            m, hl[-1], Wl, bl, pre_act_bounds[i+1], post_act_bounds_l_minus_1, split_fn, P, N_binary
        ))
    # add output layer (no bigM because no relu)
    (W, b) = params[-1]
    m.addConstr(out_var == W @ hl[-1] + b)


def addPartitionedConstr(m, hl_minus_1, Wl, bl, pre_act_bounds_l, post_act_bounds_l_minus_1, split_fn, P, N_binary):
    """
    Add the partitioned formulation constraints for hl = ReLU(Wl @ hl_minus_1 + bl).
    Parameters:
        m: gurobi model to add the constraints to
        hl_minus_1: previous layer variable
        Wl: weight matrix for the current layer
        bl: bias vector for the current layer
        pre_act_bounds_l: tuple (Ll, Ul) such that Ll <= Wl @ hl_minus_1 + bl <= Ul
        post_act_bounds_l_minus_1: tuple (Ll_minus_1, Ul_minus_1) such that Ll_minus_1 <= hl_minus_1 <= Ul_minus_1
        split_fn: function to compute partitions with
        P: number of partitions to use
        N_binary: number of binary (instead of continuous) variables to use in this layer
    """
    nl = bl.size
    MLl, MUl = pre_act_bounds_l
    assert N_binary <= nl and N_binary >= 0
    # calculate splits indices for each weight vector
    splits_l = [split_fn(Wl[i], P) for i in range(Wl.shape[0])]
    # can't have more splits than variables in previous layer
    P = min(P, Wl.shape[1])
    # compute the split bounds for each partitioned variable from the post-activation bounds of the previous layer
    p_bounds_l = _computeSplitBounds(Wl, splits_l, post_act_bounds_l_minus_1)
    # define next layer variable
    hl = m.addMVar(nl, lb=relu(MLl), ub=relu(MUl))
    # randomly allocate binary and continuous variables
    binary_indices = np.random.choice(nl, N_binary, replace=False)
    vtypes = np.array([GRB.CONTINUOUS] * nl)
    vtypes[binary_indices] = GRB.BINARY
    zl = m.addMVar(nl, ub=1, vtype=vtypes)

    # declare auxiliary split variables
    vl_pos = m.addMVar((nl, P), lb=-np.inf)

    # add bigM formulation constraints
    m.addConstr(hl >= Wl @ hl_minus_1 + bl)
    m.addConstr(hl <= Wl @ hl_minus_1 + bl - MLl * (1 - zl))
    m.addConstr(hl <= MUl * zl)
    m.addConstr(hl >= 0)
    m.addConstr(Wl @ hl_minus_1 + bl <= MUl)  # this is not encoded by the original bigM formulation when u < 0
    m.addConstr(Wl @ hl_minus_1 + bl >= MLl)

    # add partitioned formulation constraints
    m.addConstr(hl == vl_pos.sum(1) + bl * zl)
    m.addConstr(0 <= vl_pos.sum(1) + bl * zl)

    m.addConstrs(0 >= sum([Wl[i, splits_l[i][p]] @ hl_minus_1[splits_l[i][p]] - vl_pos[i, p]
                 for p in range(P)]) + bl[i] * (1 - zl[i]) for i in range(nl))
    m.addConstrs(p_bounds_l[0][p, i] * (1 - zl[i]) <= Wl[i, splits_l[i][p]] @ hl_minus_1[splits_l[i][p]] - vl_pos[i, p]
                 for i in range(nl) for p in range(P))
    m.addConstrs(p_bounds_l[1][p, i] * (1 - zl[i]) >= Wl[i, splits_l[i][p]] @ hl_minus_1[splits_l[i][p]] - vl_pos[i, p]
                 for i in range(nl) for p in range(P))

    m.addConstr(p_bounds_l[0] * zl[None] <= vl_pos.T)
    m.addConstr(p_bounds_l[1] * zl[None] >= vl_pos.T)

    return hl


def _computeSplitBounds(Wl, splits_l, post_act_bounds_l_minus_1):
    """
    Given the pre-activation bounds and the indices of each split, compute bounds on each partitioned variable
    using interval arithmetic.
    Parameters:
        Wl: weight matrix for the current layer
        splits_l: splits_l[i, p] is a list of indices in the p-th partition of Wl[i]
        post_act_bounds_l_minus_1: tuple (Ll_minus_1, Ul_minus_1) such that Ll_minus_1 <= hl_minus_1 <= Ul_minus_1
    Returns:
        p_lb: p_lb[p, i] <= Wl[i, splits_l[p]] @ hl_minus_1[splits_l[p]]
        p_ub: p_ub[p, i] >= Wl[i, splits_l[p]] @ hl_minus_1[splits_l[p]]
    """
    Ll_minus_1, Ul_minus_1 = post_act_bounds_l_minus_1
    p_lb = np.zeros((len(splits_l[0]), Wl.shape[0]))
    p_ub = np.zeros((len(splits_l[0]), Wl.shape[0]))

    for i in range(Wl.shape[0]):
        for p, split in enumerate(splits_l[i]):
            p_lb[p, i] = relu(Wl[i, split]) @ Ll_minus_1[split] - relu(-Wl[i, split]) @ Ul_minus_1[split]
            p_ub[p, i] = relu(Wl[i, split]) @ Ul_minus_1[split] - relu(-Wl[i, split]) @ Ll_minus_1[split]
    return p_lb, p_ub


"""
Partitioning strategies
"""


def uniformSplit(Wi, P):
    """
    Split the indices of a layer into P parts in order.
    """
    N = min(P, Wi.size)
    return np.array_split(np.arange(Wi.size), N)


def equalSizeSplit(Wi, P):
    """
    Split the indices of a layer into P parts based on the size of the weights.
    """
    N = min(P, Wi.size)
    return np.array_split(np.argsort(Wi), N)


def posNegSplit(Wi, P):
    """
    Split the indices of a layer into P parts based on the size of the weights, where each split contains only positive or negative weights.
    """
    N = min(Wi.size, P)
    if N == 1:
        return [np.arange(Wi.size)]

    # calculate number of weights per split if all splits are equal
    n_per_split = Wi.size / N
    n_pos_weights = Wi[Wi > 0].size
    n_neg_weights = Wi[Wi < 0].size

    # calculate number of splits for positive and negative weights
    n_pos_splits = n_pos_weights / n_per_split
    n_neg_splits = n_neg_weights / n_per_split

    # adjust number of splits to be integers
    if n_pos_splits < 1:  # need at least one split for each group
        n_pos_splits, n_neg_splits = ceil(n_pos_splits), floor(n_neg_splits)
    elif n_neg_splits < 1:
        n_pos_splits, n_neg_splits = floor(n_pos_splits), ceil(n_neg_splits)
    elif n_pos_splits % 1 > n_neg_splits:  # assign extra split to the larger group
        n_pos_splits, n_neg_splits = ceil(n_pos_splits), floor(n_neg_splits)
    else:
        n_pos_splits, n_neg_splits = floor(n_pos_splits), ceil(n_neg_splits)
    assert n_pos_splits + n_neg_splits == N
    # get indices for positive and negative weights splits
    sort_idx = np.argsort(Wi)
    neg_splits = np.array_split(sort_idx[:n_neg_weights], n_neg_splits) if n_neg_weights > 0 else []
    pos_splits = np.array_split(sort_idx[n_neg_weights:], n_pos_splits) if n_pos_weights > 0 else []
    return neg_splits + pos_splits
