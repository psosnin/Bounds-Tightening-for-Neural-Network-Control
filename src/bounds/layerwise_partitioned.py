import numpy as np
from gurobipy import GRB

from ..utils.helpers import relu, getGurobiModel
from ..models.partitioned import addPartitionedConstr, uniformSplit


def getBounds(params, input_bounds, frac_binary, P=2, split_fn=uniformSplit):
    """
    Calculate the pre-activation bounds for the network with the given parameters by solving layerwise relaxed partitioned
    formulation.
    Paramters:
        params: list of tuples (W, b) where W is the weight matrix and b is the bias vector for each layer
        input_bounds: tuple (L, U) where L is the lower bound and U is the upper bound on the input
        frac_binary: fraction of binary vs continuous variables to use in the big-M model for each layer
        P: number of partitions to use
        split_fn: function to use for partitioning
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
        hl = addPartitionedConstr(m, hl_minus_1, W[l], b[l], bounds[-1], prev_bounds, split_fn, P, int(b[l].size * fb))
        ML, MU = tighten(m, W[l+1] @ hl + b[l+1])
        bounds.append((ML, MU))
    return bounds


def getMultiStepBounds(params, input_bounds, A, B, K, relax=False, P=2, split_fn=uniformSplit):
    """
    Calculate the pre-activation bounds for the network with the given parameters over multiple time-steps of the linear
    system

        x = A x + B f_nn(x)

    by solving layerwise MIP/LPs using the partitioned formulation.
    Parameters:
        params: list of tuples (W, b) where W is the weight matrix and b is the bias vector for each layer
        input_bounds: tuple (L, U) where L is the lower bound and U is the upper bound on the input
        A: system state dynamics
        B: system control dynamics
        K: number of time-steps
        relax: whether to use the linear relaxation of the network constraints
        P: number of partitions to use
        split_fn: function to use for partitioning
    Returns:
        bounds: list of list of bounds such that bounds[k][l] is the bounds on the l-th hidden variable at timestep k
    """
    fb = 0 if relax else 1
    WL, bL = params[-1]
    WL_minus_1, bL_minus_1 = params[-2]
    assert len(params) >= 2  # the code below only works for 1 or more hidden layers

    # container to hold bounds, pre_relu_bounds[k][l] = (L, U) for the bounds of the l-th hidden variable at time-step k
    pre_relu_bounds = []
    next_state_bounds = input_bounds

    for _ in range(K):
        # get layerwise bounds for the whole network for the k-th time step
        hidden_bounds = getBounds(params, next_state_bounds, fb, P=2, split_fn=uniformSplit)
        pre_relu_bounds.append(hidden_bounds)

        # to compute bounds on the linear system dynamics, consider the last relu in the network
        m = getGurobiModel()
        h0 = m.addMVar(hidden_bounds[0][0].size, lb=hidden_bounds[0][0], ub=hidden_bounds[0][1])
        prev_bounds = [relu(b) for b in hidden_bounds[-3]] if len(hidden_bounds) > 3 else hidden_bounds[-3]
        hL_minus_2 = m.addMVar(hidden_bounds[-3][0].size, lb=prev_bounds[0], ub=prev_bounds[1])
        hL_minus_1 = addPartitionedConstr(
            m, hL_minus_2, WL_minus_1, bL_minus_1, hidden_bounds[-2],
            prev_bounds, split_fn, P, int(fb * bL_minus_1.size)
        )
        hL = m.addMVar(hidden_bounds[-1][0].size, lb=hidden_bounds[-1][0], ub=hidden_bounds[-1][1])
        m.addConstr(hL == WL @ hL_minus_1 + bL)
        next_state_bounds = tighten(m, A @ h0 + B @ hL)

    pre_relu_bounds.append([next_state_bounds])
    return pre_relu_bounds


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
