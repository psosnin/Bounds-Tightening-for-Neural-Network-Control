import numpy as np
from gurobipy import GRB

from ..utils.helpers import relu, getGurobiModel
from ..models.partitioned import addPartitionedConstr, uniformSplit


def getBounds(params, input_bounds, relax=False, split_fn=uniformSplit, P=2):
    """
    Calculate the pre-activation bounds for the network with the given parameters by solving the relaxed LP for
    the whole network.
    Paramters:
        params: list of tuples (W, b) where W is the weight matrix and b is the bias vector for each layer
        input_bounds: tuple (L, U) where L is the lower bound and U is the upper bound on the input
        relax: whether to use the linear relaxation of the network constraints
        split_fn: function to compute partitions with
        P: number of partitions to use
    """
    m = getGurobiModel()

    # add input variable
    h_list = [m.addMVar(input_bounds[0].size, lb=input_bounds[0], ub=input_bounds[1])]
    pre_relu_bounds = [input_bounds]

    for l in range(len(params)):
        Wl, bl = params[l]
        hl_minus_1 = h_list[-1]
        ML, MU = tighten(m, Wl @ hl_minus_1 + bl)
        N_binary = 0 if relax else bl.size
        post_relu_bounds = (relu(b) for b in pre_relu_bounds[-1]) if l >= 1 else input_bounds
        hl = addPartitionedConstr(m, hl_minus_1, Wl, bl, (ML, MU), post_relu_bounds, split_fn, P, N_binary)
        h_list.append(hl)
        pre_relu_bounds.append((ML, MU))

    return pre_relu_bounds


def getMultiStepBounds(params, input_bounds, A, B, K, relax=False, split_fn=uniformSplit, P=2):
    """
    Calculate the pre-activation bounds for the network with the given parameters over multiple time-steps of the linear
    system

        x = A x + B f_nn(x)

    by solving the MIP/LP with the partitioned formulation for the whole system.
    Parameters:
        params: list of tuples (W, b) where W is the weight matrix and b is the bias vector for each layer
        input_bounds: tuple (L, U) where L is the lower bound and U is the upper bound on the input
        A: system state dynamics
        B: system control dynamics
        K: number of time-steps
        relax: whether to use the linear relaxation of the network constraints
        split_fn: function to compute partitions with
        P: number of partitions to use
    Returns:
        bounds: list of list of bounds such that bounds[k][l] is the bounds on the l-th hidden variable at timestep k
    """

    m = getGurobiModel()
    state_dim = A.shape[1]
    L = len(params)

    # hidden variables for each time step
    hlk = [[m.addMVar(state_dim, lb=input_bounds[0], ub=input_bounds[1])]]

    # container to hold bounds, pre_relu_bounds[k][l] = (L, U) for the bounds of the l-th hidden variable at time-step k
    pre_relu_bounds = [[input_bounds]]

    # for each time step
    for k in range(K):
        for l in range(L):
            Wl, bl = params[l]
            hl_minus_1 = hlk[k][-1]
            ML, MU = tighten(m, Wl @ hl_minus_1 + bl)
            if l < L - 1:  # hidden layers have relu
                N_binary = 0 if relax else bl.size
                post_relu_bounds = (relu(b) for b in pre_relu_bounds[k][-1]) if l >= 1 else pre_relu_bounds[k][-1]
                hl = addPartitionedConstr(m, hl_minus_1, Wl, bl, (ML, MU), post_relu_bounds, split_fn, P, N_binary)
            elif l == L - 1:  # last layer has no relu
                hl = m.addMVar(ML.size, lb=ML, ub=MU)
                m.addConstr(Wl @ hl_minus_1 + bl == hl)
            hlk[k].append(hl)
            pre_relu_bounds[k].append((ML, MU))

        # compute bounds on next state variable
        ML, MU = tighten(m, A @ hlk[k][0] + B @ hlk[k][-1])

        # initialise the next state variable and connect dynamics
        xk = m.addMVar(state_dim, lb=ML, ub=MU)
        m.addConstr(xk == A @ hlk[k][0] + B @ hlk[k][-1])
        hlk.append([xk])
        pre_relu_bounds.append([(ML, MU)])

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
