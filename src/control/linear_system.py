import numpy as np
import matplotlib.pyplot as plt

import gurobipy as gp
from gurobipy import GRB

from ..utils.helpers import getGurobiModel, relu
from ..models.bigm import addBigMToModel


def simulateLTI(system, x0, N=50, controller=None, plot=True, labels=None):
    """
    Simulate and plot the LTI system with the given dynamics and controller.
    """
    A, B, x_bounds = system['A'], system['B'], system['x_bounds']
    state_dim, control_dim = A.shape[1], B.shape[1]

    x = np.zeros((N+1, state_dim))  # array to store results
    x[0] = x0
    u = np.zeros((N, control_dim))  # array to store control inputs

    for t in range(N):
        u[t] = 0 if controller is None else controller(x[t])
        x[t+1] = A @ x[t] + B @ u[t]

    # plot results
    n_plots = state_dim if controller is None else state_dim + control_dim

    if not plot:
        return x, u

    f, axs = plt.subplots(ncols=n_plots, figsize=(3 * n_plots, 2.5))

    for i in range(state_dim):
        axs[i].plot(x[:, i])
        axs[i].set_ylim(x_bounds[0][i], x_bounds[1][i])
        if labels:
            axs[i].set_title(labels[i])

    if controller is not None:
        for i in range(control_dim):
            axs[state_dim + i].plot(u[:, i])
            axs[state_dim + i].set_title("Control input")

    plt.show()
    return x, u


def getControllerFromGurobi(m, in_var, out_var):
    """
    Given a gurobi model that solves a control problem from a given input to a given output, return a
    function that can be called on any given state to compute the controller output.
    """
    def controller(x0):
        try:
            intial = m.addConstr(in_var == x0)
            m.optimize()
            m.remove(intial)
            return out_var[0].x
        except gp.GurobiError as e:
            # this usually happens due to overflow in x when the system blows up
            return 0
    return controller


def getBoundsUnderNNControl(A, B, params, input_bounds, pre_activation_bounds, relax=True):
    """
    Given a linear system x = A @ x + B @ u where u is a control input computed by a neural network with
    the given parameters, compute bounds on the new state by solving the following problems:

        min/maximise x1[i]
        subject to: x1 == A @ x0 + B @ u
                    u == WL @ hL_minus_1 + bL
                    hL_minus_1 = ReLU(WL_minus_1 @ hL_minus_2 + bL_minus_1) (bigM constraints)
                    x0 in [-x_bounds, x_bounds]

    Parameters:
        A: system dynamics matrix
        B: input dynamics matrix
        params: list of tuples (Wi, bi) for the i = 1, ..., L layers in the network
        input_bounds: bounds on the initial state st input_bounds[0] <= x0 <= input_bounds[1]
        pre_activation_bounds: list of tuples (MLi, MUi) such that MLi <= Wi @ hi-1 + bi <= MUi
        relax: whether to use the lp relaxation in the big-M constraints
    Returns:
        x1_lbounds, x1_ubounds such that x1_lbounds <= A @ x0 + B @ u <= x1_ubounds
    """
    m = getGurobiModel()

    n_x = input_bounds[0].size

    # set up parameters
    assert len(params) >= 2
    input_dim = params[0][0].shape[1]
    output_dim = params[-1][0].shape[0]
    x0 = m.addMVar(input_dim, lb=input_bounds[0], ub=input_bounds[1])
    u = m.addMVar(output_dim, lb=-np.inf, ub=np.inf)
    addBigMToModel(m, x0, u, params, pre_activation_bounds, relax=relax)

    # second state variable
    x1 = m.addMVar(input_dim, lb=-np.inf)
    # add constraints
    m.addConstr(x1 == A @ x0 + B @ u)

    x1_lbounds, x1_ubounds = -np.inf * np.ones(n_x), np.inf * np.ones(n_x)
    for i in range(n_x):
        m.setObjective(x1[i], GRB.MINIMIZE), m.reset(), m.optimize()
        x1_lbounds[i] = x1.x[i]
        m.setObjective(x1[i], GRB.MAXIMIZE), m.reset(), m.optimize()
        x1_ubounds[i] = x1.x[i]

    return x1_lbounds, x1_ubounds
