"""
This module contains functions to initialise and solve model predictive controllers using gurobi.
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from ..utils.helpers import getGurobiModel, relu


def getMPController(system, N_p=10, quiet=True):
    """
    Initialise an MPC controller for the linear system with dynamics x_{k+1} = A x_k + B u_k
    Parameters:
        system: dictionary of linear system parameters
        N_p: prediction horizon
        quiet: if True, suppress gurobi output
    Returns:
        controller: function that takes in a state and returns the MPC control decision
    """
    # Initialise a (quiet) gurobi model
    m = getGurobiModel(quiet)

    # system parameters
    A, B = system['A'], system['B']
    state_dim, control_dim = A.shape[1], B.shape[1]
    C_x, c_x = system['C_x'], system['c_x']
    C_u, c_u = system['C_u'], system['c_u']
    C_in, c_in = system['C_in'], system['c_in']
    R, P, Q = system['R'], system['P'], system['Q']
    x_bounds, u_bounds = system['x_bounds'], system['u_bounds']

    # declare decision variables
    x = m.addMVar((N_p + 1, state_dim), lb=x_bounds[0], ub=x_bounds[1])
    u = m.addMVar((N_p, control_dim), lb=u_bounds[0], ub=u_bounds[1])
    m.addConstrs(x[i] == A @ x[i - 1] + B @ u[i - 1] for i in range(1, N_p + 1))
    m.addConstrs(C_x @ x[i] <= c_x for i in range(1, N_p + 1))
    m.addConstrs(C_u @ u[i] <= c_u for i in range(N_p))
    m.addConstr(C_in @ x[0] <= c_in)

    # set mpc quadratic objective
    obj1 = sum(x[i] @ Q @ x[i] for i in range(N_p))
    obj2 = sum(u[i] @ R @ u[i] for i in range(N_p))
    obj3 = x[N_p] @ P @ x[N_p]  # terminal cost
    m.setObjective(obj1 + obj2 + obj3, GRB.MINIMIZE)

    def controller(x0):
        intial = m.addConstr(x[0] == x0)
        m.reset()
        m.optimize()
        m.remove(intial)
        return u[0].x

    return controller


def getKKTController(system, N_p=10, quiet=True):
    """
    Initialise an MPC controller where the KKT optimality conditions are solved as a feasibility problem.
    This controller is equivalent to the MPC controller.
    Parameters:
        system: dictionary of linear system parameters
        N_p: prediction horizon
        quiet: if True, suppress gurobi output
    Returns:
        controller: function that takes in a state and returns the MPC control decision
    """
    # Initialise gurobi model
    m = getGurobiModel(quiet)
    x_in_bounds, u_bounds = system['x_in_bounds'], system['u_bounds']
    x = m.addMVar(system['state_dim'], lb=x_in_bounds[0], ub=x_in_bounds[1])
    u = m.addMVar(system['control_dim'], lb=u_bounds[0], ub=u_bounds[1])

    addKKTtoModel(m, x, u, system, N_p)

    # objective
    m.setObjective(0, GRB.MINIMIZE)

    def controller(x0):
        intial = m.addConstr(x == x0)
        m.reset()
        m.optimize()
        m.remove(intial)
        return u.x

    return controller


def addKKTtoModel(m, in_var, out_var, system, N_p):
    """
    Add KKT conditions to the optimization model.

    Parameters:
        m: gurobi model
        in_var: state variable
        out_var: control variable
        system: system parameters
        N_p: MPC prediction horizon
    """

    # system parameters
    A, B = system['A'], system['B']
    state_dim, control_dim = A.shape[1], B.shape[1]
    C_x, c_x = system['C_x'], system['c_x']
    C_u, c_u = system['C_u'], system['c_u']
    C_in, c_in = system['C_in'], system['c_in']
    R, P, Q = system['R'], system['P'], system['Q']
    x_bounds, u_bounds = system['x_bounds'], system['u_bounds']

    # declare primal variables
    # note that the KKT conditions below are only valid if the polyhedral bounds are inside these box bounds
    x = m.addMVar((N_p + 1, state_dim), lb=x_bounds[0], ub=x_bounds[1])
    u = m.addMVar((N_p, control_dim), lb=u_bounds[0], ub=u_bounds[1])

    # declare dual variables
    lambda_x = m.addMVar((N_p + 1, state_dim), lb=-np.inf)  # lm for x = Ax + Bu
    mu_in = m.addMVar(C_in.shape[0], lb=-np.inf)  # lm for C_in x0 <= c_in
    mu_x = m.addMVar((N_p, C_x.shape[0]), lb=-np.inf)  # lm for C_x x <= c_x
    mu_u = m.addMVar((N_p, C_u.shape[0]), lb=-np.inf)  # lm for C_u u <= c_u

    # primal feasibility
    m.addConstr(C_in @ x[0] <= c_in)
    m.addConstrs(C_x @ x[i] <= c_x for i in range(1, N_p + 1))
    m.addConstrs(C_u @ u[i] <= c_u for i in range(N_p))
    m.addConstrs(x[i+1] == A @ x[i] + B @ u[i] for i in range(N_p))

    # dual feasibility
    m.addConstr(mu_in >= 0)
    m.addConstr(mu_x >= 0)
    m.addConstr(mu_u >= 0)

    # stationarity
    m.addConstr(2 * Q @ x[0] + mu_in.T @ C_in + lambda_x[0] - lambda_x[1].T @ A == 0)
    m.addConstrs(2 * Q @ x[i] + mu_x[i - 1].T @ C_x + lambda_x[i] - lambda_x[i + 1].T @ A == 0 for i in range(1, N_p))
    m.addConstr(2 * P @ x[N_p] + mu_x[N_p - 1].T @ C_x + lambda_x[N_p] == 0)
    m.addConstrs(2 * R @ u[i] + mu_u[i].T @ C_u - lambda_x[i + 1].T @ B == 0 for i in range(N_p))

    # complementarity
    m.addConstr(mu_in.T @ (C_in @ x[0] - c_in) == 0)
    m.addConstrs(mu_x[i].T @ (C_x @ x[i + 1] - c_x) == 0 for i in range(N_p))
    m.addConstrs(mu_u[i].T @ (C_u @ u[i] - c_u) == 0 for i in range(N_p))
    m.setParam('NonConvex', 2)

    # connect the input and output variables
    m.addConstr(in_var == x[0])
    m.addConstr(out_var == u[0])


def getFJController(system, N_p=10, quiet=True):
    """
    Initialise an MPC controller where the FJ optimality conditions are solved as a feasibility problem.
    This controller is equivalent to the MPC controller.
    Parameters:
        system: dictionary of linear system parameters
        N_p: prediction horizon
        quiet: if True, suppress gurobi output
    Returns:
        controller: function that takes in a state and returns the MPC control decision
    """
    # Initialise gurobi model
    m = getGurobiModel(quiet)
    x_in_bounds, u_bounds = system['x_in_bounds'], system['u_bounds']
    x = m.addMVar(system['state_dim'], lb=x_in_bounds[0], ub=x_in_bounds[1])
    u = m.addMVar(system['control_dim'], lb=u_bounds[0], ub=u_bounds[1])

    addFJtoModel(m, x, u, system, N_p)

    # objective
    m.setObjective(0, GRB.MINIMIZE)

    def controller(x0):
        intial = m.addConstr(x == x0)
        m.reset()
        m.optimize()
        m.remove(intial)
        return u.x

    return controller


def addFJtoModel(m, in_var, out_var, system, N_p):
    """
    Add FJ conditions to the optimization model. Using the FJ conditions instead of the KKT conditions allows us to 
    easily bound the dual variables and thus linearise the complementarity constraints using big-M.

    Parameters:
        m: gurobi model
        in_var: state variable
        out_var: control variable
        system: system parameters
        N_p: MPC prediction horizon
    """

    # system parameters
    A, B = system['A'], system['B']
    state_dim, control_dim = A.shape[1], B.shape[1]
    C_x, c_x = system['C_x'], system['c_x']
    C_u, c_u = system['C_u'], system['c_u']
    C_in, c_in = system['C_in'], system['c_in']
    R, P, Q = system['R'], system['P'], system['Q']
    u_bounds, x_bounds, x_in_bounds = system['u_bounds'], system['x_bounds'], system['x_in_bounds']

    # declare primal variables
    # note that the FJ conditions below are only valid if the polyhedral bounds are inside these box bounds
    x = m.addMVar((N_p + 1, state_dim), lb=x_bounds[0], ub=x_bounds[1])
    u = m.addMVar((N_p, control_dim), lb=u_bounds[0], ub=u_bounds[1])

    # declare dual variables, we can bound them by M because we use the FJ conditions
    M = 1
    lambda_x = m.addMVar((N_p + 1, state_dim), lb=-M, ub=M)  # lm for x = Ax + Bu
    mu_in = m.addMVar(C_in.shape[0], lb=0, ub=M)  # lm for C_in x0 <= c_in
    mu_x = m.addMVar((N_p, C_x.shape[0]), lb=0, ub=M)  # lm for C_x x <= c_x
    mu_u = m.addMVar((N_p, C_u.shape[0]), lb=0, ub=M)  # lm for C_u u <= c_u
    # the lower bound on l0 should be small but larger than the feasibility tolerance.
    # in theory, it should be 0 so if the lower bound is not small enough then this eliminate feasible solutions
    # in practice, setting it to 1e-3 works well but we should always check the returned solution
    l0 = m.addVar(lb=1e-3, ub=M)

    # primal feasibility
    m.addConstr(C_in @ x[0] <= c_in)
    m.addConstrs(C_x @ x[i] <= c_x for i in range(1, N_p + 1))
    m.addConstrs(C_u @ u[i] <= c_u for i in range(N_p))
    m.addConstrs(x[i+1] == A @ x[i] + B @ u[i] for i in range(N_p))

    # stationarity
    m.addConstr(2 * Q @ x[0] * l0 + mu_in.T @ C_in + lambda_x[0] - lambda_x[1].T @ A == 0)
    m.addConstrs(2 * Q @ x[i] * l0 + mu_x[i-1].T @ C_x + lambda_x[i] - lambda_x[i+1].T @ A == 0 for i in range(1, N_p))
    m.addConstr(2 * P @ x[N_p] * l0 + mu_x[N_p - 1].T @ C_x + lambda_x[N_p] == 0)
    m.addConstrs(2 * R @ u[i] * l0 + mu_u[i].T @ C_u - lambda_x[i + 1].T @ B == 0 for i in range(N_p))

    # additional dual variable bounds
    m.addConstr(mu_in.sum() + mu_u.sum() <= M)
    m.addConstr(mu_x.sum() + mu_u.sum() <= M)
    m.addConstr(mu_in.sum() + mu_x.sum() <= M)
    m.addConstr(mu_u.sum() <= M)
    m.addConstr(mu_x.sum() <= M)
    m.addConstr(mu_in.sum() <= M)

    # complementarity
    M_u = c_u - (relu(C_u) @ u_bounds[0] - relu(-C_u) @ u_bounds[1])
    M_x = c_x - (relu(C_x) @ x_bounds[0] - relu(-C_x) @ x_bounds[1])
    M_in = c_in - (relu(C_in) @ x_in_bounds[0] - relu(-C_in) @ x_in_bounds[1])
    beta_u = m.addMVar((N_p, C_u.shape[0]), vtype=GRB.BINARY)
    beta_x = m.addMVar((N_p, C_x.shape[0]), vtype=GRB.BINARY)
    beta_in = m.addMVar(C_in.shape[0], vtype=GRB.BINARY)
    m.addConstr(mu_u <= M * beta_u)
    m.addConstr(mu_x <= M * beta_x)
    m.addConstr(mu_in <= M * beta_in)
    m.addConstrs(c_u - C_u @ u[i] <= M_u * (1 - beta_u[i]) for i in range(N_p))
    m.addConstrs(c_x - C_x @ x[i+1] <= M_x * (1 - beta_x[i]) for i in range(N_p))
    m.addConstr(c_in - C_in @ x[0] <= M_in * (1 - beta_in))
    m.setParam('NonConvex', 2)
    m.setParam('PreQLinearize', 2)

    # connect the input and output variables
    m.addConstr(in_var == x[0])
    m.addConstr(out_var == u[0])


def getLinearKKTController(system, N_p=10, quiet=True, M=1e4):
    """
    Initialise an MPC controller where the KKT optimality conditions are solved as a feasibility problem.
    This controller is equivalent to the MPC controller.
    Parameters:
        system: dictionary of linear system parameters
        N_p: prediction horizon
        quiet: if True, suppress gurobi output
    Returns:
        controller: function that takes in a state and returns the MPC control decision
    """
    # Initialise gurobi model
    m = getGurobiModel(quiet)
    x_in_bounds, u_bounds = system['x_in_bounds'], system['u_bounds']
    x = m.addMVar(system['state_dim'], lb=x_in_bounds[0], ub=x_in_bounds[1])
    u = m.addMVar(system['control_dim'], lb=u_bounds[0], ub=u_bounds[1])

    addLinearKKTtoModel(m, x, u, system, N_p, M)

    # objective
    m.setObjective(0, GRB.MINIMIZE)

    def controller(x0):
        intial = m.addConstr(x == x0)
        m.reset()
        m.optimize()
        m.remove(intial)
        return u.x

    return controller


def addLinearKKTtoModel(m, in_var, out_var, system, N_p, M):
    """
    Add KKT conditions to the optimization model.

    Parameters:
        m: gurobi model
        in_var: state variable
        out_var: control variable
        system: system parameters
        N_p: MPC prediction horizon
    """

    # system parameters
    A, B = system['A'], system['B']
    state_dim, control_dim = A.shape[1], B.shape[1]
    C_x, c_x = system['C_x'], system['c_x']
    C_u, c_u = system['C_u'], system['c_u']
    C_in, c_in = system['C_in'], system['c_in']
    R, P, Q = system['R'], system['P'], system['Q']
    x_bounds, u_bounds = system['x_bounds'], system['u_bounds']
    x_in_bounds = system['x_in_bounds']

    # declare primal variables
    # note that the KKT conditions below are only valid if the polyhedral bounds are inside these box bounds
    x = m.addMVar((N_p + 1, state_dim), lb=x_bounds[0], ub=x_bounds[1])
    u = m.addMVar((N_p, control_dim), lb=u_bounds[0], ub=u_bounds[1])

    # declare dual variables
    lambda_x = m.addMVar((N_p + 1, state_dim), lb=-np.inf)  # lm for x = Ax + Bu
    mu_in = m.addMVar(C_in.shape[0], lb=-np.inf)  # lm for C_in x0 <= c_in
    mu_x = m.addMVar((N_p, C_x.shape[0]), lb=-np.inf)  # lm for C_x x <= c_x
    mu_u = m.addMVar((N_p, C_u.shape[0]), lb=-np.inf)  # lm for C_u u <= c_u

    # primal feasibility
    m.addConstr(C_in @ x[0] <= c_in)
    m.addConstrs(C_x @ x[i] <= c_x for i in range(1, N_p + 1))
    m.addConstrs(C_u @ u[i] <= c_u for i in range(N_p))
    m.addConstrs(x[i+1] == A @ x[i] + B @ u[i] for i in range(N_p))

    # dual feasibility
    m.addConstr(mu_in >= 0)
    m.addConstr(mu_x >= 0)
    m.addConstr(mu_u >= 0)

    # stationarity
    m.addConstr(2 * Q @ x[0] + mu_in.T @ C_in + lambda_x[0] - lambda_x[1].T @ A == 0)
    m.addConstrs(2 * Q @ x[i] + mu_x[i - 1].T @ C_x + lambda_x[i] - lambda_x[i + 1].T @ A == 0 for i in range(1, N_p))
    m.addConstr(2 * P @ x[N_p] + mu_x[N_p - 1].T @ C_x + lambda_x[N_p] == 0)
    m.addConstrs(2 * R @ u[i] + mu_u[i].T @ C_u - lambda_x[i + 1].T @ B == 0 for i in range(N_p))

    # complementarity
    M_u = c_u - (relu(C_u) @ u_bounds[0] - relu(-C_u) @ u_bounds[1])
    M_x = c_x - (relu(C_x) @ x_bounds[0] - relu(-C_x) @ x_bounds[1])
    M_in = c_in - (relu(C_in) @ x_in_bounds[0] - relu(-C_in) @ x_in_bounds[1])
    beta_u = m.addMVar((N_p, C_u.shape[0]), vtype=GRB.BINARY)
    beta_x = m.addMVar((N_p, C_x.shape[0]), vtype=GRB.BINARY)
    beta_in = m.addMVar(C_in.shape[0], vtype=GRB.BINARY)
    m.addConstr(mu_u <= M * beta_u)
    m.addConstr(mu_x <= M * beta_x)
    m.addConstr(mu_in <= M * beta_in)
    m.addConstrs(c_u - C_u @ u[i] <= M_u * (1 - beta_u[i]) for i in range(N_p))
    m.addConstrs(c_x - C_x @ x[i+1] <= M_x * (1 - beta_x[i]) for i in range(N_p))
    m.addConstr(c_in - C_in @ x[0] <= M_in * (1 - beta_in))

    # connect the input and output variables
    m.addConstr(in_var == x[0])
    m.addConstr(out_var == u[0])
