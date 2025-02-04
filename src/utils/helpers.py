import os
import numpy as np
import gurobipy as gp
from gurobipy import GRB


def getGurobiModel(quiet=True):
    """
    Initialise a gurobi model.
    """
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model(env=env) if quiet else gp.Model()
    mip_test = os.environ.get("MIPTest", False)
    if mip_test:
        m.setParam("MIPGap", 1e-9)
        m.setParam("OptimalityTol", 1e-9)
        m.setParam("FeasibilityTol", 1e-9)
    return m


def relu(h):
    return np.maximum(0, h)


def getLogGeometricMean(pre_activation_bounds):
    diffs = [U - L for L, U in pre_activation_bounds]
    return [np.log(diff).sum() / diff.size for diff in diffs]
