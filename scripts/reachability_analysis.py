import sys
import time
import argparse
from copy import copy

import wandb
import numpy as np
import gurobipy as gp
from gurobipy import GRB

sys.path.append("..")  # nopep8
from src import control, models, utils
from configs import *

"""
Get command line arguments
"""

parser = argparse.ArgumentParser()
parser.add_argument("--seed", "-s", type=int)
parser.add_argument("--network", "-n", type=int)
args = parser.parse_args()

seed = args.seed
shape = SHAPES[args.network]
network_type = args.network

np.random.seed(seed)

"""
Set up weights and biases run
"""

run = wandb.init(
    project=PROJECT,
    entity=ENTITY,
    mode=MODE,
    job_type=REACHABILITY_JOB_TYPE,
    name=REACHABILITY_JOB_NAME.format(network_type=network_type, seed=seed),
    group=GROUP.format(seed=seed),
    config={
        "seed": seed,
        "system": SYSTEM_NAME,
        "network": network_type,
        "shape": str(shape),
        "N_p": N_P,
        "K_step": K_STEPS
    }
)
config = wandb.config

"""
Load neural network and linear system parameters
"""

model_artifact = run.use_artifact(
    f"{ENTITY}/{PROJECT}/{NETWORK_ARTIFACT_NAME.format(network_type=network_type, seed=seed)}:latest"
)
model_dir = model_artifact.download()
params = np.load(f"{model_dir}/{PARAMS_FILENAME}", allow_pickle=True)
system_name = model_artifact.metadata["system"]
system = control.example_systems.getSystem(system_name)
mpc_controller = control.mpc.getMPController(system, N_P)
state_dim, control_dim = system["state_dim"], system["control_dim"]
x_bounds, u_bounds = system['x_bounds'], system['u_bounds']
C_in, c_in, x_in_bounds = system['C_in'], system['c_in'], system['x_in_bounds']
A, B = system['A'], system['B']

"""
Set up results table
"""

table = wandb.Table(columns=[
    "bound", "solve_time", "status", "work", "explored_nodes",
    "iterations", "objective_val", "relaxation_val", "coordinate"
])


"""
Perform reachability analysis
"""

# generate random hyperplane to tighten over
c = np.random.uniform(-1, 1, state_dim)

for bound_name, bound_fn in MULTI_STEP_BOUNDS_METHODS.items():
    # get bound artifact
    bound_artifact = run.use_artifact(
        f"{ENTITY}/{PROJECT}/{MULTI_STEP_BOUND_ARTIFACT_NAME.format(bound_name=bound_name, seed=seed)}:latest"
    )
    bound_dir = bound_artifact.download()
    multibounds = np.load(f"{bound_dir}/{MULTI_STEP_BOUND_FILENAME}", allow_pickle=True)

    # set up reachability analysis gurobi model
    model = utils.helpers.getGurobiModel(quiet=False)
    model.setParam("TimeLimit", REACHABILITY_TIMEOUT)
    model.setParam("MIPFocus", 2)

    x = []  # state variables
    u = []  # control variables

    # initialise first state variable and add to list
    x0 = model.addMVar(state_dim, lb=x_in_bounds[0], ub=x_in_bounds[1])
    x.append(x0)

    for k in range(K_STEPS):
        # get bounds
        k_step_bounds = multibounds[k]
        # set up control variable
        uk = model.addMVar(control_dim, lb=k_step_bounds[-1][0], ub=k_step_bounds[-1][1])
        # add neural network to the gurobi model
        models.bigm.addBigMToModel(model, x[-1], uk, params, k_step_bounds, False)
        # add next state variable
        next_step_bounds = multibounds[k + 1]
        xk = model.addMVar(state_dim, lb=next_step_bounds[0][0], ub=next_step_bounds[0][1])
        # add linear dynamics
        model.addConstr(xk == A @ x[-1] + B @ uk)
        # add to list
        x.append(xk)
        u.append(uk)

    print("Starting reachability analysis for", bound_name)
    model.setObjective(c.T @ x[-1], GRB.MAXIMIZE)
    # get linear relaxation
    model.update(), model.reset()
    relaxed_model = model.relax()
    relaxed_model.optimize()
    relaxed_objval = relaxed_model.objVal if relaxed_model.status == GRB.OPTIMAL else np.inf
    # compute mip solution
    start = time.time()
    model.optimize()
    objval = model.objVal if model.status == GRB.OPTIMAL else np.nan
    table.add_data(
        bound_name, time.time() - start, model.status, model.Work, model.NodeCount,
        model.IterCount, objval, relaxed_objval, str(c)
    )
    wandb.log({"reachability_table": copy(table)})
