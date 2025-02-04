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
    job_type=WORST_CASE_ERROR_JOB_TYPE,
    name=WORST_CASE_ERROR_JOB_NAME.format(network_type=network_type, seed=seed),
    group=GROUP.format(seed=seed),
    config={
        "seed": seed,
        "system": SYSTEM_NAME,
        "network": network_type,
        "shape": str(shape),
        "N_p": N_P,
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

"""
Set up results table
"""

table = wandb.Table(columns=[
    "bound", "solve_time", "status", "work", "explored_nodes",
    "iterations", "objective_val", "relaxation_gap", "x0", "u_nn", "u_mpc"
])

"""
Perform bound calculation runs
"""

for bound_name, bound_fn in SINGLE_STEP_BOUNDS_METHODS.items():
    print("Starting", bound_name, "worst case error")
    # full mip and rh mip are the same for a 1 layer network
    if "rh" in bound_name and network_type == 2:
        continue
    # get bound artifact
    bound_artifact = run.use_artifact(
        f"{ENTITY}/{PROJECT}/{BOUND_ARTIFACT_NAME.format(bound_name=bound_name, network_type=network_type, seed=seed)}:latest"
    )
    bound_dir = bound_artifact.download()
    bound = np.load(f"{bound_dir}/{BOUND_FILENAME}", allow_pickle=True)

    # set up maximise inaccuracy gurobi model
    model = utils.helpers.getGurobiModel(quiet=False)
    model.setParam("TimeLimit", WORST_CASE_ERROR_TIMEOUT)
    model.setParam("MIPFocus", 2)

    # add neural network
    x_nn = model.addMVar(state_dim, lb=x_in_bounds[0], ub=x_in_bounds[1])
    u_nn = model.addMVar(control_dim, lb=-np.inf, ub=np.inf)
    models.bigm.addBigMToModel(model, x_nn, u_nn, params, bound, False)

    # add mpc
    x_mpc = model.addMVar(state_dim, lb=x_in_bounds[0], ub=x_in_bounds[1])
    u_mpc = model.addMVar(control_dim, lb=u_bounds[0], ub=u_bounds[1])
    control.mpc.addFJtoModel(model, x_mpc, u_mpc, system, N_P)

    # add input constraints
    model.addConstr(C_in @ x_mpc <= c_in)
    model.addConstr(x_mpc == x_nn)

    # set objective
    inacc = model.addMVar(control_dim, lb=-np.inf, ub=np.inf)
    max_inacc = model.addVar(lb=0, ub=np.inf)
    model.addConstr(inacc == u_mpc - u_nn)
    model.addConstr(max_inacc == gp.norm(inacc, np.inf))
    model.setObjective(max_inacc, GRB.MAXIMIZE)

    # get linear relaxation
    model.update()
    relaxed_model = model.relax()
    relaxed_model.optimize()
    relaxed_objval = relaxed_model.objVal if relaxed_model.status == GRB.OPTIMAL else np.inf

    # compute mip solution
    start = time.time()
    model.optimize()
    objval = model.objVal if model.status == GRB.OPTIMAL else np.nan

    # check results of optimal solution, if something is wrong then return an error code
    if model.status == GRB.OPTIMAL:
        try:
            true_mpc_output = mpc_controller(x_mpc.x)
        except gp.GurobiError as e:
            true_mpc_output = np.zeros_like(u_mpc.x)
            wandb.alert(title=f"Worst Case Error {seed}", text="MPC controller failed to solve")
            print("Warning: mpc controller failed to solve")
            print("Gurobi error:", e)
        if not np.allclose(utils.network.forward(x_nn.x, params), u_nn.x):
            wandb.alert(title=f"Worst Case Error {seed}", text="Network output does not match gurobi solution")
            print("Warning: network output does not match gurobi solution")
            print("x0:", x_nn.x)
            print("gurobi network output:", u_nn.x)
            print("true network output:", utils.network.forward(x_nn.x, params))
        if not np.allclose(true_mpc_output, u_mpc.x, 1e-4, 1e-4):
            wandb.alert(title=f"Worst Case Error {seed}", text="MPC output does not match gurobi solution")
            print("Warning: mpc output does not match gurobi solution")
            print("x0:", x_mpc.x)
            print("gurobi mpc output:", u_mpc.x)
            print("true mpc output:", true_mpc_output)
        if not np.allclose(model.objVal, np.max(np.abs(u_nn.x - u_mpc.x))):
            wandb.alert(title=f"Worst Case Error {seed}", text="Objective value does not match gurobi solution")
            print("Warning: objective value does not match gurobi solution")
            print("gurobi objective value:", model.objVal)
            print("true objective value:", np.max(np.abs(u_nn.x - u_mpc.x)))

    x_nn = x_nn.x if model.status == GRB.OPTIMAL else np.nan
    u_nn = u_nn.x if model.status == GRB.OPTIMAL else np.nan
    u_mpc = u_mpc.x if model.status == GRB.OPTIMAL else np.nan

    table.add_data(
        bound_name, time.time() - start, model.status, model.Work, model.NodeCount,
        model.IterCount, objval, relaxed_objval - objval, str(x_nn), str(u_nn), str(u_mpc)
    )

    wandb.log({"worst_case_table": copy(table)})
