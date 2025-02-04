import sys
import argparse

import wandb
import numpy as np

sys.path.append("..")  # nopep8
from src import control
from configs import *

"""
Set up configuration and wandb run
"""
parser = argparse.ArgumentParser()
parser.add_argument("--seed", "-s", type=int)
args = parser.parse_args()

seed = args.seed
np.random.seed(seed)

run = wandb.init(
    project=PROJECT,
    entity=ENTITY,
    job_type=MPC_JOB_TYPE,
    group=GROUP.format(seed=seed),
    name=MPC_JOB_NAME.format(seed=seed),
    mode=MODE,
    config={
        "seed": seed,
        "system": SYSTEM_NAME,  # system to use
        "N_p": N_P,  # MPC prediction horizon
        "N_trajectories": N_TRAJECTORIES,  # number of trajectories to simulate
        "N_steps": N_STEPS,  # number of steps in each trajectory simulation,
    }
)
config = wandb.config

"""
Set up linear system and controller
"""

system = control.example_systems.getSystem(SYSTEM_NAME)  # linear system parameters
state_dim, control_dim = system["state_dim"], system["control_dim"]  # problem dimensions
A, B = system["A"], system["B"]  # system dynamics
C_in, c_in, x_in_bounds = system["C_in"], system["c_in"], system["x_in_bounds"]  # input domain
controller = control.mpc.getMPController(system, config.N_p)  # MPC controller

"""
Generate training data
"""

x_train = np.zeros((config.N_trajectories, config.N_steps, state_dim))
u_train = np.zeros((config.N_trajectories, config.N_steps, control_dim))
for i in range(config.N_trajectories):
    print(i, end="\r")
    # generate initial sample using rejection sampling
    x_sim = np.random.uniform(low=x_in_bounds[0], high=x_in_bounds[1])
    while not np.all(C_in @ x_sim <= c_in):
        x_sim = np.random.uniform(low=x_in_bounds[0], high=x_in_bounds[1])
    for j in range(config.N_steps):
        # compute control step
        u_sim = controller(x_sim)
        x_train[i, j] = x_sim
        u_train[i, j] = u_sim
        # apply control step
        x_sim = A @ x_sim + B @ u_sim

x_train = x_train.reshape(-1, x_train.shape[-1])
u_train = u_train.reshape(-1, u_train.shape[-1])

"""
Save training data and log artifacts
"""

artifact = wandb.Artifact(
    name=DATASET_ARTIFACT_NAME.format(seed=seed),
    type=DATASET_ARTIFACT_TYPE,
    metadata=dict(config)
)

with artifact.new_file(X_FILENAME, mode="wb") as file:
    np.save(file, x_train)

with artifact.new_file(U_FILENAME, mode="wb") as file:
    np.save(file, u_train)

run.log_artifact(artifact)
