import sys
import time
import argparse
from copy import copy

import wandb
import numpy as np

sys.path.append("..")  # nopep8
from src import control
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
    job_type=SINGLE_STEP_BOUNDS_JOB_TYPE,
    name=SINGLE_STEP_BOUNDS_JOB_NAME,
    group=GROUP.format(seed=seed),
    config={
        "seed": seed,
        "system": SYSTEM_NAME,
        "network": network_type,
        "shape": str(shape),
    }
)
config = wandb.config

"""
Load neural network and linear system
"""

model_artifact = run.use_artifact(
    f"{ENTITY}/{PROJECT}/{NETWORK_ARTIFACT_NAME.format(network_type=network_type, seed=seed)}:latest"
)
model_dir = model_artifact.download()
params = np.load(f"{model_dir}/{PARAMS_FILENAME}", allow_pickle=True)
system_name = model_artifact.metadata["system"]
system = control.example_systems.getSystem(system_name)
x_in_bounds = system["x_in_bounds"]

table = wandb.Table(
    columns=["bound_name", "bound_time", "mean_final_layer_width", "max_final_layer_width"],
)

for bound_name, bound_fn in SINGLE_STEP_BOUNDS_METHODS.items():
    start = time.time()
    bound = bound_fn(params, x_in_bounds)

    # calculate metrics on bounds
    total_time = time.time() - start
    mean_final_layer_width = np.mean(bound[-1][1] - bound[-1][0])
    max_final_layer_width = np.max(bound[-1][1] - bound[-1][0])

    table.add_data(bound_name, total_time, mean_final_layer_width, max_final_layer_width)

    artifact = wandb.Artifact(
        name=BOUND_ARTIFACT_NAME.format(bound_name=bound_name, network_type=network_type, seed=seed),
        type=BOUND_ARTIFACT_TYPE,
        metadata=dict(config)
    )

    with artifact.new_file(BOUND_FILENAME, mode="wb") as file:
        np.save(file, np.array(bound, dtype=object))

    run.log_artifact(artifact)
    run.log({"single_bounds_table": copy(table)})
