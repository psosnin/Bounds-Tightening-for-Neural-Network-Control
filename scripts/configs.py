import sys

sys.path.append("..")  # nopep8
from src.models.partitioned import posNegSplit
from src import bounds

PROJECT = "bound-tightening"
ENTITY = "ps1623"
GROUP = "group_{seed}"
MODE = "online"

MPC_JOB_NAME = "generate_mpc_data"
MPC_JOB_TYPE = "generate_mpc_data"

TRAINING_JOB_NAME = "train_neural_network"
TRAINING_JOB_TYPE = "train_neural_network"

SINGLE_STEP_BOUNDS_JOB_NAME = "single_step_bounds"
SINGLE_STEP_BOUNDS_JOB_TYPE = "single_step_bounds"

WORST_CASE_ERROR_JOB_NAME = "worst_case_error"
WORST_CASE_ERROR_JOB_TYPE = "worst_case_error"
WORST_CASE_ERROR_TIMEOUT = 3600

MULTI_STEP_BOUNDS_JOB_NAME = "multi_step_bounds"
MULTI_STEP_BOUNDS_JOB_TYPE = "multi_step_bounds"

REACHABILITY_JOB_NAME = "reachability"
REACHABILITY_JOB_TYPE = "reachability_analysis"
REACHABILITY_TIMEOUT = 3600

DATASET_ARTIFACT_TYPE = "mpc_dataset"
DATASET_ARTIFACT_NAME = "mpc_dataset_{seed}"
X_FILENAME = "x_train.npy"
U_FILENAME = "u_train.npy"

NETWORK_ARTIFACT_TYPE = "trained_network"
NETWORK_ARTIFACT_NAME = "network_{seed}"
PARAMS_FILENAME = "params.npy"
MODEL_FILENAME = "model.ckpt"

BOUND_ARTIFACT_TYPE = "single_bounds"
BOUND_ARTIFACT_NAME = "single_{bound_name}_{seed}"
BOUND_FILENAME = "bounds.npy"

MULTI_STEP_BOUND_ARTIFACT_TYPE = "multi_bounds"
MULTI_STEP_BOUND_ARTIFACT_NAME = "multi_{bound_name}_{seed}"
MULTI_STEP_BOUND_FILENAME = "bounds.npy"

SYSTEM_NAME = "two_dim_control"
N_P = 9  # mpc prediction horizon
N_TRAJECTORIES = 3000  # number of trajectories to simulate
N_STEPS = 10  # number of steps in each trajectory simulation

SHAPES = [
    [10, 10, 10, 10, 10, 10],
    [120],
    [60],
    [5, 5]  # for quick testing
]  # don't modify these, just add new shapes to this dict
NUM_EPOCHS = 5  # number of epochs in neural network training
BATCHSIZE = 100  # batch size in neural network training
LEARNING_RATE = 0.01  # learning rate in neural network training

SINGLE_STEP_BOUNDS_METHODS = {
    "interval": bounds.interval.getBounds,
    "crown": bounds.crown.getBounds,
    "alpha-crown": bounds.alpha_crown.getBounds,
    "full_lp_bigm": lambda p, b: bounds.full_bigm.getBounds(p, b, relax=True),
    "full_lp_pttn_p2": lambda p, b: bounds.full_partitioned.getBounds(p, b, relax=True, split_fn=posNegSplit, P=2),
    "full_lp_pttn_p4": lambda p, b: bounds.full_partitioned.getBounds(p, b, relax=True, split_fn=posNegSplit, P=4),
    "rh_mip_bigm": lambda p, b: bounds.layerwise_bigm.getBounds(p, b, 1),
    "full_mip_bigm": lambda p, b: bounds.full_bigm.getBounds(p, b, relax=False),
}

K_STEPS = 4  # number of steps for the reachability analysis

MULTI_STEP_BOUNDS_METHODS = {
    "interval": lambda p, b, A, B, K: bounds.interval.getMultiStepBounds(p, b, A, B, K),
    "crown": lambda p, b, A, B, K: bounds.alpha_crown.getMultiStepBounds(p, b, A, B, K, method="crown"),
    "alpha-crown": lambda p, b, A, B, K: bounds.alpha_crown.getMultiStepBounds(p, b, A, B, K),
    "full_lp_bigm": lambda p, b, A, B, K: bounds.full_bigm.getMultiStepBounds(p, b, A, B, K, relax=True),
    "full_lp_pttn_p2": lambda p, b, A, B, K: bounds.full_partitioned.getMultiStepBounds(p, b, A, B, K, relax=True, P=2),
    "full_lp_pttn_p4": lambda p, b, A, B, K: bounds.full_partitioned.getMultiStepBounds(p, b, A, B, K, relax=True, P=4),
    "rh_mip_bigm": lambda p, b, A, B, K: bounds.layerwise_bigm.getMultiStepBounds(p, b, A, B, K, relax=True),
    "full_mip_bigm": lambda p, b, A, B, K: bounds.full_bigm.getMultiStepBounds(p, b, A, B, K, relax=False),
}
