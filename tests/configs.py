import sys

sys.path.append('..')  # nopep8
from src import models, control
from src.bounds import layerwise_bigm, layerwise_bigm_interval, layerwise_extended, layerwise_partitioned
from src.bounds import full_bigm, full_partitioned, interval, alpha_crown, fastlin, crown

"""
This file contains configurations for the tests
"""

# network shapes to test
SHAPES = [
    [1, 1, 1],
    [1, 3, 1],
    [3, 3, 3],
    [6, 5, 4, 3, 2],
    [2, 3, 4, 5, 6],
    [2, 4, 4, 3],
]

# bounding methods to test
BOUNDING_METHODS = {
    "interval": interval.getBounds,
    "full lp bigm": lambda p, b: full_bigm.getBounds(p, b, relax=True),
    "layerwise bigm + interval (1.0)": lambda p, b: layerwise_bigm_interval.getBounds(p, b, 1),
    "full lp partitioned (P = 1)": lambda p, b: full_partitioned.getBounds(p, b, relax=True, P=1),
    "full lp partitioned (P = 2)": lambda p, b: full_partitioned.getBounds(p, b, relax=True, P=2),
    "layerwise bigm + interval (0.5)": lambda p, b: layerwise_bigm_interval.getBounds(p, b, 0.5),
    "layerwise bigm + interval (0.0)": lambda p, b: layerwise_bigm_interval.getBounds(p, b, 0),
    "layerwise bigm + lp (1.0)": lambda p, b: layerwise_bigm.getBounds(p, b, 1),
    "layerwise bigm + lp (0.5)": lambda p, b: layerwise_bigm.getBounds(p, b, 0.5),
    "layerwise bigm + lp (0.0)": lambda p, b: layerwise_bigm.getBounds(p, b, 0),
    "layerwise extended + lp (1.0)": lambda p, b: layerwise_extended.getBounds(p, b, 1),
    "layerwise extended + lp (0.5)": lambda p, b: layerwise_extended.getBounds(p, b, 0.5),
    "layerwise extended + lp (0.0)": lambda p, b: layerwise_extended.getBounds(p, b, 0),
    "layerwise partitioned + lp (1.0)": lambda p, b: layerwise_partitioned.getBounds(p, b, 1),
    "layerwise partitioned + lp (0.5)": lambda p, b: layerwise_partitioned.getBounds(p, b, 0.5),
    "layerwise partitioned + lp (0.0)": lambda p, b: layerwise_partitioned.getBounds(p, b, 0),
    "layerwise bigm + lp progressive": lambda p, b: layerwise_bigm.getBounds(p, b, [0.5] * len(p)),
    "layerwise extended + lp progressive": lambda p, b: layerwise_extended.getBounds(p, b, [0.5] * len(p)),
    "crown": crown.getBounds,
    "fastlin": fastlin.getBounds,
    "alpha-crown": alpha_crown.getBounds,
    "lirpa-ibp": alpha_crown.getBounds,
}

# layerwise mip bounding methods
LAYERWISE_MIP = {
    "layerwise bigm mip": lambda p, b: layerwise_bigm.getBounds(p, b, 1),
    "layerwise extended mip": lambda p, b: layerwise_extended.getBounds(p, b, 1),
    "layerwise partitioned mip": lambda p, b: layerwise_partitioned.getBounds(p, b, 1),
}

# full mip bounding methods
FULL_MIP = {
    "full bigm mip": lambda p, b: full_bigm.getBounds(p, b, relax=False),
    "full partitioned mip (P = 1)": lambda p, b: full_partitioned.getBounds(p, b, relax=False, P=1),
    "full partitioned mip (P = 2)": lambda p, b: full_partitioned.getBounds(p, b, relax=False, P=2),
}

# split functions to test with the partitioned model
SPLIT_FNS = [
    models.partitioned.uniformSplit,
    models.partitioned.equalSizeSplit,
    models.partitioned.posNegSplit
]

# linear systems to test
SYSTEMS = [
    control.example_systems.two_dim_control,
]

# bounding methods for multi-step bounds
MULTISTEP_BOUNDING_METHODS = {
    "layerwise bigm lp": lambda p, b, A, B, K: layerwise_bigm.getMultiStepBounds(p, b, A, B, K, relax=True),
    "layerwise partitioned lp (P=1)": lambda p, b, A, B, K: layerwise_partitioned.getMultiStepBounds(p, b, A, B, K, relax=True, P=1),
    "layerwise partitioned lp (P=2)": lambda p, b, A, B, K: layerwise_partitioned.getMultiStepBounds(p, b, A, B, K, relax=True, P=2),
    "full bigm lp": lambda p, b, A, B, K: full_bigm.getMultiStepBounds(p, b, A, B, K, relax=True),
    "full partitioned lp (P=1)": lambda p, b, A, B, K: full_partitioned.getMultiStepBounds(p, b, A, B, K, relax=True, P=1),
    "full partitioned lp (P=2)": lambda p, b, A, B, K: full_partitioned.getMultiStepBounds(p, b, A, B, K, relax=True, P=2),
    "interval": lambda p, b, A, B, K: interval.getMultiStepBounds(p, b, A, B, K),
    "lirpa-ibp": lambda p, b, A, B, K: alpha_crown.getMultiStepBounds(p, b, A, B, K, method='IBP'),
    "alpha-crown": lambda p, b, A, B, K: alpha_crown.getMultiStepBounds(p, b, A, B, K),
}

# full mip bounding methods for multi-step bounds
MULTISTEP_FULL_MIP = {
    "full bigm mip": lambda p, b, A, B, K: full_bigm.getMultiStepBounds(p, b, A, B, K, relax=False),
    "full partitioned mip (P=1)": lambda p, b, A, B, K: full_partitioned.getMultiStepBounds(p, b, A, B, K, relax=False, P=1),
    "full partitioned mip (P=2)": lambda p, b, A, B, K: full_partitioned.getMultiStepBounds(p, b, A, B, K, relax=False, P=2),
}

# layerwise mip bounding methods for multi-step bounds
MULTISTEP_LAYERWISE_MIP = {
    "layerwise bigm mip": lambda p, b, A, B, K: layerwise_bigm.getMultiStepBounds(p, b, A, B, K, relax=False),
    "layerwise partitioned mip (P=1)": lambda p, b, A, B, K: layerwise_partitioned.getMultiStepBounds(p, b, A, B, K, relax=False, P=1),
    "layerwise partitioned mip (P=2)": lambda p, b, A, B, K: layerwise_partitioned.getMultiStepBounds(p, b, A, B, K, relax=False, P=2),
}
