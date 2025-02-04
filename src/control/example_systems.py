"""
Parameters for the inverted pendulum system.
"""

import numpy as np

# example B from http://ieeexplore.ieee.org/document/5617688/
two_dim_control = {
    'name': 'two_dim_control',
    'A': np.array([
        [0.7, -0.1, 0.0, 0.0],
        [0.2, -0.5, 0.1, 0.0],
        [0.0, 0.1, 0.1, 0.0],
        [0.5, 0.0, 0.5, 0.5]
    ]),
    'B': np.array([[0.0, 0.1], [0.1, 1.0], [0.1, 0.0], [0.0, 0.0]]),
    'Q': np.eye(4),  # state cost
    'R': np.eye(2),  # constrol cost
    'P': np.array([
        [3.1852, -0.1511, 0.5518, 0.5047],
        [-0.1511, 1.1520, -0.0158, -0.0007],
        [0.5518, -0.0158, 1.3823, 0.3483],
        [0.5047, -0.0007, 0.3483, 1.3326]
    ]),  # terminal cost from idare
    'C_in': np.array([
        [0.8359,   0.0354,   0.4179,   0.3542],
        [-0.8359,  -0.0354,  -0.4179,  -0.3542],
        [-0.8593,   0.0716,  -0.3581,  -0.3581],
        [0.8381,   0.0000,   0.4191,   0.3492],
        [0.8593,  -0.0716,   0.3581,   0.3581],
        [-0.8381,   0.0000,  -0.4191,  -0.3492],
        [-0.5774,   0.0000,  -0.5774,  -0.5774],
        [0.5774,   0.0000,   0.5774,   0.5774],
        [0.0000,   1.0000,   0.0000,   0.0000],
        [0.0000,   0.0000,   1.0000,   0.0000],
        [0.0000,   0.0000,   0.0000,   1.0000],
        [0.0000,  -1.0000,   0.0000,   0.0000],
        [0.0000,   0.0000,  -1.0000,   0.0000],
        [0.0000,   0.0000,   0.0000,  -1.0000]
    ]),  # polyhedral input domain, which here is the maximum invariant set computed using MPT3
    'c_in': np.array([
        1.4521, 1.4521, 1.7903, 1.3969,
        1.7903, 1.3969, 0.5774, 0.5774,
        6.0000, 1.0000, 0.5000, 6.0000,
        1.0000, 0.5000
    ]),
    # bounding box of C_in @ x <= c_in
    'x_in_bounds': (-np.array([2.37513423, 6.0, 1.0, 0.5]), np.array([2.37513423, 6.0, 1.0, 0.5])),
    'u_bounds': (np.array([-5, -5]), np.array([5, 5])),
    'x_bounds': (np.array([-6, -6, -1, -0.5]), np.array([6, 6, 1, 0.5])),
    'control_dim': 2,
    'state_dim': 4,
    'C_x': np.vstack((np.eye(4), -np.eye(4))),  # polyhedral representation of state bounds
    'c_x': np.array([6, 6, 1, 0.5, 6, 6, 1, 0.5]),
    'C_u': np.vstack((np.eye(2), -np.eye(2))),  # polyhedral representation of control bounds
    'c_u': np.array([5, 5, 5, 5])
}


def getSystem(system_name):
    """
    Function to get the system parameters by name.
    """
    if system_name == "two_dim_control":
        return two_dim_control
    else:
        raise ValueError(f"System {system_name} not found.")
