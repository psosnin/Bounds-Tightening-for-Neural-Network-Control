"""
Test the gurobi formulations of model predictive control
"""
import os
import sys
import unittest

import numpy as np
from numpy.random import uniform

sys.path.append('..')  # nopep8
from src import control, utils
from configs import SYSTEMS


class TestMPC(unittest.TestCase):

    def setUp(self):
        """
        Runs before all tests.
        """
        np.random.seed(0)
        os.environ["MIPTest"] = "1"

    def testControl(self):
        """
        Test the different encodings of the model predictive controller.
        """
        for system in SYSTEMS:
            with self.subTest(system=system['name']):
                # initialise the controllers
                C_in, c_in, x_in_bounds = system['C_in'], system['c_in'], system['x_in_bounds']
                N_p = 3
                controller1 = control.mpc.getMPController(system, N_p)
                controller2 = control.mpc.getKKTController(system, N_p)
                controller3 = control.mpc.getFJController(system, N_p)
                for _ in range(100):
                    # generate a point from the input set
                    x0 = uniform(low=x_in_bounds[0], high=x_in_bounds[1])
                    while not np.all(C_in @ x0 <= c_in):
                        x0 = uniform(*system['x_in_bounds'])
                    # get the control decisions
                    u1 = controller1(x0)
                    u2 = controller2(x0)
                    u3 = controller3(x0)
                    # check that the results are (approximately) the same
                    assert np.allclose(u1, u2, 1e-3, 1e-3), f"{u1 - u2}, {x0}"
                    assert np.allclose(u1, u3, 1e-3, 1e-3), f"{u1 - u3}, {x0}"
