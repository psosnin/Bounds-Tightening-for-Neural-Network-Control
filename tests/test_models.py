"""
Test the MILP models in src/models
"""

import os
import unittest
import sys
import math

import numpy as np
from numpy.random import uniform

sys.path.append('..')  # nopep8
from src import models, bounds, utils
from configs import SHAPES, SPLIT_FNS


class TestModels(unittest.TestCase):

    def setUp(self):
        """
        Runs before all the tests.
        """
        np.random.seed(0)
        os.environ["MIPTest"] = "1"

    def generateNetwork(self, shape):
        """
        Randomly generate a neural network with a random shape and set the input set and bounds
        """
        self.input_bounds = (uniform(-1, 0, size=shape[0]), uniform(0, 1, size=shape[0]))
        self.x = uniform(self.input_bounds[0], self.input_bounds[1], size=shape[0])
        W = [np.random.uniform(-1, 1, size=(shape[i+1], shape[i])) for i in range(len(shape)-1)]
        b = [np.random.uniform(-1, 1, size=shape[i+1]) for i in range(len(shape)-1)]
        self.params = list(zip(W, b))
        self.bounds = bounds.interval.getBounds(self.params, self.input_bounds)
        self.y = utils.network.forward(self.x, self.params)

    def initialiseBigM(self):
        """
        Initialise the bigM formulation
        """
        m = utils.helpers.getGurobiModel()
        in_var = m.addMVar(self.params[0][0].shape[1], lb=self.input_bounds[0], ub=self.input_bounds[1])
        out_var = m.addMVar(self.params[-1][1].size, lb=-np.inf, ub=np.inf)
        models.bigm.addBigMToModel(m, in_var, out_var, self.params, self.bounds, False)
        return m, in_var, out_var

    def initialiseExtended(self):
        """
        Initialise the extended formulation
        """
        m = utils.helpers.getGurobiModel()
        in_var = m.addMVar(self.params[0][0].shape[1], lb=self.input_bounds[0], ub=self.input_bounds[1])
        out_var = m.addMVar(self.params[-1][1].size, lb=-np.inf, ub=np.inf)
        models.extended.addExtendedToModel(m, in_var, out_var, self.params, self.bounds, False)
        return m, in_var, out_var

    def initialisePartitioned(self, P, split_fn):
        """
        Initialise the partitioned formulation
        """
        m = utils.helpers.getGurobiModel()
        in_var = m.addMVar(self.params[0][0].shape[1], lb=self.input_bounds[0], ub=self.input_bounds[1])
        out_var = m.addMVar(self.params[-1][1].size, lb=-np.inf, ub=np.inf)
        models.partitioned.addPartitionedToModel(
            m, in_var, out_var, self.params, self.bounds, split_fn, P, False
        )
        return m, in_var, out_var

    def testMILPEncodings(self):
        """
        Generate a random neural network with the given shape and check that it is correctly encoded in MILPs.
        """
        for shape in SHAPES:
            with self.subTest(shape=shape):
                self.generateNetwork(shape)
                m_bigm, in_bigm, out_bigm = self.initialiseBigM()
                m_ext, in_ext, out_ext = self.initialiseExtended()
                m_part, in_part, out_part = self.initialisePartitioned(2, models.partitioned.uniformSplit)

                # test that the big-m model correctly encodes the forward pass
                c = m_bigm.addConstr(in_bigm == self.x)
                m_bigm.optimize()
                self.assertTrue(np.allclose(self.y, out_bigm.x), msg="Error in bigM forward encoding")
                m_bigm.remove(c)
                m_bigm.reset()

                # test that the extended model correctly encodes the forward pass
                c = m_ext.addConstr(in_ext == self.x)
                m_ext.optimize()
                self.assertTrue(np.allclose(self.y, out_ext.x), msg="Error in extended forward encoding")
                m_ext.remove(c)
                m_ext.reset()

                # test that the partitioned model correctly encodes the forward pass
                c = m_part.addConstr(in_part == self.x)
                m_part.optimize()
                self.assertTrue(np.allclose(self.y, out_part.x), msg="Error in partitioned forward encoding")
                m_part.remove(c)
                m_part.reset()

                # test that all formulations give the same objective value
                m_bigm.setObjective(out_bigm.sum())
                m_bigm.optimize()
                m_ext.setObjective(out_ext.sum())
                m_ext.optimize()
                m_part.setObjective(out_part.sum())
                m_part.optimize()
                self.assertAlmostEqual(
                    m_bigm.objVal, m_ext.objVal, delta=1e-6, msg="BigM and extended models do not match"
                )
                self.assertAlmostEqual(
                    m_bigm.objVal, m_part.objVal, delta=1e-6, msg="BigM and partitioned models do not match"
                )

    def testPartitioningFns(self):
        """
        Test the forward encoding and optimisation of different partitioned formulation configurations.
        """
        for shape in SHAPES:
            with self.subTest(shape=shape):
                self.generateNetwork(shape)
                # get bigm as base case
                m_bigm, in_bigm, out_bigm = self.initialiseBigM()
                m_bigm.setObjective(out_bigm.sum())
                m_bigm.optimize()

                for split_fn in SPLIT_FNS:
                    for P in [1, math.ceil(max(shape) / 2), max(shape)]:
                        m_part, in_part, out_part = self.initialisePartitioned(P, split_fn)
                        m_part.setObjective(out_part.sum())
                        m_part.optimize()
                        true_forward = utils.network.forward(in_part.x, self.params)
                        # check that the partitioned model correctly encodes the forward pass
                        self.assertTrue(
                            np.allclose(out_part.x, true_forward),
                            msg=f"Error in forward encoding for P={P}, split_fn={split_fn.__name__}"
                        )
                        # check that optimisation with the partitioned model gives the same as bigm
                        self.assertAlmostEqual(
                            m_bigm.objVal, m_part.objVal, delta=1e-6,
                            msg=f"Partitioned model P={P}, split_fn={split_fn.__name__} does not match bigM"
                        )
