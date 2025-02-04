"""
Test the validity of the bounding methods in src/bounds
"""
import os
import unittest
import sys
from itertools import product

import numpy as np
import torch
from numpy.random import uniform

sys.path.append('..')  # nopep8
from src import bounds, utils
from configs import SHAPES, MULTISTEP_BOUNDING_METHODS, MULTISTEP_FULL_MIP, SYSTEMS, MULTISTEP_LAYERWISE_MIP


class TestMultiStepBounds(unittest.TestCase):

    def setUp(self):
        """
        Runs before all the tests and initializes the bounding methods to test.
        """
        np.random.seed(0)
        os.environ["MIPTest"] = "1"

    def generateNetwork(self, shape):
        """
        Randomly generate a neural network with a random shape and set the input set and bounds
        """
        W = [np.random.uniform(-1, 1, size=(shape[i+1], shape[i])) for i in range(len(shape)-1)]
        b = [np.random.uniform(-1, 1, size=shape[i+1]) for i in range(len(shape)-1)]
        self.params = list(zip(W, b))

    def assertValidBounds(self, tight_bound, loose_bound):
        """
        Assert that the loose bounds are valid and looser than the tight bounds.
        """
        eps = 1e-5
        assert len(tight_bound) == len(loose_bound)
        for i in range(len(tight_bound)):
            assert len(tight_bound[i]) == len(loose_bound[i])
            for j in range(len(tight_bound[i])):
                # check that bounds have the same shape
                assert tight_bound[i][j][0].shape == loose_bound[i][j][0].shape
                # check that bounds are self-consistent
                assert np.all(tight_bound[i][j][0] <= tight_bound[i][j][1] + eps)
                assert np.all(loose_bound[i][j][0] <= loose_bound[i][j][1] + eps), f"{loose_bound[i][j]}, {i}, {j}"
                # check that tight bounds are tighter than loose bounds
                assert np.all(loose_bound[i][j][0] <= tight_bound[i][j][0] + eps), f"Bounds not consistent {i}"
                assert np.all(tight_bound[i][j][1] <= loose_bound[i][j][1] + eps), f"Bounds not consistent {i}"

    def assertEqualBounds(self, bound1, bound2):
        """
        Assert that both bounds are the same
        """
        assert len(bound1) == len(bound2)
        for i in range(len(bound1)):
            assert len(bound1[i]) == len(bound2[i])
            for j in range(len(bound1[i])):
                # check that bounds have the same shape
                assert bound1[i][j][0].shape == bound2[i][j][0].shape
                # check that bounds are self-consistent
                assert np.all(bound1[i][j][0] <= bound1[i][j][1])
                assert np.all(bound2[i][j][0] <= bound2[i][j][1])
                # check that the bounds are the same
                assert np.allclose(bound1[i][j][0], bound2[i][j][0], 1e-5, 1e-6)
                assert np.allclose(bound1[i][j][1], bound2[i][j][1], 1e-5, 1e-6)

    def testMultiBounds(self):
        """
        Generate a random neural network, calculate intermediate bounds and validate that they are correct.
        """
        for (shape, system) in product(SHAPES, SYSTEMS):
            # change the shape to be compatible with the linear system
            shape[0], shape[-1] = system["state_dim"], system["control_dim"]
            self.generateNetwork(shape)
            input_bounds = (uniform(-1, 0, size=shape[0]), uniform(0, 1, size=shape[0]))
            A, B, K = system["A"], system["B"], 2

            # get tightest upper and lower bounds via full MIP OBBT
            mip_bounds = bounds.full_bigm.getMultiStepBounds(self.params, input_bounds, A, B, K, relax=False)

            # check that other bounds are valid and looser than the exact method
            for name, method in MULTISTEP_BOUNDING_METHODS.items():
                with self.subTest(name, shape=shape, system=system["name"]):
                    test_bounds = method(self.params, input_bounds, A, B, K)
                    self.assertValidBounds(mip_bounds, test_bounds)

    def testMultiBoundShapes(self):
        """
        Check that the bounds returned by each method are of the correct shape.
        """
        for (shape, system) in product(SHAPES, SYSTEMS):
            # change the shape to be compatible with the linear system
            shape[0], shape[-1] = system["state_dim"], system["control_dim"]
            self.generateNetwork(shape)
            input_bounds = (uniform(-1, 0, size=shape[0]), uniform(0, 1, size=shape[0]))
            A, B, K = system["A"], system["B"], 2
            # check that other bounds are valid and looser than the exact method
            for name, method in MULTISTEP_BOUNDING_METHODS.items():
                with self.subTest(name, shape=shape, system=system["name"]):
                    test_bounds = method(self.params, input_bounds, A, B, K)
                    for bound in test_bounds[:-1]:
                        assert all([len(b) == 2 for b in bound])
                        bound_shapes = [len(b[0]) for b in bound]
                        self.assertEqual(bound_shapes, shape)
                    assert len(test_bounds[-1]) == 1

    def testMultiLayerwiseMIP(self):
        """
        Check that layerwise MIP bounds are the same for different formulations.
        """
        for (shape, system) in product(SHAPES, SYSTEMS):
            # change the shape to be compatible with the linear system
            shape[0], shape[-1] = system["state_dim"], system["control_dim"]
            with self.subTest(shape=shape, system=system["name"]):
                # generate a random network of the given shape
                self.generateNetwork(shape)
                # generate a random input domain
                input_bounds = (uniform(-1, 0, size=shape[0]), uniform(0, 1, size=shape[0]))
                A, B, K = system["A"], system["B"], 2
                layerwise_bounds = []

                for _, method in MULTISTEP_LAYERWISE_MIP.items():
                    layerwise_bounds.append(method(self.params, input_bounds, A, B, K))

                # check that the methods are pairwise the same
                for i in range(len(layerwise_bounds) - 1):
                    self.assertEqualBounds(layerwise_bounds[i], layerwise_bounds[i+1])

    def testMultiFullMIP(self):
        """
        Check that full MIP bounds are the same for different formulations.
        """
        for (shape, system) in product(SHAPES, SYSTEMS):
            # change the shape to be compatible with the linear system
            shape[0], shape[-1] = system["state_dim"], system["control_dim"]
            with self.subTest(shape=shape, system=system["name"]):
                # generate a random network of the given shape
                self.generateNetwork(shape)
                # generate a random input domain
                input_bounds = (uniform(-1, 0, size=shape[0]), uniform(0, 1, size=shape[0]))
                A, B, K = system["A"], system["B"], 2
                full_bounds = []
                for _, method in MULTISTEP_FULL_MIP.items():
                    full_bounds.append(method(self.params, input_bounds, A, B, K))

                # check that the methods are pairwise the same
                for i in range(len(full_bounds) - 1):
                    self.assertEqualBounds(full_bounds[i], full_bounds[i+1])

    def testLirpaIBP(self):
        """
        Check that the auto-Lirpa model is set up correctly by verifying that the IBP bounds are the same as ours.
        """
        for (shape, system) in product(SHAPES, SYSTEMS):
            # change the shape to be compatible with the linear system
            shape[0], shape[-1] = system["state_dim"], system["control_dim"]
            # generate a random network of the given shape
            self.generateNetwork(shape)
            # generate a random input domain
            input_bounds = (uniform(-1, 0, size=shape[0]), uniform(0, 1, size=shape[0]))
            A, B, K = system["A"], system["B"], 2

            for K in range(5):
                with self.subTest(K=K, shape=shape, system=system["name"]):
                    ibp = bounds.interval.getMultiStepBounds(self.params, input_bounds, A, B, K)
                    ibp2 = bounds.alpha_crown.getMultiStepBounds(self.params, input_bounds, A, B, K, method='IBP')
                    self.assertEqualBounds(ibp, ibp2)

    def testLirpaModel(self):
        """
        Test that the pytorch Lirpa model correctly encodes the multi-timestep problem.
        """
        for (shape, system) in product(SHAPES, SYSTEMS):
            # change the shape to be compatible with the linear system
            shape[0], shape[-1] = system["state_dim"], system["control_dim"]
            with self.subTest(shape=shape, system=system["name"]):
                self.generateNetwork(shape)
                input_bounds = (uniform(-1, 0, size=shape[0]), uniform(0, 1, size=shape[0]))
                A, B, K = system["A"], system["B"], 2
                model = bounds.alpha_crown.MultiStepModel(self.params, input_bounds, A, B, K)

                input_point = uniform(input_bounds[0], input_bounds[1], size=(1, shape[0]))
                torch_input_point = torch.from_numpy(input_point).float()
                rescaled_input_point = (torch_input_point - model.x_mu) / model.x_rad

                model_output = model.forward(rescaled_input_point)

                x = input_point.flatten()
                for _ in range(K):
                    nn_out = utils.network.forward(x, self.params)
                    x = A @ x + B @ nn_out

                assert np.allclose(x, model_output.detach().cpu().numpy(), 1e-5, 1e-6)
