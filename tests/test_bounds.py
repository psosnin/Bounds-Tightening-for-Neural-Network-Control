"""
Test the validity of the bounding methods in src/bounds
"""
import os
import unittest
import sys

import numpy as np
from numpy.random import uniform

sys.path.append('..')  # nopep8
from src import bounds
from configs import SHAPES, BOUNDING_METHODS, LAYERWISE_MIP, FULL_MIP


class TestBounds(unittest.TestCase):

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

    def assertValid(self, tight_bound, loose_bound):
        """
        Assert that the loose bounds are valid and looser than the tight bounds.
        """
        eps = 1e-4
        assert len(tight_bound) == len(loose_bound)
        for i in range(len(tight_bound)):
            # check that bounds have the same shape
            assert tight_bound[i][0].shape == loose_bound[i][0].shape
            # check that bounds are self-consistent
            assert np.all(tight_bound[i][0] <= tight_bound[i][1])
            assert np.all(loose_bound[i][0] <= loose_bound[i][1])
            # check that tight bounds are tighter than loose bounds
            assert np.all(loose_bound[i][0] <= tight_bound[i][0] + eps), f"Bounds not consistent for layer {i}"
            assert np.all(tight_bound[i][1] <= loose_bound[i][1] + eps), f"Bounds not consistent for layer {i}"

    def testBounds(self):
        """
        Generate a random neural network, calculate intermediate bounds and validate that they are correct.
        """
        for shape in SHAPES:
            self.generateNetwork(shape)

            # first check bounding methods that work on an arbitrary box
            input_bounds = (uniform(-1, 0, size=shape[0]), uniform(0, 1, size=shape[0]))

            # get tightest upper and lower bounds via full MIP OBBT
            mip_bounds = bounds.full_bigm.getBounds(self.params, input_bounds, relax=False)

            # check that other bounds are valid and looser than the exact method
            for name, method in BOUNDING_METHODS.items():
                with self.subTest(name, shape=shape):
                    test_bounds = method(self.params, input_bounds)
                    self.assertValid(mip_bounds, test_bounds)

    def testBoundShapes(self):
        """
        Check that the bounds returned by each method are of the correct shape.
        """
        for shape in SHAPES:
            self.generateNetwork(shape)
            # first check bounding methods that work on an arbitrary box
            input_bounds = (uniform(-1, 0, size=shape[0]), uniform(0, 1, size=shape[0]))
            # check that other bounds are valid and looser than the exact method
            for name, method in BOUNDING_METHODS.items():
                with self.subTest(name, shape=shape):
                    test_bounds = method(self.params, input_bounds)
                    assert all([len(b) == 2 for b in test_bounds])
                    bound_shapes = [len(b[0]) for b in test_bounds]
                    self.assertEqual(bound_shapes, shape)

    def testLayerwiseMIP(self):
        """
        Check that layerwise MIP bounds are the same for different formulations.
        """
        for shape in SHAPES:
            with self.subTest(shape=shape):
                # generate a random network of the given shape
                self.generateNetwork(shape)
                # generate a random input domain
                input_bounds = (uniform(-1, 0, size=shape[0]), uniform(0, 1, size=shape[0]))

                # generate bounds for each layerwise MIP method
                layerwise_bounds = []
                names = []
                for name, method in LAYERWISE_MIP.items():
                    layerwise_bounds.append(method(self.params, input_bounds))
                    names.append(name)

                # check that the methods are pairwise the same
                for i in range(len(layerwise_bounds) - 1):
                    for j in range(len(layerwise_bounds[0])):
                        self.assertTrue(
                            np.allclose(layerwise_bounds[i][j], layerwise_bounds[i+1][j]),
                            msg=f"{names[i]} != {names[i+1]}"
                        )
                        self.assertTrue(
                            np.allclose(layerwise_bounds[i][j], layerwise_bounds[i+1][j]),
                            msg=f"{names[i]} != {names[i+1]}"
                        )

    def testFullMIP(self):
        """
        Check that full MIP bounds are the same for different formulations.
        """
        for shape in SHAPES:
            with self.subTest(shape=shape):
                # generate a random network of the given shape
                self.generateNetwork(shape)
                # generate a random input domain
                input_bounds = (uniform(-1, 0, size=shape[0]), uniform(0, 1, size=shape[0]))

                # generate bounds for each MIP method
                full_bounds = []
                names = []
                for name, method in FULL_MIP.items():
                    full_bounds.append(method(self.params, input_bounds))
                    names.append(name)
                # check that the methods are pairwise the same
                for i in range(len(full_bounds) - 1):
                    for j in range(len(full_bounds[0])):
                        assert np.allclose(full_bounds[i][j], full_bounds[i+1][j]), f"{names[i]} != {names[i+1]}"
                        assert np.allclose(full_bounds[i][j], full_bounds[i+1][j]), f"{names[i]} != {names[i+1]}"

    def testLirpa(self):
        """
        Check that the auto-Lirpa model is set up correctly by verifying that the IBP bounds are the same as ours.
        """
        for shape in SHAPES:
            with self.subTest(shape=shape):
                # generate a random network of the given shape
                self.generateNetwork(shape)
                # generate a random input domain
                input_bounds = (uniform(-1, 0, size=shape[0]), uniform(0, 1, size=shape[0]))

                interval1 = bounds.interval.getBounds(self.params, input_bounds)
                interval2 = bounds.alpha_crown.getBounds(self.params, input_bounds, method='IBP')
                for i in range(len(interval1)):
                    assert np.allclose(interval1[i][0], interval2[i][0])
                    assert np.allclose(interval1[i][1], interval2[i][1])
