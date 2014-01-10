import scipy.io as sio
from lasso import lasso_admm, lasso_cost
import unittest
import numpy as np

class TestSequenceFunctions(unittest.TestCase):
    def setUp(self):
        D = sio.loadmat('admm.mat')
        self.A = D['A']
        self.X = D['X']
        self.B_ref = D['B']
        self.cost = D['cost']



    def test_admm(self):
        """
        Test lasso_admm works.

        The values were obtained by running them via matlab.
        """
        B , cost = lasso_admm(self.X, self.A, gamma=1)
        np.testing.assert_array_almost_equal(B, self.B_ref)
        np.testing.assert_almost_equal(cost[-1], self.cost.flatten(1)[-1])


    def test_admm_non_square(self):
        """
        Test lasso_admm works when A is not square
        """
        A = np.random.randn(10,9)
        X = np.random.randn(10,8)
        B , cost = lasso_admm(X, A, gamma=1)


    def test_admm_solve(self):
        """
        Test lasso_admm works as a solver when gamma is zero
        """
        # First let's define an equation of the form: X*B = A, where B is the unknown.
        X = np.array([[8],[4]])
        A = np.array([[2,3], [5, -2]])
        # Let's verify the solution obtained manually works
        B_manual = np.array([[ 1.47368421],  [1.68421053]])
        np.testing.assert_array_almost_equal(X, np.dot(A, B_manual))

        # Now let's solve it using lstsq
        B_lstsq = np.linalg.lstsq(A, X)[0]
        # And compare with the manual solution
        np.testing.assert_array_almost_equal(B_lstsq, B_manual)

        # We drive gamma to zero to ignore the sparsity requirement of the problem
        # and have it perform like a regular solver.
        gamma = 0
        # This an exact solution, the cost should be zero.
        cost_expected = 0

        # Now let's use ADMM to solve the same equation
        B_admm, cost_admm = lasso_admm(X, A, gamma=gamma)

        # Let's evaluate the cost of the manual solution
        cost_manual = lasso_cost(X, A, B_manual, gamma)

        # Check the cost is what we expected
        np.testing.assert_almost_equal(cost_manual, cost_expected)

        # And compare it with the B obtained manually
        np.testing.assert_array_almost_equal(B_admm, B_manual)

        # Check the expected cost is also the result of lasso_admm
        np.testing.assert_almost_equal(cost_admm[-1], cost_expected)

        # Check the cost is higher when gamma is bigger than 0.
        gamma = 1
        B_admm2, cost_admm2 = lasso_admm(X, A, gamma=gamma)
        cost_nonzero_gamma = cost_admm2[-1]

        msg = "Expected cost higher than %s for gamma=%s, got %s" % (cost_expected, gamma, cost_nonzero_gamma)
        assert cost_nonzero_gamma > cost_expected


if __name__ == '__main__':
    unittest.main()
