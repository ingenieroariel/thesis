import scipy.io as sio
from lasso import lasso_admm
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
        # First let's define an equation of the form: X*B = A, where B is the unknown.
        A = np.array([8,4])
        X = np.array([[2,3], [5, -2]])
        # Let's verify the solution obtained manually works
        B_manual = np.array([ 1.47368421,  1.68421053])
        np.testing.assert_array_almost_equal(A, np.dot(X, B_manual))


        # Now let's solve it using lstsq
        B_lstsq = np.linalg.lstsq(X, A)[0]
        # And compare with the manual solution
        np.testing.assert_array_almost_equal(B_lstsq, B_manual)

        # Now let's use ADMM to solve the same equation
        B_admm, cost = lasso_admm(X, A, gamma=1e-10)

        # Is this an exact solution? The cost should be zero, right?
        np.testing.assert_almost_equal(cost[-1], 0)

        # And compare it with the B obtained manually
        np.testing.assert_array_almost_equal(B_admm, B_manual)



if __name__ == '__main__':
    unittest.main()
