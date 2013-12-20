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

if __name__ == '__main__':
    unittest.main()
