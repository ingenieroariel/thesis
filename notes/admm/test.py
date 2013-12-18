import scipy.io as sio
from lasso import lasso_admm
import unittest
import numpy as np

class TestSequenceFunctions(unittest.TestCase):
  def setUp(self):
        D = sio.loadmat('admm.mat')
        self.A = D['A']
        self.X = D['X']
        self.C = D['C_initial']
        self.B_ref = D['B']
        self.cost = D['cost']

  def test_admm_no_random(self):
      """
      Test lasso_admm works with a preseeded C
      """
      B , cost = lasso_admm(self.X, self.A, gamma=1, C=self.C)
      np.testing.assert_array_almost_equal(B, self.B_ref)
      assert self.cost[499] == cost[499]

  def test_admm(self):
      """
      Test lasso_admm works
      """
      B , cost = lasso_admm(self.X, self.A, gamma=1)
      np.testing.assert_array_almost_equal(B, self.B_ref)


if __name__ == '__main__':
    unittest.main()
