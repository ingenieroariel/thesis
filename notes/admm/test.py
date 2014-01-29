import scipy.io as sio
from lasso import lasso_admm, lasso_cost
import unittest
import numpy as np
from scipy import linalg


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


    def test_sparse_encode(self):
        """Test the sparse encode using admm behaves like sklearn's sparse_encode
        """
        from sklearn.decomposition import sparse_encode

        alpha = 1
        X = np.array([[ 1.76405235,  0.40015721,  0.97873798,  2.2408932 ,  1.86755799,
                        -0.97727788,  0.95008842, -0.15135721],
                       [-0.10321885,  0.4105985 ,  0.14404357,  1.45427351,  0.76103773,
                         0.12167502,  0.44386323,  0.33367433],
                       [ 1.49407907, -0.20515826,  0.3130677 , -0.85409574, -2.55298982,
                         0.6536186 ,  0.8644362 , -0.74216502],
                       [ 2.26975462, -1.45436567,  0.04575852, -0.18718385,  1.53277921,
                         1.46935877,  0.15494743,  0.37816252],
                       [-0.88778575, -1.98079647, -0.34791215,  0.15634897,  1.23029068,
                         1.20237985, -0.38732682, -0.30230275],
                       [-1.04855297, -1.42001794, -1.70627019,  1.9507754 , -0.50965218,
                        -0.4380743 , -1.25279536,  0.77749036],
                       [-1.61389785, -0.21274028, -0.89546656,  0.3869025 , -0.51080514,
                        -1.18063218, -0.02818223,  0.42833187],
                       [ 0.06651722,  0.3024719 , -0.63432209, -0.36274117, -0.67246045,
                        -0.35955316, -0.81314628, -1.7262826 ],
                       [ 0.17742614, -0.40178094, -1.63019835,  0.46278226, -0.90729836,
                         0.0519454 ,  0.72909056,  0.12898291],
                       [ 1.13940068, -1.23482582,  0.40234164, -0.68481009, -0.87079715,
                        -0.57884966, -0.31155253,  0.05616534]])

        # start with sensible defaults
        code, S, dictionary = linalg.svd(X, full_matrices=False)

        code_sklearn = sparse_encode(X, dictionary, alpha=alpha, init=code)

        #TODO: Verify that sparse_encode indeed returns a code that is sparse:
        #np.testing.assert_almost_equal(code, code_sklearn)
        norm1 = lambda x: np.sum(np.abs(x.flatten(1)))

        self.assertGreaterEqual(norm1(code), norm1(code_sklearn))

        code_admm, costs = lasso_admm(X, dictionary, gamma=alpha)

        #TODO: Verify that lasso_admm indeed returns a code that is sparse:
        self.assertGreaterEqual(norm1(code), norm1(code_admm))

        # Check if both codes are similar.
        #np.testing.assert_array_almost_equal(code_sklearn, code_admm)
        norm2 = lambda x: np.dot(x.flatten(1).transpose(), x.flatten(1))
        cost = lambda X, code, dictionary: 0.5* norm2(X - np.dot(code, dictionary))

        # Compare the costs with svd.
        cost_svd =  lasso_cost(X, code, dictionary, alpha)
        cost_sklearn =  lasso_cost(X, code_sklearn, dictionary, alpha)
        cost_admm =  lasso_cost(X, code_admm, dictionary, alpha)

        self.assertLessEqual(cost_sklearn, cost_svd)
        import ipdb;ipdb.set_trace()
#        self.assertLessEqual(cost_admm, cost_svd)


    def test_dictionary_learning(self):
        """Test admm dictionary learning behaves like sklearn's dict_learning
        """
        from sklearn.decomposition import dict_learning

        rng_global = np.random.RandomState(0)
        n_samples, n_features = 10, 8
        X = rng_global.randn(n_samples, n_features)

        rng = np.random.RandomState(0)
        n_components = 8
        code, dictionary, errors = dict_learning(X, n_components=n_components,
                                                 alpha=1, random_state=rng)

        np.testing.assert_almost_equal(code.shape, (n_samples, n_components))
        np.testing.assert_almost_equal(dictionary.shape, (n_components, n_features))
        np.testing.assert_almost_equal(np.dot(code, dictionary).shape, X.shape)


        rng = np.random.RandomState(0)

        from lasso import dict_learning as admm_dict_learning
        code2, dictionary2, errors2 = admm_dict_learning(X, method='admm',
                                                      n_components=n_components,
                                                      alpha=1, random_state=rng)

        np.testing.assert_almost_equal(code2.shape, (n_samples, n_components))
        np.testing.assert_almost_equal(dictionary2.shape, (n_components, n_features))
        np.testing.assert_almost_equal(np.dot(code2, dictionary2).shape, X.shape)

        # And compare it with the B obtained manually
        #np.testing.assert_array_almost_equal(code, code2)
        #np.testing.assert_array_almost_equal(dictionary, dictionary2)


if __name__ == '__main__':
    unittest.main()
