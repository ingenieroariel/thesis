import numpy as np
from lasso import lasso_admm, lasso_cost, init_dictionary
from lasso import dict_learning as admm_dict_learning

rng_global = np.random.RandomState(0)
n_samples, n_features = 10, 8
X = rng_global.randn(n_samples, n_features)

n_components = 6
rng = np.random.RandomState(0)

code2, dictionary2, errors2 = admm_dict_learning(X, method='admm',
                                              n_components=n_components,
                                              alpha=1, random_state=rng)
