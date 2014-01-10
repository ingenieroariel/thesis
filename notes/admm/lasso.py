import numpy as np
from scipy import sparse as sp


def lasso_admm(X, A, gamma=1, C=None, double=False, max_rho=5.0, rho=1e-4, max_iter=500):
    # Cast to matrix if input is an array
    if len(A.shape) == 1:
        A = A.reshape(A.shape[0],1)

    c = X.shape[1]
    r = A.shape[1]

    L = np.zeros((r, c))
    I = sp.eye(r)

    # Initialize C with zeros if it is not passed
    if C is None:
        C = np.zeros((r, c))

    fast_sthresh = lambda x, th: np.sign(x) * np.maximum(np.abs(x) - th, 0)

    norm2 = lambda x: np.dot(x.flatten(1).transpose(), x.flatten(1))
    norm1 = lambda x: np.sum(np.abs(x.flatten(1)))

    cost = []

    for n in range(1, max_iter):
        #import ipdb;ipdb.set_trace()
        # Define terms for sub-problem
        F = np.dot(A.transpose(), A) + np.dot(rho, I)
        G = np.dot(A.transpose(), X) + np.dot(rho,C) - L

        B,resid,rank,s = np.linalg.lstsq(F,G)

        # Verify B is a solution to: F dot G = B
        np.testing.assert_array_almost_equal(np.dot(F, B), G)

        # Solve sub-problem to solve C
        C = fast_sthresh(B + L/rho, gamma/rho)
    
        # Update the Lagrangian
        L = L + np.dot(rho, B - C)
    
        rho = min(max_rho, rho*1.1)
    
        # get the current cost
        current_cost = 0.5 * norm2(X - np.dot(A, B)) + gamma * norm1(B)
        cost.append(current_cost)
    
    return B, cost
