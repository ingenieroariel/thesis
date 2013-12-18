import numpy as np
from scipy import sparse as sp


def lasso_admm(X, A, gamma=1, C=None):
    c = X.shape[1]
    r = A.shape[1]

    L = np.zeros(X.shape)
    rho = 1e-4
    max_iter = 500
    I = sp.eye(r)
    max_rho = 5

    # Initialize C randomly if it is not passed
    if C is None:
        C = np.random.randn(r, c)

    fast_sthresh = lambda x, th: np.sign(x) * np.maximum(np.abs(x) - th, 0)

    norm2 = lambda x: np.dot(x.flatten(1).transpose(), x.flatten(1))
    norm1 = lambda x: np.sum(np.abs(x.flatten(1)))

    cost = []

    assert norm2(np.array([1,2])) == 5

    for n in range(1, max_iter):
        # Solve sub-problem to solve B
        B = np.linalg.solve(np.dot(A.transpose(), A) + np.dot(rho, I), np.dot(A.transpose(), X) + np.dot(rho,C) - L)
    
        # Solve sub-problem to solve C
        C = fast_sthresh(B + L/rho, gamma/rho)
    
        # Update the Lagrangian
        L = L + np.dot(rho, B - C)
    
        rho = min(max_rho, rho*1.1)
    
        # get the current cost
        current_cost = 0.5 * norm2(X - np.dot(A, B)) + gamma * norm1(B)
        cost.append(current_cost)
    
        print current_cost
    return B, cost
