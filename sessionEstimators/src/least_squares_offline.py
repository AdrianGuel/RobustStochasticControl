# least_squares_offline.py

import numpy as np

def least_squares_offline(u, y, n, a=1.0, gamma=1.0):
    """
    Offline weighted least squares estimation.

    Args:
        u: input vector
        y: output vector
        n: model order
        a: weighting parameter
        gamma: decay factor

    Returns:
        theta_0: estimated parameters
        P: covariance matrix
        f_0: last regression vector
    """
    N = len(u) - 1
    W = np.diag([a * gamma**(N - i) for i in range(N - n + 1)])
    F = np.zeros((N - n + 1, 2 * n))
    Y = y[n:]

    for k in range(N - n + 1):
        F[k, :n] = y[k : k + n][::-1]  # reverse y
        F[k, n:] = u[k : k + n][::-1]  # reverse u

    theta_0 = np.linalg.inv(F.T @ W @ F) @ F.T @ W @ Y
    P = np.linalg.inv(F.T @ W @ F)
    f_0 = np.concatenate([y[-n:][::-1], u[-n:][::-1]])
    return theta_0, P, f_0
