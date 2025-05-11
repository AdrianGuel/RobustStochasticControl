import numpy as np

def least_squares_offline(u, y, n, a, gamma):
    """
    Offline weighted least squares estimator.
    
    Parameters:
    u : ndarray
        Input signal (1D array)
    y : ndarray
        Output signal (1D array)
    n : int
        Model order
    a : float
        Weighting factor (typically 1)
    gamma : float
        Forgetting factor (between 0 and 1)
    
    Returns:
    theta_0 : ndarray
        Estimated parameter vector
    P : ndarray
        Covariance matrix
    f_0 : ndarray
        Last regressor vector
    """
    u = np.asarray(u).flatten()
    y = np.asarray(y).flatten()
    N = len(u) - 1
    rows = N - n + 1
    
    # Construct the weighting matrix W
    W = np.diag([a * gamma**(N - i) for i in range(rows)])
    
    # Construct the regression matrix F and the output vector Y
    F = np.zeros((rows, 2 * n))
    for k in range(rows):
        F[k, :n] = y[k + n - 1::-1][:n]
        F[k, n:] = u[k + n - 1::-1][:n]
    
    Y = y[n:]
    
    # Least squares estimation
    P = np.linalg.inv(F.T @ W @ F)
    f_0 = F[-1, :]
    theta_0 = P @ F.T @ W @ Y
    
    return theta_0, P, f_0
