# recursive_least_squares.py

import numpy as np

def recursive_least_squares(u_k, y_k, P, theta, f_k, a=1.0, gamma=1.0, n=2):
    """
    Recursive least squares update.

    Args:
        u_k: current input
        y_k: current output
        P: previous covariance matrix
        theta: previous parameter estimate
        f_k: previous regression vector
        a, gamma: forgetting factors
        n: model order

    Returns:
        theta_k: updated parameters
        P_k: updated covariance
        f_k: updated regression vector
    """
    f_k = f_k.reshape(-1, 1)
    L_k = (1 / gamma) * P @ f_k / ((1 / a) + (1 / gamma) * f_k.T @ P @ f_k)
    theta_k = theta + L_k * (y_k - f_k.T @ theta)
    P_k = (1 / gamma) * (np.eye(len(P)) - L_k @ f_k.T) @ P

    Ya = np.roll(f_k[:n], 1)
    Yb = np.roll(f_k[n:], 1)
    f_k_new = np.vstack([Ya, Yb])
    f_k_new[0, 0] = y_k
    f_k_new[n, 0] = u_k
    return theta_k, P_k, f_k_new
