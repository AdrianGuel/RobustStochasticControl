# final_algorithm.py

import numpy as np
import plotly.graph_objects as go
from src.simulate_experiment import simulate_experiment
from src.least_squares_offline import least_squares_offline
from src.recursive_least_squares import recursive_least_squares

# Parameters
gamma = 1.0
a = 1.0
n = 2
ts = 0.1

# Simulate system
t, u, y = simulate_experiment(duration=30, dt=ts)
theta_0, P, f_k = least_squares_offline(u, y, n, a, gamma)

# Initialize
theta = theta_0.copy()
Theta = [theta.flatten()]
y_est = []

# Online estimation
for k in range(n, len(u)):
    y_k = y[k]
    u_k = u[k]
    theta, P, f_k = recursive_least_squares(u_k, y_k, P, theta, f_k, a, gamma, n)
    y_hat = np.squeeze(f_k.T @ theta)
    y_est.append(y_hat)
    Theta.append(theta.flatten())

# Plot results
t_valid = t[n:]
fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=y, name="True Output"))
fig.add_trace(go.Scatter(x=t_valid, y=y_est, name="Estimated Output"))
fig.update_layout(title="Output vs Estimated Output", xaxis_title="Time", yaxis_title="Output")
fig.show()
