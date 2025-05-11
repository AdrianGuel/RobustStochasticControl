import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from numpy.linalg import inv

# First set of measurements
t1 = np.array([0, 1, 2])
Y1 = np.array([6, 0, 0]).reshape(-1, 1)
H1 = np.vstack([np.ones_like(t1), t1, t1**2]).T

# Least squares estimate for original 3 points
X1_hat = inv(H1.T @ H1) @ H1.T @ Y1

# Generate time values for plotting
t_vals = np.linspace(0, 4, 100)
y_fit1 = X1_hat[0] + X1_hat[1] * t_vals + X1_hat[2] * t_vals**2

# Second set with additional point at t=0.5
t2 = np.array([0, 0.5, 1, 2])
Y2 = np.array([6, 5, 0, 0]).reshape(-1, 1)
H2 = np.vstack([np.ones_like(t2), t2, t2**2]).T

# New least squares estimate
X2_hat = inv(H2.T @ H2) @ H2.T @ Y2
y_fit2 = X2_hat[0] + X2_hat[1] * t_vals + X2_hat[2] * t_vals**2

# Plot both fits
fig = make_subplots(rows=1, cols=2, subplot_titles=[
    "y = 6 - 9t + 3t²", f"y = {X2_hat[0,0]:.1f} {X2_hat[1,0]:+.1f}t {X2_hat[2,0]:+.1f}t²"])

fig.add_trace(go.Scatter(x=t_vals, y=y_fit1.flatten(), mode='lines', name='Fit 1'), row=1, col=1)
fig.add_trace(go.Scatter(x=t_vals, y=y_fit2.flatten(), mode='lines', name='Fit 2'), row=1, col=2)
fig.add_trace(go.Scatter(x=t2, y=Y2.flatten(), mode='markers', name='Data Points'), row=1, col=2)

fig.update_layout(title_text="Parabola Fitting using Least Squares", showlegend=False)
fig.update_xaxes(title_text="Time", row=1, col=1)
fig.update_xaxes(title_text="Time", row=1, col=2)
fig.update_yaxes(title_text="Value", row=1, col=1)
fig.update_yaxes(title_text="Value", row=1, col=2)

fig.show()
