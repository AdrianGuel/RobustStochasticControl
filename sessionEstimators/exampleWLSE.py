import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from numpy.linalg import inv
import pandas as pd

# Measurement times and values
t2 = np.array([0, 0.5, 1, 2])
Y2 = np.array([6, 5, 0, 0]).reshape(-1, 1)
H2 = np.vstack([np.ones_like(t2), t2, t2**2]).T

# Weight matrix: emphasize y(0.5)
W = np.diag([0.05, 0.8, 0.05, 0.1])

# Weighted least squares estimate
X_wlse = inv(H2.T @ W @ H2) @ H2.T @ W @ Y2

# Prediction and error
t_vals = np.linspace(0, 4, 100)
y_fit_wlse = X_wlse[0] + X_wlse[1] * t_vals + X_wlse[2] * t_vals**2
error_wlse = Y2 - H2 @ X_wlse

# Display results
print("Weighted LS Estimate:\n", X_wlse)
print("Error vector:\n", error_wlse.flatten())

# Plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=t_vals, y=y_fit_wlse.flatten(), mode='lines', name='Weighted LS Fit', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=t2, y=Y2.flatten(), mode='markers', name='Data Points', marker=dict(color='blue')))
fig.update_layout(
    title_text=f"y = {X_wlse[0,0]:.2f} {X_wlse[1,0]:+.2f}t {X_wlse[2,0]:+.2f}t² (Weighted LS)",
    xaxis_title="Time",
    yaxis_title="Value"
)
fig.show()
