import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Define parameters
A = np.array([[2, 1], [-1, 1]])
B = np.array([[0], [1]])
x0 = np.array([[2], [-3]])

P = np.zeros((2, 2, 11))  # P(:,:,10) is final condition
K = np.zeros((1, 2, 9))
x = np.zeros((2, 1, 11))
x[:, :, 0] = x0

P[:, :, 10] = np.diag([5, 5])
R = np.array([[2]])
Q = np.diag([2, 0.1])
t = np.arange(11)

# Backward Riccati recursion
for i in reversed(range(9)):
    BT_PB = B.T @ P[:, :, i + 1] @ B
    K[:, :, i] = np.linalg.inv(R + BT_PB) @ B.T @ P[:, :, i + 1] @ A
    A_BK = A - B @ K[:, :, i]
    P[:, :, i] = A_BK.T @ P[:, :, i + 1] @ A_BK + Q + K[:, :, i].T @ R @ K[:, :, i]

# Forward simulate system
for i in range(9):
    x[:, :, i + 1] = A @ x[:, :, i] - B @ K[:, :, i] @ x[:, :, i]

# Create subplots layout
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                    subplot_titles=("States over Time", 
                                    "Feedback Gains", 
                                    "Riccati Matrix Elements"))

# --- Subplot 1: State trajectories ---
fig.add_trace(go.Scatter(x=t, y=np.squeeze(x[0, 0, :]), 
                         mode='lines+markers', name='x1'), row=1, col=1)
fig.add_trace(go.Scatter(x=t, y=np.squeeze(x[1, 0, :]), 
                         mode='lines+markers', name='x2'), row=1, col=1)

# --- Subplot 2: Gains ---
fig.add_trace(go.Scatter(x=t[:-2], y=np.squeeze(K[0, 0, :]), 
                         mode='lines+markers', name='K1'), row=2, col=1)
fig.add_trace(go.Scatter(x=t[:-2], y=np.squeeze(K[0, 1, :]), 
                         mode='lines+markers', name='K2'), row=2, col=1)

# --- Subplot 3: Riccati matrix entries ---
fig.add_trace(go.Scatter(x=t[:-1], y=[P[0, 0, i] for i in range(10)], 
                         mode='lines+markers', name='P11'), row=3, col=1)
fig.add_trace(go.Scatter(x=t[:-1], y=[P[1, 1, i] for i in range(10)], 
                         mode='lines+markers', name='P22'), row=3, col=1)
fig.add_trace(go.Scatter(x=t[:-1], y=[P[0, 1, i] for i in range(10)], 
                         mode='lines+markers', name='P12'), row=3, col=1)

# Final layout adjustments
fig.update_layout(height=900, width=800, 
                  title_text="Bellman Optimality: State, Gain, and Riccati Evolution",
                  showlegend=True)
fig.update_xaxes(title_text="Time", row=3, col=1)
fig.show()