"""
Bellman Optimality Simulation and Visualization
Author: Adrian Guel

Description:
This script demonstrates the solution to a finite-horizon discrete-time 
Linear Quadratic Regulator (LQR) problem using the Bellman optimality principle. 
It solves the Riccati recursion backward in time to compute the optimal gain matrix K,
then simulates the state evolution forward in time using the optimal policy.

The results — system state trajectories, feedback gains, and Riccati matrix elements —
are visualized using interactive subplots in Plotly.

This code is intended for teaching and demonstration purposes in optimal control.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------------------------------
# System Definition and LQR Setup
# -------------------------------

# System dynamics matrices
A = np.array([[2, 1],     # State transition matrix
              [-1, 1]])
B = np.array([[0],        # Input matrix
              [1]])

# Initial condition
x0 = np.array([[2], [-3]])

# Time horizon
t = np.arange(11)

# Cost function weights
Q = np.diag([2, 0.1])     # State cost matrix
R = np.array([[2]])       # Control cost (as 1x1 matrix for @ compatibility)
P = np.zeros((2, 2, 11))  # Riccati matrix P_k for k = 0 to 10
K = np.zeros((1, 2, 9))   # Optimal feedback gain K_k for k = 0 to 8
x = np.zeros((2, 1, 11))  # State trajectory for k = 0 to 10

# Set terminal cost
P[:, :, 10] = np.diag([5, 5])
x[:, :, 0] = x0

# -------------------------------
# Backward Riccati Recursion
# -------------------------------

for i in reversed(range(9)):
    # Compute gain matrix K_k = (R + Bᵀ P_{k+1} B)⁻¹ Bᵀ P_{k+1} A
    BT_PB = B.T @ P[:, :, i + 1] @ B
    K[:, :, i] = np.linalg.inv(R + BT_PB) @ B.T @ P[:, :, i + 1] @ A
    
    # Closed-loop system matrix
    A_BK = A - B @ K[:, :, i]
    
    # Riccati update: P_k = A_clᵀ P_{k+1} A_cl + Q + Kᵀ R K
    P[:, :, i] = A_BK.T @ P[:, :, i + 1] @ A_BK + Q + K[:, :, i].T @ R @ K[:, :, i]

# -------------------------------
# Forward Simulation of Dynamics
# -------------------------------

for i in range(9):
    # Apply control policy: x_{k+1} = A x_k - B K_k x_k
    x[:, :, i + 1] = A @ x[:, :, i] - B @ K[:, :, i] @ x[:, :, i]

# -------------------------------
# Plotting with Plotly
# -------------------------------

# Create 3 subplots: states, gains, and Riccati elements
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                    subplot_titles=("States over Time", 
                                    "Feedback Gains", 
                                    "Riccati Matrix Elements"))

# --- Subplot 1: State Trajectories ---
fig.add_trace(go.Scatter(x=t, y=np.squeeze(x[0, 0, :]),
                         mode='lines+markers', name='x1'), row=1, col=1)
fig.add_trace(go.Scatter(x=t, y=np.squeeze(x[1, 0, :]),
                         mode='lines+markers', name='x2'), row=1, col=1)

# --- Subplot 2: Feedback Gains ---
fig.add_trace(go.Scatter(x=t[:-2], y=np.squeeze(K[0, 0, :]),
                         mode='lines+markers', name='K1'), row=2, col=1)
fig.add_trace(go.Scatter(x=t[:-2], y=np.squeeze(K[0, 1, :]),
                         mode='lines+markers', name='K2'), row=2, col=1)

# --- Subplot 3: Riccati Matrix Elements ---
fig.add_trace(go.Scatter(x=t[:-1], y=[P[0, 0, i] for i in range(10)],
                         mode='lines+markers', name='P11'), row=3, col=1)
fig.add_trace(go.Scatter(x=t[:-1], y=[P[1, 1, i] for i in range(10)],
                         mode='lines+markers', name='P22'), row=3, col=1)
fig.add_trace(go.Scatter(x=t[:-1], y=[P[0, 1, i] for i in range(10)],
                         mode='lines+markers', name='P12'), row=3, col=1)

# Final figure layout adjustments
fig.update_layout(height=900, width=800, 
                  title_text="Bellman Optimality: State, Gain, and Riccati Evolution",
                  showlegend=True)
fig.update_xaxes(title_text="Time", row=3, col=1)
fig.show()
