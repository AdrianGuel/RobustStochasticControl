import numpy as np
import plotly.graph_objects as go
from scipy.linalg import solve_continuous_are
from numpy.random import default_rng

# ----------------------------------------
# Simulation parameters
T = 100.0
dt = 0.1
N = int(T / dt)
time = np.linspace(0, T, N)

# ----------------------------------------
# System matrices
A = np.array([[0, 1], [0, -0.1]])
B = np.array([[0], [0.001]])
C = np.array([[1, 0]])
Bw = np.array([[0], [0.001]])

# ----------------------------------------
# Noise covariances (continuous-time)
Sw = 5e3#5000
Sv = 100
Qw = Bw @ Bw.T * Sw  # Process noise covariance
Rv = np.array([[Sv]])  # Measurement noise covariance

# ----------------------------------------
# Cost function for LQR
q = 18e1
r = 1
Q = np.array([[q, 0], [0, 0]])
R = np.array([[r]])

# ----------------------------------------
# Initial conditions
x = np.zeros((2, N))
x_hat = np.zeros((2, N))
u = np.zeros(N)
y = np.zeros(N)
P = np.zeros((2, 2, N))  # Error covariance matrix
P[:, :, 0] = np.eye(2)   # Start with identity

rng = default_rng(42)

# ----------------------------------------
# LQR gain (fixed)
P_inf = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P_inf

# Logging arrays
K1_log = np.full(N-1, K[0, 0])
K2_log = np.full(N-1, K[0, 1])
L1_log = np.zeros(N-1)
L2_log = np.zeros(N-1)

# ----------------------------------------
# Simulation loop (Kalman filter gain updated at each step)
for k in range(N - 1):
    # Noise samples
    w = rng.normal(0, np.sqrt(Sw / dt))
    v = rng.normal(0, np.sqrt(Sv / dt))

    # Plant dynamics
    x[:, k + 1] = x[:, k] + dt * (A @ x[:, k] + B.flatten() * u[k] + Bw.flatten() * w)
    y[k] = C @ x[:, k] + v

    # Kalman gain computation (discrete-time update)
    P_k = P[:, :, k]
    S_k = C @ P_k @ C.T + Rv
    L_k = P_k @ C.T @ np.linalg.inv(S_k)

    # State estimate update
    y_tilde = y[k] - C @ x_hat[:, k]
    x_hat[:, k + 1] = x_hat[:, k] + dt * (A @ x_hat[:, k] + B.flatten() * u[k] + L_k @ y_tilde)

    # Covariance update (continuous-time Riccati integration step)
    P_dot = A @ P_k + P_k @ A.T + Qw - L_k @ S_k @ L_k.T
    P[:, :, k + 1] = P_k + dt * P_dot

    # Save Kalman gain values
    L1_log[k] = L_k[0, 0]
    L2_log[k] = L_k[1, 0]

    # Control law
    u[k + 1] = -K @ x_hat[:, k + 1]

# ----------------------------------------
# Plot 1: Feedback Gains
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=time[:-1], y=K1_log, name="K1"))
fig1.add_trace(go.Scatter(x=time[:-1], y=K2_log, name="K2"))
fig1.update_layout(title="Feedback Gains", xaxis_title="Time (s)", yaxis_title="Gain")

# Plot 2: Kalman Gains (time-varying)
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=time[:-1], y=L1_log, name="L1"))
fig2.add_trace(go.Scatter(x=time[:-1], y=L2_log, name="L2"))
fig2.update_layout(title="Kalman Gains", xaxis_title="Time (s)", yaxis_title="Gain")

# Plot 3: Angle Error
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=time[:-1], y=x[0, :-1], name="Actual"))
fig3.add_trace(go.Scatter(x=time[:-1], y=x_hat[0, :-1], name="Estimated"))
fig3.update_layout(title="Angle Error (deg)", xaxis_title="Time (sec)", yaxis_title="Angle (deg)")

# Plot 4: Control Input
fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=time[:-1], y=u[:-1], name="Control Torque"))
fig4.update_layout(title="Control Input (N-m)", xaxis_title="Time (s)", yaxis_title="Torque")

# Show plots
fig1.show()
fig2.show()
fig3.show()
fig4.show()
