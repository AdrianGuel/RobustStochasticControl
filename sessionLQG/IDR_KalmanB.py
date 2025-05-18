import numpy as np
import plotly.graph_objects as go

# Time parameters
dt = 0.01
T = 20
steps = int(T / dt)
t = np.linspace(0, T, steps)
r = 1.0  # reference

# Noise parameters
sigma_w = 0.01  # process noise std
sigma_v = 0.01  # measurement noise std
np.random.seed(0)

Q_kf = sigma_w**2 * np.eye(3)  # process noise covariance
R_kf = sigma_v**2              # measurement noise covariance (scalar)

# LQR controller design (Page 23)
from scipy.linalg import solve_continuous_are

A_lqr = np.array([
    [-1.0, 1.0, 0.0],
    [0.0, -0.1, 0.0],
    [-1.0, 0.0, 0.0]
])
B_lqr = np.array([[0.0], [0.1], [0.0]])
Q = np.diag([0.0, 0.0, 1.0])
R = np.array([[0.0001]])

P_lqr = solve_continuous_are(A_lqr, B_lqr, Q, R)
K = np.linalg.inv(R) @ B_lqr.T @ P_lqr

# System model
A = np.array([
    [-1.0, 1.0, 1.0],
    [0.0, -0.1, 0.0],
    [0.0, 0.0, 0.0]
])
B = np.array([[0.0], [0.1], [0.0]])
C = np.array([[1.0, 0.0, 0.0]])

# Initialization
x_hat = np.zeros((3,))  # [omega, tauM, tauL]
xI = 0.0
P_k = np.eye(3)         # Initial error covariance

omega_hist, u_hist, tauL_hist = [], [], []

for k in range(steps):
    # Measurement (simulate perfect output from estimator state)
    y = C @ x_hat + np.random.normal(0, sigma_v)

    # Control input
    u = -K[0, 0] * x_hat[0] - K[0, 1] * x_hat[1] - K[0, 2] * xI

    # ----------- Kalman Filter Prediction ------------
    x_pred = x_hat + dt * (A @ x_hat + B.flatten() * u)
    P_pred = A @ P_k @ A.T * dt**2 + Q_kf  # Euler-discretized process

    # ----------- Kalman Filter Update ----------------
    S = C @ P_pred @ C.T + R_kf
    L_k = P_pred @ C.T / S
    y_pred = C @ x_pred
    x_hat = x_pred + L_k.flatten() * (y - y_pred)
    P_k = (np.eye(3) - L_k @ C) @ P_pred

    # ----------- Integrator --------------------------
    dxI = r - y
    xI += dxI * dt

    # Log
    omega_hist.append(x_hat[0])
    u_hist.append(u.item())
    tauL_hist.append(x_hat[2])

# Plot omega
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=t, y=omega_hist, name="Estimated ω(t)", line=dict(color="blue")))
fig1.add_trace(go.Scatter(x=t, y=[r]*len(t), name="Reference", line=dict(dash="dash")))
fig1.update_layout(title="Angular Velocity Tracking with Dynamic Kalman Gain",
                   xaxis_title="Time (s)", yaxis_title="ω(t) [rad/s]")
fig1.show()

# Plot control input
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=t, y=u_hist, name="Control u(t)", line=dict(color="red")))
fig2.update_layout(title="Control Input u(t)", xaxis_title="Time (s)", yaxis_title="Field Voltage (V)")
fig2.show()

# Plot estimated disturbance
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=t, y=tauL_hist, name="Estimated τ_L(t)", line=dict(color="green")))
fig3.update_layout(title="Estimated Load Torque Disturbance", xaxis_title="Time (s)", yaxis_title="τ_L(t)")
fig3.show()
