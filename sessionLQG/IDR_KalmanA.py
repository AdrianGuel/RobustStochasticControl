import numpy as np
import plotly.graph_objects as go
from scipy.linalg import solve_continuous_are
# Time parameters
dt = 0.01
T = 100
steps = int(T / dt)
t = np.linspace(0, T, steps)
r = np.sin(0.1*t)  # reference

# Noise parameters
sigma_T = 0.1  # std dev of omega_T
sigma_v = 0.01  # measurement noise std

np.random.seed(0)

# LQR controller design considering integral control
A_lqr = np.array([
    [-1.0, 1.0, 0.0],
    [0.0, -0.1, 0.0],
    [-1.0, 0.0, 0.0]
])
B_lqr = np.array([[0.0], [0.1], [0.0]])
Q = np.diag([0.0, 0.0, 10.0])
R = np.array([[0.00001]])

# Solve CARE
P = solve_continuous_are(A_lqr, B_lqr, Q, R)
K = np.linalg.inv(R) @ B_lqr.T @ P

# Estimator design (Page 24)
A_est = np.array([
    [-1.0, 1.0, 1.0],
    [0.0, -0.1, 0.0],
    [0.0, 0.0, 0.0]
])
B_est = np.array([
    [0.0],
    [0.1],
    [0.0]
])
C = np.array([[1.0, 0.0, 0.0]])
L = np.array([[1.8], [0.1], [3.2]])

# Initial states
x_hat = np.zeros((3,))  # estimator states: [omega, tauM, tauL]
xI = 0.0                # integrator state
y = 0.0
u = 0.0

# Logging
omega_hist, u_hist, tauL_hist = [], [], []

# Simulate dynamics using Euler method
for k in range(steps):
    # Measurement
    y = C @ x_hat + np.random.normal(0, sigma_v)

    # Control law
    u = -K[0, 0] * x_hat[0] - K[0, 1] * x_hat[1] - K[0, 2] * xI

    # Disturbance: torque perturbation (white noise)
    omega_T = np.random.normal(0, sigma_T)
    
    # Estimator dynamics with noise-driven tau_L
    dx_hat = (A_est - L @ C) @ x_hat + B_est.flatten() * u + L.flatten() * y
    dx_hat[2] += omega_T  # only tau_L is driven by omega_T

    # Integrator update
    dxI = r[k] - y

    # Euler update
    x_hat += dx_hat * dt
    xI += dxI * dt

    # Log
    omega_hist.append(x_hat[0])
    u_hist.append(float(u))
    tauL_hist.append(x_hat[2])

print(u_hist)
# Plot omega
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=t, y=omega_hist, name="Estimated ω(t)", line=dict(color="blue")))
fig1.add_trace(go.Scatter(x=t, y=r, name="Reference", line=dict(dash="dash")))
fig1.update_layout(title="Angular Velocity Tracking with Disturbance",
                   xaxis_title="Time (s)", yaxis_title="ω(t) [rad/s]")
fig1.show()

# Plot control input
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=t, y=u_hist, name="Control u(t)", line=dict(color="red")))
fig2.update_layout(title="Control Input u(t)", xaxis_title="Time (s)", yaxis_title="Field Voltage (V)")
fig2.show()

# Plot disturbance estimate
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=t, y=tauL_hist, name="Estimated τ_L(t)", line=dict(color="green")))
fig3.update_layout(title="Estimated Load Torque Disturbance", xaxis_title="Time (s)", yaxis_title="τ_L(t)")
fig3.show()
