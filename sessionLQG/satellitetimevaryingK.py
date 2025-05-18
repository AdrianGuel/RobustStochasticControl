import numpy as np
import plotly.graph_objects as go
from scipy.integrate import solve_ivp
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
# Noise covariances
Sw = 5000
Sv = 1
Qw = Bw @ Bw.T * Sw
Rv = np.array([[Sv]])

# ----------------------------------------
# LQR cost weights
q = 180
r = 1
Q = np.array([[q, 0], [0, 0]])
R = np.array([[r]])
H = np.zeros((2, 2))  # Terminal cost

# ----------------------------------------
# Riccati differential equation
def riccati_ode(t, P_flat, A, B, Q, R):
    P = P_flat.reshape(2, 2)
    dPdt = -(A.T @ P + P @ A - P @ B @ np.linalg.inv(R) @ B.T @ P + Q)
    return dPdt.flatten()

# Solve DRE backward in time
t_span = [T, 0]
t_eval = np.linspace(T, 0, N)
sol = solve_ivp(riccati_ode, t_span, H.flatten(), t_eval=t_eval, args=(A, B, Q, R))

# Reformat and reverse P_t to match simulation forward in time
P_t = sol.y.T.reshape(-1, 2, 2)[::-1]
K_t = np.array([np.linalg.inv(R) @ B.T @ P for P in P_t])  # shape (N, 1, 2)

# ----------------------------------------
# Initializations
x = np.zeros((2, N))
x_hat = np.zeros((2, N))
u = np.zeros(N)
y = np.zeros(N)
P = np.zeros((2, 2, N))
P[:, :, 0] = np.eye(2)
rng = default_rng(42)

# Logging arrays
K1_log = np.zeros(N - 1)
K2_log = np.zeros(N - 1)
L1_log = np.zeros(N - 1)
L2_log = np.zeros(N - 1)

# ----------------------------------------
# Simulation loop
for k in range(N - 1):
    w = rng.normal(0, np.sqrt(Sw / dt))
    v = rng.normal(0, np.sqrt(Sv / dt))

    # Plant dynamics
    x[:, k + 1] = x[:, k] + dt * (A @ x[:, k] + B.flatten() * u[k] + Bw.flatten() * w)
    y[k] = C @ x[:, k] + v

    # Kalman gain
    P_k = P[:, :, k]
    S_k = C @ P_k @ C.T + Rv
    L_k = P_k @ C.T @ np.linalg.inv(S_k)

    # Estimate state
    y_tilde = y[k] - C @ x_hat[:, k]
    x_hat[:, k + 1] = x_hat[:, k] + dt * (A @ x_hat[:, k] + B.flatten() * u[k] + L_k @ y_tilde)

    # Update covariance
    P_dot = A @ P_k + P_k @ A.T + Qw - L_k @ S_k @ L_k.T
    P[:, :, k + 1] = P_k + dt * P_dot

    # Time-varying LQR gain
    K_k = K_t[k]  # shape (1, 2)
    u[k + 1] = -K_k @ x_hat[:, k + 1]

    # Logging
    K1_log[k] = K_k[0, 0]
    K2_log[k] = K_k[0, 1]
    L1_log[k] = L_k[0, 0]
    L2_log[k] = L_k[1, 0]

# ----------------------------------------
# Plot 1: Feedback Gains
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=time[:-1], y=K1_log, name="K1"))
fig1.add_trace(go.Scatter(x=time[:-1], y=K2_log, name="K2"))
fig1.update_layout(title="Time-Varying Feedback Gains", xaxis_title="Time (s)", yaxis_title="Gain")

# Plot 2: Kalman Gains
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
