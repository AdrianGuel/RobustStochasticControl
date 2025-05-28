## Final example
## Cart system with LQG + integral control

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
import plotly.graph_objects as go
from numpy.random import default_rng

# Time parameters
dt = 0.01
T = 100
steps = int(T / dt)
t = np.linspace(0, T, steps)

#System parameters
b=0.2
m=1.0
A = np.array([[0.0,1.0],[0.0,-b/m]])
B = np.array([[0.0],[1/m]])
C = np.array([[1, 0]])

# Augment system with integrator
A_aug = np.block([
    [A, np.zeros((2, 1))],
    [-C, np.zeros((1, 1))]
])
B_aug = np.vstack([B, np.zeros((1, 1))])
E_aug = np.vstack([np.zeros((2, 1)), [[1]]])  # for reference input r(t)
C_aug = np.hstack([C, np.zeros((1, 1))])
print(C_aug)
#Optimal control
Q = np.diag([0.0, 0.0, 1.0])
R = np.array([[0.0001]])

# Solve Riccati equation
P = solve_continuous_are(A_aug, B_aug, Q, R)
K = np.linalg.inv(R) @ B_aug.T @ P
Kx = K[0, :2]
Ki = K[0, 2]

# Reference
r = np.ones(steps)#np.sin(0.1*t)
#states
x = np.zeros([3,steps])
y = np.zeros([steps])
u = np.zeros([steps])
rng = default_rng(42)

##Kalman filter parameters
# Noise covariances (continuous-time)
Sw = 0
Sv = 5e-5
Bw = np.array([[1.0],[1.0],[0.0]])
Qw = Bw @ Bw.T * Sw  # Process noise covariance
Rv = np.array([[Sv]])  # Measurement noise covariance
Pminus = np.eye(3)
Pplus = np.eye(3)
x_hat = np.zeros([3,steps])
x_pred = np.zeros([3,steps])
Ad = np.eye(3)+dt*A_aug # Euler method
Bd = dt*B_aug
Ed = dt*E_aug

for k in range(steps-1):
    # Noise samples
    w = rng.normal(0, np.sqrt(Sw / dt), size=(3,))
    v = rng.normal(0, np.sqrt(Sv / dt))

    # Control effort
    u[k] = -Kx @ x_hat[:2,k]- Ki*x_hat[2,k]
    # Real system (sol using Euler method):
    x[:,k+1] = x[:,k] + dt*(A_aug @ x[:,k] + B_aug.flatten() * u[k]+E_aug.flatten() * r[k])+ w
    y[k] = C_aug @ x[:,k+1] + v

    #Propagate/Update/Prediction model
    x_pred[:,k] = Ad @ x_hat[:, k] + Bd.flatten() * u[k]+Ed.flatten() * r[k]
    Pminus = Ad@Pplus@Ad.T + Qw

    # Kalma gain computation
    S_k = C_aug @ Pminus @ C_aug.T + Rv
    L_k = Pminus @ C_aug.T @ np.linalg.inv(S_k)

    # Update estimate with measurement
    y_tilde = y[k] - C_aug @ x_pred[:, k]
    x_hat[:, k+1] = x_pred[:, k] + L_k @ y_tilde

    #Update covariance
    Pplus = (np.eye(3)-L_k@C_aug)@Pminus 

# Plotly plots
fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=y, name="y(t)", line=dict(color="blue")))
fig.add_trace(go.Scatter(x=t, y=x_hat[0,:], name="x_est(t)", line=dict(color="black")))
fig.add_trace(go.Scatter(x=t, y=r, name="Reference", line=dict(dash="dash")))
fig.update_layout(title="Integral Control Tracking",
                  xaxis_title="Time (s)",
                  yaxis_title="Output")
fig.show()

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=t, y=u, name="u(t)", line=dict(color="red")))
fig2.update_layout(title="Control Input",
                   xaxis_title="Time (s)",
                   yaxis_title="Control Torque")
fig2.show()
