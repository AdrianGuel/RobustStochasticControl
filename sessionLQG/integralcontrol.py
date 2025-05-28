import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
import plotly.graph_objects as go

# System matrices
A = np.array([[0, 1],
              [-10, -1]])
B = np.array([[0],
              [1]])
C = np.array([[1, 0]])

# Augment system with integrator
A_aug = np.block([
    [A, np.zeros((2, 1))],
    [-C, np.zeros((1, 1))]
])
B_aug = np.vstack([B, np.zeros((1, 1))])
# E_aug = np.vstack([np.zeros((2, 1)), [[1]]])  # for reference input r(t)

# LQR cost matrices
Q = np.diag([0, 0, 1])  # penalize only integral error
R = np.array([[0.0001]])  # small penalty on control

# Solve Riccati equation
P = solve_continuous_are(A_aug, B_aug, Q, R)
K = np.linalg.inv(R) @ B_aug.T @ P
print(K)
Kx = K[0, :2]
Ki = K[0, 2]

# Reference
r = 1.0

# Closed-loop dynamics
def closed_loop(t, z):
    x = z[:2]
    x_i = z[2]
    u = -Kx @ x - Ki * x_i
    dxdt = A @ x + B.flatten() * u
    dxidt = r - C @ x
    return np.concatenate([dxdt, dxidt])

# Time span
t_span = (0, 3)
t_eval = np.linspace(*t_span, 1000)
z0 = np.zeros(3)

# Integrate system
sol = solve_ivp(closed_loop, t_span, z0, t_eval=t_eval)
y = sol.y[0]
u_values = np.array([-Kx @ sol.y[:2, i] - Ki * sol.y[2, i] for i in range(sol.y.shape[1])])

# Plotly plots
fig = go.Figure()
fig.add_trace(go.Scatter(x=sol.t, y=y, name="y(t)", line=dict(color="blue")))
fig.add_trace(go.Scatter(x=sol.t, y=[r]*len(sol.t), name="Reference", line=dict(dash="dash")))
fig.update_layout(title="Integral Control Tracking",
                  xaxis_title="Time (s)",
                  yaxis_title="Output y(t)")
fig.show()

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=sol.t, y=u_values, name="u(t)", line=dict(color="red")))
fig2.update_layout(title="Control Input",
                   xaxis_title="Time (s)",
                   yaxis_title="Control Torque")
fig2.show()
