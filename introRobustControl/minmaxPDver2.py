from scipy.optimize import minimize, Bounds, NonlinearConstraint
import numpy as np
import plotly.graph_objects as go
import control as ctl
from typing import Tuple

# === System Parameters ===
m: float = 1.0
b: float = 0.5
T_final: float = 10.0
dt: float = 0.01
time: np.ndarray = np.arange(0, T_final, dt)
reference: np.ndarray = np.ones_like(time)

def simulate_with_control(Kp, Kd, L, m, b, t, r, pade_order=3) -> Tuple[np.ndarray, np.ndarray]:
    num = [1.0]
    den = [m, b, 0]
    G = ctl.TransferFunction(num, den)
    C = ctl.TransferFunction([Kd, Kp], [1])
    if L > 0:
        num_d, den_d = ctl.pade(L, pade_order)
        D = ctl.TransferFunction(num_d, den_d)
        G_delayed = D * G
    else:
        G_delayed = G
    T_cl = ctl.feedback(C * G_delayed, 1)
    _, y = ctl.forced_response(T_cl, T=t, U=r)
    e = r - y
    return y, e

def combined_objective(params):
    L, Kp, Kd = params
    return -L

def constraint_error(params, m, b, t, r, epsilon):
    L, Kp, Kd = params
    _, e = simulate_with_control(Kp, Kd, L, m, b, t, r)
    return epsilon - np.mean(e**2)

# === Main Optimization ===
bounds = Bounds([0.0, 0.0, 0.0], [2.0, 100.0, 100.0])
nonlin_constr = NonlinearConstraint(lambda p: constraint_error(p, m, b, time, reference, 0.2), 0, np.inf)

res = minimize(combined_objective, [0.1, 1.0, 1.0], method='SLSQP', bounds=bounds, constraints=[nonlin_constr])
L_opt, Kp_opt, Kd_opt = res.x
print(f"Optimal delay L = {L_opt:.3f}, Kp = {Kp_opt:.3f}, Kd = {Kd_opt:.3f}")

# === Plot Time Response ===
y_opt, _ = simulate_with_control(Kp_opt, Kd_opt, L_opt, m, b, time, reference)
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=time, y=reference, mode='lines', name='Reference'))
fig1.add_trace(go.Scatter(x=time, y=y_opt, mode='lines', name='System Response'))
fig1.update_layout(
    title=f"Time Response (L={L_opt:.3f}, Kp={Kp_opt:.3f}, Kd={Kd_opt:.3f})",
    xaxis_title="Time [s]",
    yaxis_title="Output",
    template="plotly_white"
)

## Uncomment to see Cost function (computationally expensive!!)
# # === 3D Constrained Cost Surface ===
# Kp_vals = np.linspace(1, 50, 20)
# Kd_vals = np.linspace(1, 50, 20)
# Kp_grid, Kd_grid = np.meshgrid(Kp_vals, Kd_vals)
# L_grid = np.zeros_like(Kp_grid)

# for i in range(Kp_grid.shape[0]):
#     for j in range(Kp_grid.shape[1]):
#         Kp_test = Kp_grid[i, j]
#         Kd_test = Kd_grid[i, j]
#         try:
#             bounds_2d = Bounds([0.0], [2.0])
#             cons_2d = NonlinearConstraint(
#                 lambda L: constraint_error([L[0], Kp_test, Kd_test], m, b, time, reference, 0.2), 0, np.inf)
#             result = minimize(lambda L: -L[0], x0=[0.1], bounds=bounds_2d, constraints=[cons_2d])
#             if result.success:
#                 L_grid[i, j] = result.x[0]
#             else:
#                 L_grid[i, j] = np.nan
#         except:
#             L_grid[i, j] = np.nan

# fig2 = go.Figure(data=[
#     go.Surface(z=-L_grid, x=Kp_grid, y=Kd_grid, colorscale='Viridis')
# ])
# fig2.update_layout(
#     title="Constrained Cost Surface: -L (Only feasible regions shown)",
#     scene=dict(
#         xaxis_title='Kp',
#         yaxis_title='Kd',
#         zaxis_title='-L (Cost)'
#     ),
#     template="plotly_white"
# )

# === Show Plots ===
fig1.show()
# fig2.show()
