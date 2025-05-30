## MinMax problem with a second order time delay system 
## System: G(s)= exp(-Ls)/(s(ms+b))
## Controller: C(s) = Kp + Kds
## Author: Adrian Guel 2025

import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize
import control as ctl
from typing import Tuple, List

# === System Parameters ===
m: float = 1.0  # Mass (kg)
b: float = 0.5  # Damping coefficient
T_final: float = 10.0  # Total simulation time (s)
dt: float = 0.01  # Sampling time (s)
time: np.ndarray = np.arange(0, T_final, dt)  # Time vector
reference: np.ndarray = np.ones_like(time)  # Step reference input

# === PD-Controlled System Simulation with Delay ===
def simulate_with_control(
    Kp: float,
    Kd: float,
    L: float,
    m: float,
    b: float,
    t: np.ndarray,
    r: np.ndarray,
    pade_order: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate the closed-loop response of a delayed second-order system with PD control.

    Parameters:
        Kp: Proportional gain
        Kd: Derivative gain
        L: Time delay (s)
        m: Mass of the system
        b: Damping coefficient
        t: Time vector
        r: Reference signal
        pade_order: Order of the Pade approximation for delay

    Returns:
        y: Output signal
        e: Tracking error signal (r - y)
    """
    # Plant transfer function G(s) = 1 / (m s^2 + b s)
    num = [1.0]
    den = [m, b, 0]
    G = ctl.TransferFunction(num, den)

    # PD controller: C(s) = Kd * s + Kp
    C = ctl.TransferFunction([Kd, Kp], [1])

    # Time delay approximation using Pade
    if L > 0:
        num_delay, den_delay = ctl.pade(L, pade_order)
        D = ctl.TransferFunction(num_delay, den_delay)
        G_delayed = D * G
    else:
        G_delayed = G

    # Closed-loop transfer function with unity feedback
    T_cl = ctl.feedback(C * G_delayed, 1)

    # Simulate the time response
    _, y = ctl.forced_response(T_cl, T=t, U=r)
    e = r - y
    return y, e

# === Objective Function for PD Tuning ===
def cost_function(
    params: np.ndarray,
    L: float,
    m: float,
    b: float,
    t: np.ndarray,
    r: np.ndarray
) -> float:
    """
    Cost function to minimize: mean squared tracking error.

    Parameters:
        params: Array with [Kp, Kd]
        L: Time delay
        m, b: System parameters
        t, r: Time vector and reference input

    Returns:
        Mean squared error between reference and output
    """
    Kp, Kd = params
    _, e = simulate_with_control(Kp, Kd, L, m, b, t, r)
    return np.mean(e**2)

# === Inner Optimization: Best PD Gains for a Fixed Delay ===
def optimize_gains_for_delay(
    L: float,
    m: float,
    b: float,
    t: np.ndarray,
    r: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Find the optimal PD gains that minimize error for a given delay L.

    Returns:
        (minimum error, [Kp_opt, Kd_opt])
    """
    res = minimize(cost_function, [1.0, 1.0], args=(L, m, b, t, r),
                   bounds=[(0, 100), (0, 100)], method='L-BFGS-B')
    return res.fun, res.x

# === Outer Loop: Find Maximum Admissible Delay ===
L_vals: np.ndarray = np.linspace(0.0, 2.0, 20)  # Delay values to test
errors: List[float] = []
optimal_gains: List[np.ndarray] = []

for L in L_vals:
    err, gains = optimize_gains_for_delay(L, m, b, time, reference)
    errors.append(err)
    optimal_gains.append(gains)

# Determine the maximum L that keeps error below threshold
threshold: float = 0.2
valid_Ls = [L for L, err in zip(L_vals, errors) if err < threshold]
max_L: float = max(valid_Ls) if valid_Ls else None

# === Plotting MSE vs Delay ===
fig = go.Figure()
fig.add_trace(go.Scatter(x=L_vals, y=errors, mode='lines+markers',
                         name='MSE vs Delay L'))
fig.add_hline(y=threshold, line_dash="dash", annotation_text="Error Threshold")
fig.update_layout(title='Error vs Delay under Optimal PD Gains',
                  xaxis_title='Delay L (s)', yaxis_title='Mean Squared Error')
fig.show()

# === Report Results and Plot Time Response ===
if max_L is not None:
    print(f"Max admissible delay L: {max_L:.3f} s")

    # Retrieve corresponding optimal gains
    idx = np.argmax(np.array(L_vals) == max_L)
    Kp_opt, Kd_opt = optimal_gains[idx]

    # Simulate final time response with optimal gains
    y_opt, e_opt = simulate_with_control(Kp_opt, Kd_opt, max_L, m, b, time, reference)

    # Plot the time response
    fig_response = go.Figure()
    fig_response.add_trace(go.Scatter(x=time, y=reference, mode='lines', name='Reference'))
    fig_response.add_trace(go.Scatter(x=time, y=y_opt, mode='lines', name='Output y(t)'))
    fig_response.add_trace(go.Scatter(x=time, y=e_opt, mode='lines', name='Tracking Error', line=dict(dash='dot')))
    fig_response.update_layout(
        title=f'Time Response with Optimal PD Gains at L = {max_L:.3f}s',
        xaxis_title='Time (s)',
        yaxis_title='Amplitude',
        legend=dict(x=0.01, y=0.99)
    )
    fig_response.show()
else:
    print("No admissible delay found under error threshold.")
