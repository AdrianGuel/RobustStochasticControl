import numpy as np
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px

def discrete_aircraft_model_3d(dt=0.1):
    Ad = np.eye(6)
    for i in range(0,6,2):
        Ad[i,i+1]= dt

    Hd= np.zeros((3,6))
    Hd[0,0], Hd[1,2],Hd[2,4]=1,1,1
    return Ad, Hd 

def simulate_aircraft_trajectory(A, H, Q, R, x0, steps):
    """Simulate true aircraft motion and radar measurements."""
    n, m = A.shape[0], H.shape[0]
    x_true = np.zeros((n, steps))
    y_meas = np.zeros((m, steps))
    x = x0.copy()

    for k in range(steps):
        x_true[:, k] = x
        y_meas[:, k] = H @ x + np.random.multivariate_normal(np.zeros(m), R)
        x = A @ x + np.random.multivariate_normal(np.zeros(n), Q)

    return x_true, y_meas

def kalman_filter_3d(y_meas, A, H, Q, R, x0, P0):
    """Estimate states using a linear Kalman filter."""
    m, N = y_meas.shape
    n = A.shape[0]
    x_est = np.zeros((n, N))
    P_est = np.zeros((n, n, N))

    x, P = x0.copy(), P0.copy()

    for k in range(N):
        x_pred = A @ x
        P_pred = A @ P @ A.T + Q

        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        x = x_pred + K @ (y_meas[:, k] - H @ x_pred)
        P = (np.eye(n) - K @ H) @ P_pred

        x_est[:, k], P_est[:, :, k] = x, P

    return x_est, P_est

def plot_3d_trajectory(x_true, y_meas, x_est):
    """Plot 3D aircraft trajectories (true, measured, estimated)."""
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=x_true[0], y=x_true[2], z=x_true[4],
        mode='lines', name='True Trajectory', line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter3d(
        x=y_meas[0], y=y_meas[1], z=y_meas[2],
        mode='markers', name='Radar Measurements',
        marker=dict(size=3, color='red', opacity=0.6)
    ))

    fig.add_trace(go.Scatter3d(
        x=x_est[0], y=x_est[2], z=x_est[4],
        mode='lines', name='Kalman Estimate', line=dict(color='green', dash='dash')
    ))

    fig.update_layout(
        title='3D Aircraft Tracking: True vs Measured vs Estimated',
        scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)'),
        width=800, height=600, margin=dict(l=0, r=0, b=0, t=30)
    )

    fig.show()


def main():
    # Parameters
    dt = 0.1
    T = 10.0
    steps = int(T / dt)
    process_noise_std = 0.1
    measurement_noise_std = 5.0

    # System model
    A, H = discrete_aircraft_model_3d(dt)
    Q = (process_noise_std ** 2) * np.eye(6)
    R = (measurement_noise_std ** 2) * np.eye(3)

    # Initial condition
    x0 = np.array([0.0, 50.0, 0.0, 30.0, 1000.0, -5.0])
    P0 = np.eye(6) * 100.0

    # Simulate system
    x_true, y_meas = simulate_aircraft_trajectory(A, H, Q, R, x0, steps)

    # Estimate states
    x_est, _ = kalman_filter_3d(y_meas, A, H, Q, R, np.zeros(6), P0)

    # Visualize
    plot_3d_trajectory(x_true, y_meas, x_est)

if __name__ == "__main__":
    main()