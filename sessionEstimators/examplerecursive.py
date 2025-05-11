import numpy as np
import plotly.graph_objects as go

# ----------------------------------------
# Parameters
n = 2                         # Dimension of state
m = 1                         # Dimension of each measurement
steps = 50                   # Total time steps

# True state (constant but unknown)
X_true = np.array([[2.0], [-1.0]])

# Generate synthetic time-varying measurements
np.random.seed(0)
H_seq = [np.random.randn(m, n) for _ in range(steps)]
R_seq = [0.1 * np.eye(m) for _ in range(steps)]
v_seq = [np.random.multivariate_normal(np.zeros(m), R_seq[k])[:, np.newaxis] for k in range(steps)]
y_seq = [H_seq[k] @ X_true + v_seq[k] for k in range(steps)]

# ----------------------------------------
# Recursive Least Squares Estimation
X_hat = np.zeros((n, 1))               # Initial estimate
Q_inv = 1e-6 * np.eye(n)               # Initial Q^-1, Small positive definite initialization

X_estimates = [X_hat.copy()]           # Store estimates over time

for k in range(steps):
    H_k = H_seq[k]
    R_k = R_seq[k]
    y_k = y_seq[k]

    # Update covariance
    Q_inv += H_k.T @ np.linalg.inv(R_k) @ H_k
    Q = np.linalg.inv(Q_inv)

    # Innovation
    innovation = y_k - H_k @ X_hat

    # Update estimate
    X_hat = X_hat + Q @ H_k.T @ np.linalg.inv(R_k) @ innovation

    X_estimates.append(X_hat.copy())

# ----------------------------------------
# Plotting
X_estimates = np.array(X_estimates).squeeze()
time = np.arange(steps + 1)

fig = go.Figure()
fig.add_trace(go.Scatter(y=[X_true[0, 0]] * len(time), x=time,
                         mode="lines", name="True X[0]", line=dict(dash='dash')))
fig.add_trace(go.Scatter(y=[X_true[1, 0]] * len(time), x=time,
                         mode="lines", name="True X[1]", line=dict(dash='dash')))

fig.add_trace(go.Scatter(y=X_estimates[:, 0], x=time,
                         mode="lines+markers", name="Estimated X[0]"))
fig.add_trace(go.Scatter(y=X_estimates[:, 1], x=time,
                         mode="lines+markers", name="Estimated X[1]"))

fig.update_layout(title="Recursive Least Squares Estimation",
                  xaxis_title="Time Step", yaxis_title="State Value",
                  template="plotly_white")
fig.show()
