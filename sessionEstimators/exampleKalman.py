import numpy as np
import plotly.graph_objects as go

# System parameters
a = 1.0         # System dynamic
c = 1.0         # Measurement matrix
Xi_v = 1 / 4    # Measurement noise variance
Xi_w = 1        # Process noise variance

# Time steps
N = 40
np.random.seed(42)
w = np.sqrt(Xi_w) * np.random.randn(N)
v = np.sqrt(Xi_v) * np.random.randn(N)

# Initialization
x = np.zeros(N+1)               # True state
xhat_minus = np.zeros(N+1)     # Prior estimate
xhat_plus = np.zeros(N)        # Posterior estimate
Xi_x_minus = np.zeros(N+1)     # Prior covariance
Xi_x_plus = np.zeros(N)        # Posterior covariance
L = np.zeros(N)                # Kalman gain
y = np.zeros(N)                # Measurements

# Kalman filter loop
for k in range(N):
    # Kalman gain
    L[k] = Xi_x_minus[k] * c / (c * Xi_x_minus[k] * c + Xi_v)

    # True system and measurement
    x[k+1] = a * x[k] + w[k]
    y[k] = c * x[k] + v[k]

    # Measurement update
    xhat_plus[k] = xhat_minus[k] + L[k] * (y[k] - c * xhat_minus[k])
    Xi_x_plus[k] = (1 - L[k] * c) * Xi_x_minus[k]

    # Time update
    xhat_minus[k+1] = a * xhat_plus[k]
    Xi_x_minus[k+1] = a * Xi_x_plus[k] * a + Xi_w

# ---- PLOT 1: True state, estimate, and error
k = np.arange(N)
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=k, y=x[:N], mode='lines', name='true'))
fig1.add_trace(go.Scatter(x=k, y=xhat_plus, mode='lines', name='estimate'))
fig1.add_trace(go.Scatter(x=k, y=x[:N] - xhat_plus, mode='lines', name='error'))
fig1.update_layout(title='Kalman filter in action',
                   xaxis_title='Iteration',
                   yaxis_title='State',
                   template='plotly_white')
fig1.show()

# ---- PLOT 2: Error covariance before and after update
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=k, y=Xi_x_minus[:N], mode='markers+lines', name='Xi_x^-'))
fig2.add_trace(go.Scatter(x=k, y=Xi_x_plus, mode='markers+lines', name='Xi_x^+'))
fig2.update_layout(title='Error covariance before and after measurement update',
                   xaxis_title='Iteration',
                   yaxis_title='Covariance',
                   template='plotly_white')
fig2.show()
