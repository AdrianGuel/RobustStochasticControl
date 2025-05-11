# final_algorithm.py

import numpy as np
import plotly.graph_objects as go
from src.simulate_experiment import simulate_experiment
from src.least_squares_offline import least_squares_offline
from src.recursive_least_squares import recursive_least_squares
from scipy.signal import TransferFunction, dstep

# Parameters
gamma = 0.8
a = 1.0
n = 2
ts = 0.1

# Define the reference transfer function G(s) = 1 / (s^2 + 5s + 10)
num = [1]
den = [1, 5, 10]
G = TransferFunction(num, den)

# Simulate system
t, u, y = simulate_experiment(G,dt=ts)
theta_0, P, f_k = least_squares_offline(u, y, n, a, gamma)

# Construct the identified discrete-time transfer function from theta_0
numdi = np.array([theta_0[n + i] for i in range(n)])
dendi = np.zeros(n + 1)
dendi[0] = 1
for i in range(n):
    dendi[i + 1] = -theta_0[i]

# Create discrete-time transfer function
sysi = TransferFunction(numdi, dendi, dt=ts)

# Generate step responses
t_cont, y_cont = G.step(T=np.linspace(0, 5, 500))
t_disc, y_disc = dstep(sysi, n=int(np.ceil(5 / ts)))
y_disc = np.squeeze(y_disc)
t_disc = np.arange(len(y_disc)) * ts

# Create the Plotly figure
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=t, y=y,
    mode='lines',
    name='Output',
    line=dict(dash='solid')
))

fig.add_trace(go.Scatter(
    x=t, y=u,
    mode='lines',
    name='Input',
    line=dict(dash='solid')
))

fig.show()

# Create the Plotly figure
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=t_cont, y=y_cont,
    mode='lines',
    name='True system (continuous)',
    line=dict(dash='solid')
))

fig.add_trace(go.Scatter(
    x=t_disc, y=y_disc,
    mode='lines+markers',
    name='Estimated system (discrete)',
    line=dict(dash='dot'),
    marker=dict(size=4)
))

fig.update_layout(
    title='Step Response Comparison Offline RLSs',
    xaxis_title='Time [s]',
    yaxis_title='Response',
    legend=dict(x=0.01, y=0.99),
    template='plotly_white'
)

fig.show()
