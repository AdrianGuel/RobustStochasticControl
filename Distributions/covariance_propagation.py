"""
Recreates the discrete-time covariance propagation example

System:
    x_k = alpha * x_(k-1) + w_(k-1)

with:
    alpha = 0.75
    E[w_k] = 0
    Sigma_w = var(w_k) = 1
    xbar_0 = 0
    Sigma_x,0 = 50

It generates:
1. Propagation of Sigma_x,k and the steady-state covariance.
2. 100 simulated state trajectories with +/- 3 sigma bounds.
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------
# Parameters from the lecture example
# ---------------------------------------------------------------------
alpha = 0.75
Sigma_w = 1.0
Sigma_x0 = 50.0
xbar_0 = 0.0

num_steps = 20
num_trajectories = 100
rng = np.random.default_rng(seed=5530)  # Reproducible simulations


# ---------------------------------------------------------------------
# Covariance propagation:
# Sigma_x,k = alpha^2 * Sigma_x,k-1 + Sigma_w
# ---------------------------------------------------------------------
time = np.arange(num_steps + 1)

Sigma_x = np.empty(num_steps + 1)
Sigma_x[0] = Sigma_x0

for k in range(1, num_steps + 1):
    Sigma_x[k] = alpha**2 * Sigma_x[k - 1] + Sigma_w

Sigma_x_ss = Sigma_w / (1.0 - alpha**2)
three_sigma = 3.0 * np.sqrt(Sigma_x)


# ---------------------------------------------------------------------
# Monte Carlo simulation of 100 trajectories
# ---------------------------------------------------------------------
# Initial state samples: x_0 ~ N(xbar_0, Sigma_x0)
x = np.empty((num_trajectories, num_steps + 1))
x[:, 0] = rng.normal(
    loc=xbar_0,
    scale=np.sqrt(Sigma_x0),
    size=num_trajectories,
)

# Process-noise samples: w_k ~ N(0, Sigma_w)
for k in range(1, num_steps + 1):
    w_k_minus_1 = rng.normal(
        loc=0.0,
        scale=np.sqrt(Sigma_w),
        size=num_trajectories,
    )
    x[:, k] = alpha * x[:, k - 1] + w_k_minus_1


# ---------------------------------------------------------------------
# Create the interactive Plotly figure
# ---------------------------------------------------------------------
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=(
        rf"Covariance propagation; steady state $\Sigma_x={Sigma_x_ss:.2f}$",
        "Propagation of state, with error bounds",
    ),
)

# Left panel: covariance propagation
fig.add_trace(
    go.Scatter(
        x=time,
        y=Sigma_x,
        mode="markers",
        name=r"$\Sigma_{x,k}$",
        hovertemplate="Time step: %{x}<br>Covariance: %{y:.4f}<extra></extra>",
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=time,
        y=np.full_like(time, Sigma_x_ss, dtype=float),
        mode="lines",
        line=dict(dash="dash"),
        name=rf"$\Sigma_{{x,\mathrm{{ss}}}}={Sigma_x_ss:.2f}$",
        hovertemplate="Steady-state covariance: %{y:.4f}<extra></extra>",
    ),
    row=1,
    col=1,
)

# Right panel: 100 state realizations
for i in range(num_trajectories):
    fig.add_trace(
        go.Scatter(
            x=time,
            y=x[i, :],
            mode="lines",
            showlegend=False,
            hovertemplate=(
                f"Trajectory: {i + 1}<br>"
                "Time step: %{x}<br>"
                "State: %{y:.4f}<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )

# +/- 3 sigma bounds
fig.add_trace(
    go.Scatter(
        x=time,
        y=three_sigma,
        mode="lines",
        line=dict(dash="dash"),
        name=r"$+3\sqrt{\Sigma_{x,k}}$",
        hovertemplate="Upper 3σ bound: %{y:.4f}<extra></extra>",
    ),
    row=1,
    col=2,
)

fig.add_trace(
    go.Scatter(
        x=time,
        y=-three_sigma,
        mode="lines",
        line=dict(dash="dash"),
        name=r"$-3\sqrt{\Sigma_{x,k}}$",
        hovertemplate="Lower 3σ bound: %{y:.4f}<extra></extra>",
    ),
    row=1,
    col=2,
)

fig.update_xaxes(title_text="Time step, k", row=1, col=1)
fig.update_xaxes(title_text="Time step, k", row=1, col=2)
fig.update_yaxes(title_text=r"$\Sigma_{x,k}$", range=[0, 52], row=1, col=1)
fig.update_yaxes(title_text="State value", range=[-30, 30], row=1, col=2)

fig.update_layout(
    title=(
        "Scalar Discrete-Time System: Covariance Propagation and "
        "Monte Carlo State Trajectories"
    ),
    template="plotly_white",
    width=1250,
    height=550,
    legend_title_text="Traces",
    margin=dict(l=70, r=35, t=90, b=70),
)

# ---------------------------------------------------------------------
# Save an interactive, self-contained HTML file beside this script
# ---------------------------------------------------------------------
output_file = Path(__file__).with_name("covariance_propagation.html")
fig.write_html(output_file, include_plotlyjs=True, full_html=True)

print(f"Saved Plotly HTML file to: {output_file}")
print(f"Steady-state covariance: Sigma_x,ss = {Sigma_x_ss:.6f}")
