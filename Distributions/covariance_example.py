"""
Replicates the scalar continuous-time covariance propagation example

System:
    x_dot(t) = A x(t) + B_w w(t)

Covariance differential equation:
    dSigma_x/dt = 2 A Sigma_x + B_w^2 S_w

Example parameters:
    A = -1, B_w = 1, S_w = 2, Sigma_x(0) = 5

Analytical result:
    Sigma_x(t) = 1 + 4 exp(-2 t)
    Sigma_x,ss = 1
"""

import numpy as np
import plotly.graph_objects as go
from pathlib import Path


def main() -> None:
    # --- Parameters from the lecture example ---
    A = -1.0
    B_w = 1.0
    S_w = 2.0
    Sigma_x0 = 5.0

    # --- Time vector ---
    t = np.linspace(0.0, 5.0, 500)

    # --- Analytical covariance solution ---
    # General scalar solution:
    # Sigma_x(t) = (B_w^2 S_w / (2 A)) * (exp(2 A t) - 1)
    #              + Sigma_x(0) * exp(2 A t)
    Sigma_x = (
        (B_w**2 * S_w / (2.0 * A)) * (np.exp(2.0 * A * t) - 1.0)
        + Sigma_x0 * np.exp(2.0 * A * t)
    )

    # Steady-state covariance, valid for A < 0
    Sigma_x_ss = -(B_w**2 * S_w) / (2.0 * A)

    print(f"Steady-state covariance: Sigma_x,ss = {Sigma_x_ss:.2f}")
    print(f"Covariance at t = 0: Sigma_x(0) = {Sigma_x[0]:.2f}")
    print(f"Covariance at t = 5: Sigma_x(5) = {Sigma_x[-1]:.4f}")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=Sigma_x,
            mode="lines",
            line={"width": 3},
            name="Sigma_x(t)",
        )
    )
    fig.add_hline(
        y=Sigma_x_ss,
        line_dash="dash",
        line_width=2,
        annotation_text=f"Sigma_x,ss = {Sigma_x_ss:.0f}",
        annotation_position="top left",
    )
    fig.update_layout(
        title="Continuous-Time State Covariance Propagation",
        xaxis_title="Time",
        yaxis_title="Sigma_x(t)",
        template="plotly_white",
        width=700,
        height=450,
        xaxis={"range": [0, 5]},
        yaxis={"range": [0, 5.2]},
    )

    output_file = Path(__file__).resolve().with_name("covariance_example.html")
    fig.write_html(output_file, include_plotlyjs="cdn")
    print(f"Saved plot to: {output_file}")


if __name__ == "__main__":
    main()
