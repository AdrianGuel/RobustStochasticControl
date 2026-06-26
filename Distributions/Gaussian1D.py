import numpy as np
import plotly.graph_objects as go
from pathlib import Path

def gaussian_pdf(x: np.ndarray, mean: float, variance: float) -> np.ndarray:
    return (1.0 / np.sqrt(2.0 * np.pi * variance)) * np.exp(
        -((x - mean) ** 2) / (2.0 * variance)
    )

def main() -> None:
    mean = 0.0
    variances = [0.25, 1.0, 2.0, 4.0]

    max_std = np.sqrt(max(variances))
    x = np.linspace(mean - 5 * max_std, mean + 5 * max_std, 1000)

    fig = go.Figure()

    for variance in variances:
        y = gaussian_pdf(x, mean, variance)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=f"mean = {mean}, variance = {variance}",
            )
        )

    fig.add_vline(
        x=mean,
        line_dash="dash",
        line_color="black",
        annotation_text=f"mean = {mean}",
        annotation_position="top",
    )

    fig.update_layout(
        title="1D Gaussian Distributions with Same Mean and Different Variances",
        xaxis_title="x",
        yaxis_title="Probability Density",
        template="plotly_white",
        width=900,
        height=500,
    )

    output_file = Path(__file__).resolve().with_name("gaussian_1d.html")
    fig.write_html(output_file, include_plotlyjs="cdn")
    print(f"Saved plot to: {output_file}")


if __name__ == "__main__":
    main()