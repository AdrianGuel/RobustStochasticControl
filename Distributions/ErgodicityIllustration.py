import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from plotly.subplots import make_subplots


def running_average(samples: np.ndarray) -> np.ndarray:
    sample_index = np.arange(1, samples.shape[1] + 1)
    return np.cumsum(samples, axis=1) / sample_index


def generate_ergodic_process(
    rng: np.random.Generator,
    num_realizations: int,
    num_samples: int,
) -> np.ndarray:
    return rng.normal(loc=0.0, scale=1.0, size=(num_realizations, num_samples))


def generate_non_ergodic_process(
    rng: np.random.Generator,
    num_realizations: int,
    num_samples: int,
) -> np.ndarray:
    offsets = rng.normal(loc=0.0, scale=1.0, size=(num_realizations, 1))
    return np.repeat(offsets, num_samples, axis=1)


def add_realization_traces(
    fig: go.Figure,
    time: np.ndarray,
    samples: np.ndarray,
    title_prefix: str,
    row: int,
    col: int,
) -> None:
    for index, realization in enumerate(samples, start=1):
        fig.add_trace(
            go.Scatter(
                x=time,
                y=realization,
                mode="lines",
                name=f"{title_prefix} path {index}",
                line=dict(width=1.4),
                showlegend=(row == 1 and col == 1),
            ),
            row=row,
            col=col,
        )


def add_average_traces(
    fig: go.Figure,
    time: np.ndarray,
    averages: np.ndarray,
    title_prefix: str,
    row: int,
    col: int,
) -> None:
    for index, realization_average in enumerate(averages, start=1):
        fig.add_trace(
            go.Scatter(
                x=time,
                y=realization_average,
                mode="lines",
                name=f"{title_prefix} avg {index}",
                line=dict(width=1.4),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    fig.add_hline(
        y=0.0,
        line_dash="dash",
        line_color="black",
        annotation_text="ensemble mean = 0",
        annotation_position="top left",
        row=row,
        col=col,
    )


def main() -> None:
    rng = np.random.default_rng(7)
    num_realizations = 6
    num_samples = 300
    time = np.arange(num_samples)

    ergodic_samples = generate_ergodic_process(rng, num_realizations, num_samples)
    non_ergodic_samples = generate_non_ergodic_process(rng, num_realizations, num_samples)
    ergodic_averages = running_average(ergodic_samples)
    non_ergodic_averages = running_average(non_ergodic_samples)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Ergodic Process Realizations",
            "Non-Ergodic Process Realizations",
            "Ergodic Running Time Averages",
            "Non-Ergodic Running Time Averages",
        ),
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    add_realization_traces(fig, time, ergodic_samples, "ergodic", row=1, col=1)
    add_realization_traces(fig, time, non_ergodic_samples, "non-ergodic", row=1, col=2)
    add_average_traces(fig, time, ergodic_averages, "ergodic", row=2, col=1)
    add_average_traces(fig, time, non_ergodic_averages, "non-ergodic", row=2, col=2)

    fig.update_xaxes(title_text="sample index", row=2, col=1)
    fig.update_xaxes(title_text="sample index", row=2, col=2)
    fig.update_yaxes(title_text="value", row=1, col=1)
    fig.update_yaxes(title_text="value", row=1, col=2)
    fig.update_yaxes(title_text="time average", row=2, col=1)
    fig.update_yaxes(title_text="time average", row=2, col=2)

    fig.update_layout(
        title=(
            "Illustration of Ergodic vs Non-Ergodic Processes"
            "<br><sup>Ergodic paths have time averages that approach the ensemble mean; "
            "non-ergodic paths do not.</sup>"
        ),
        template="plotly_white",
        width=1200,
        height=800,
    )

    output_file = Path(__file__).resolve().with_name("ergodicity_illustration.html")
    fig.write_html(output_file, include_plotlyjs="cdn")
    print(f"Saved plot to: {output_file}")


if __name__ == "__main__":
    main()