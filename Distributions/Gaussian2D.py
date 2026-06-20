import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from plotly.subplots import make_subplots


def gaussian_pdf_2d(
	x_grid: np.ndarray,
	y_grid: np.ndarray,
	mean: np.ndarray,
	covariance: np.ndarray,
) -> np.ndarray:
	determinant = np.linalg.det(covariance)
	if determinant <= 0.0:
		raise ValueError("Covariance matrix must be positive definite.")

	inverse_covariance = np.linalg.inv(covariance)
	points = np.stack((x_grid - mean[0], y_grid - mean[1]), axis=-1)
	quadratic_form = np.einsum("...i,ij,...j->...", points, inverse_covariance, points)
	coefficient = 1.0 / (2.0 * np.pi * np.sqrt(determinant))
	exponent = -0.5 * quadratic_form
	return coefficient * np.exp(exponent)


def main() -> None:
	mean = np.array([0.0, 0.0])
	covariance = np.array([
		[1.0, 0.8],
		[0.8, 4.0],
	])

	eigenvalues = np.linalg.eigvalsh(covariance)
	if np.any(eigenvalues <= 0.0):
		raise ValueError("Covariance matrix must be positive definite.")

	max_std = np.sqrt(np.max(eigenvalues))
	x = np.linspace(mean[0] - 5 * max_std, mean[0] + 5 * max_std, 150)
	y = np.linspace(mean[1] - 5 * max_std, mean[1] + 5 * max_std, 150)
	x_grid, y_grid = np.meshgrid(x, y)
	z = gaussian_pdf_2d(x_grid, y_grid, mean, covariance)

	fig = make_subplots(
		rows=1,
		cols=2,
		specs=[[{"type": "surface"}, {"type": "heatmap"}]],
		subplot_titles=("3D Surface", "Heatmap"),
		horizontal_spacing=0.08,
	)

	fig.add_trace(
		go.Surface(
			x=x,
			y=y,
			z=z,
			colorscale="Viridis",
			showscale=False,
			name="Gaussian Surface",
		),
		row=1,
		col=1,
	)

	fig.add_trace(
		go.Heatmap(
			x=x,
			y=y,
			z=z,
			colorscale="Viridis",
			colorbar=dict(title="Density"),
			name="Gaussian Heatmap",
		),
		row=1,
		col=2,
	)

	fig.update_scenes(
		xaxis_title="x",
		yaxis_title="y",
		zaxis_title="Probability Density",
		row=1,
		col=1,
	)
	fig.update_xaxes(title_text="x", row=1, col=2)
	fig.update_yaxes(title_text="y", row=1, col=2)

	fig.update_layout(
		title=(
			"2D Gaussian Distribution with Full Covariance Matrix"
			f"<br><sup>mean = {mean.tolist()}, covariance = {covariance.tolist()}</sup>"
		),
		template="plotly_white",
		width=1200,
		height=550,
	)

	output_file = Path(__file__).resolve().with_name("gaussian_2d.html")
	fig.write_html(output_file, include_plotlyjs="cdn")
	print(f"Saved plot to: {output_file}")


if __name__ == "__main__":
	main()
