# main.py

import numpy as np
import plotly.graph_objects as go
from src.cart import Cart
from src.controllers import Controllers  # new import


def main() -> None:
    # Setup initial condition and reference
    initial_state = np.array([[0.0], [0.0]])
    reference_func = lambda t: np.sin(0.5 * t)

    # Create Cart and Controller
    cart = Cart(initial_state=initial_state, reference=reference_func, total_time=50)
    controller_bank = Controllers()

    # Choose controller
    control_func = controller_bank.get("pid")  # or "lqr"

    # Run simulation
    cart.simulate(control_func=control_func)

    # Retrieve simulation data
    t_vals, x_true, v_true, x_hat, v_hat, u_vals, ref_vals = cart.get_results()

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_vals, y=x_true, mode='lines', name='True Position'))
    fig.add_trace(go.Scatter(x=t_vals, y=x_hat, mode='lines', name='Estimated Position', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=t_vals, y=ref_vals, mode='lines', name='Reference Position', line=dict(dash='dot', color='black')))
    fig.add_trace(go.Scatter(x=t_vals, y=v_true, mode='lines', name='True Velocity'))
    fig.add_trace(go.Scatter(x=t_vals, y=v_hat, mode='lines', name='Estimated Velocity', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=t_vals, y=u_vals, mode='lines', name='Control Input', line=dict(dash='dot')))
    fig.update_layout(
        title="1D Cart with Kalman Filter and Feedback Control",
        xaxis_title="Time (s)",
        yaxis_title="Value",
        template="plotly_white"
    )
    fig.show()


if __name__ == "__main__":
    main()
