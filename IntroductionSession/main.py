# main.py

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.cart import Cart
from src.controllers import Controllers


def build_result_plot(t_vals, x_true, x_hat, ref_vals, v_true, v_hat, u_vals):
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=t_vals, y=x_true, name="True Position"))
    fig.add_trace(go.Scatter(x=t_vals, y=x_hat, name="Estimated Position", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=t_vals, y=ref_vals, name="Reference", line=dict(dash="dot", color="black")))
    fig.add_trace(go.Scatter(x=t_vals, y=v_true, name="True Velocity"))
    fig.add_trace(go.Scatter(x=t_vals, y=v_hat, name="Estimated Velocity", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=t_vals, y=u_vals, name="Control Input", line=dict(dash="dot")))

    fig.update_layout(
        height=500,
        title_text="System States, Estimates, and Control",
        template="plotly_white",
        xaxis_title="Time (s)",
        yaxis_title="Value"
    )
    return fig


def create_cart_shape(x, y, cart_width=0.5, cart_height=0.2, color="red"):
    body_x = [x - cart_width / 2, x + cart_width / 2, x + cart_width / 2, x - cart_width / 2, x - cart_width / 2]
    body_y = [y, y, y + cart_height, y + cart_height, y]
    return go.Scatter(x=body_x, y=body_y, fill='toself', mode='lines', line=dict(color=color))


def create_wheels(x, y, spacing=0.25, size=10):
    return [
        go.Scatter(x=[x - spacing], y=[y], mode='markers', marker=dict(size=size, color='gray')),
        go.Scatter(x=[x + spacing], y=[y], mode='markers', marker=dict(size=size, color='gray'))
    ]


def build_cart_animation(t_vals, x_true, ref_vals):
    frames = []
    for i in range(len(t_vals)):
        ref_cart = create_cart_shape(ref_vals[i], 0.5, color="black")
        ref_wheels = create_wheels(ref_vals[i], 0.5)
        true_cart = create_cart_shape(x_true[i], -0.5, color="red")
        true_wheels = create_wheels(x_true[i], -0.5)
        frames.append(go.Frame(data=[ref_cart] + ref_wheels + [true_cart] + true_wheels, name=str(i)))

    # Initial frame
    fig = go.Figure()
    fig.add_trace(create_cart_shape(ref_vals[0], 0.5, color="black"))
    for wheel in create_wheels(ref_vals[0], 0.5):
        fig.add_trace(wheel)
    fig.add_trace(create_cart_shape(x_true[0], -0.5, color="red"))
    for wheel in create_wheels(x_true[0], -0.5):
        fig.add_trace(wheel)

    fig.update_layout(
        height=300,
        title="1D Cart Animation",
        template="plotly_white",
        xaxis=dict(title="Cart Position", range=[min(ref_vals) - 1, max(ref_vals) + 1]),
        yaxis=dict(range=[-1, 1], showticklabels=False),
        showlegend=False,
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[dict(label="Play", method="animate",
                          args=[None, {"frame": {"duration": 30, "redraw": True}, "fromcurrent": True}])]
        )],
        sliders=[dict(
            steps=[dict(method="animate", args=[[str(i)], {"frame": {"duration": 0}, "mode": "immediate"}],
                        label=f"{t_vals[i]:.1f}") for i in range(len(t_vals))],
            transition=dict(duration=0),
            x=0.1,
            xanchor="left",
            y=0,
            yanchor="top"
        )]
    )
    fig.frames = frames
    return fig


def simulate_cart(dt: float = 0.01):
    initial_state = np.array([[0.0], [0.0]])
    reference_func = lambda t: np.sin(0.5 * t)

    cart = Cart(initial_state=initial_state, reference=reference_func, total_time=50, dt=dt)
    controller_bank = Controllers()
    control_func = controller_bank.get("pid")
    cart.simulate(control_func=control_func)
    return cart.get_results()


def main():
    st.set_page_config(layout="wide", page_title="1D Cart Simulator")

    # White background CSS
    st.markdown("""
        <style>
            body {
                background-color: white;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("1D Cart Simulation with Kalman Filter and Controllers")

    # Adjustable simulation time step
    dt = st.sidebar.slider("Sampling Time (dt)", min_value=0.005, max_value=0.1, value=0.01, step=0.005, format="%.3f")

    # Run simulation
    t_vals, x_true, v_true, x_hat, v_hat, u_vals, ref_vals = simulate_cart(dt=dt)

    # Tabs
    tab1, tab2 = st.tabs(["Simulation Results", "🎥 Animation"])

    with tab1:
        st.plotly_chart(build_result_plot(t_vals, x_true, x_hat, ref_vals, v_true, v_hat, u_vals), use_container_width=True)

    with tab2:
        st.plotly_chart(build_cart_animation(t_vals, x_true, ref_vals), use_container_width=True)


if __name__ == "__main__":
    main()
