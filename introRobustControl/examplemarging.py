import numpy as np
import plotly.graph_objects as go
import control as ctrl

def create_system(K, zeta=0.5, omega_n=1.0):
    num = [K]
    den = [1, 2*zeta*omega_n, omega_n**2]
    return ctrl.TransferFunction(num, den)

# Define systems
G1 = create_system(K=1.0)
G2 = create_system(K=300.0)

# Frequency range
omega = np.logspace(-2, 2, 1000)

# Frequency response (magnitude and phase)
mag1, phase1, _ = ctrl.frequency_response(G1, omega)
mag2, phase2, _ = ctrl.frequency_response(G2, omega)

# Convert to complex response
H1 = mag1 * np.exp(1j * phase1)
H2 = mag2 * np.exp(1j * phase2)

# Extract real and imaginary parts
real_H1 = H1.real
imag_H1 = H1.imag
real_H2 = H2.real
imag_H2 = H2.imag

# Reflect to get the full Nyquist contour
real_H1_full = np.concatenate([real_H1, real_H1[::-1]])
imag_H1_full = np.concatenate([imag_H1, -imag_H1[::-1]])

real_H2_full = np.concatenate([real_H2, real_H2[::-1]])
imag_H2_full = np.concatenate([imag_H2, -imag_H2[::-1]])

# Nyquist plot with full contour
nyquist_fig = go.Figure()
nyquist_fig.add_trace(go.Scatter(x=real_H1_full, y=imag_H1_full, mode='lines', name='K=1.0'))
nyquist_fig.add_trace(go.Scatter(x=real_H2_full, y=imag_H2_full, mode='lines', name='K=3.0'))

nyquist_fig.add_trace(go.Scatter(x=[-1], y=[0], mode='markers+text',
                                 text=["-1"], textposition="top right",
                                 marker=dict(size=10, color="red"),
                                 name='Critical Point'))
nyquist_fig.update_layout(title='Nyquist Plot',
                          xaxis_title='Re',
                          yaxis_title='Im',
                          width=700, height=500,
                          showlegend=True)

# Step responses
T = np.linspace(0, 10, 1000)
T1, y1 = ctrl.step_response(ctrl.feedback(G1), T)
T2, y2 = ctrl.step_response(ctrl.feedback(G2), T)

# Step response plot
step_fig = go.Figure()
step_fig.add_trace(go.Scatter(x=T1, y=y1, mode='lines', name='K=3'))
step_fig.add_trace(go.Scatter(x=T2, y=y2, mode='lines', name='K=100.0'))
step_fig.update_layout(title='Step Response Comparison',
                       xaxis_title='Time (s)',
                       yaxis_title='Output',
                       width=700, height=500,
                       showlegend=True)

# Show both plots
nyquist_fig.show()
step_fig.show()
