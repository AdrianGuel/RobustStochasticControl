# simulate_experiment.py

import numpy as np
from scipy.signal import TransferFunction, lsim

def simulate_experiment(duration=30, dt=0.01, zeta=0.7, omega_n=1.0, seed=42):
    """
    Simulate the response of a second-order system to random step-like inputs.

    Args:
        duration: total time in seconds
        dt: time step
        zeta: damping ratio
        omega_n: natural frequency
        seed: random seed for reproducibility

    Returns:
        t: time array
        u: input signal (random step-wise)
        y: output signal (system response)
    """
    np.random.seed(seed)
    t = np.arange(0, duration, dt)

    # Define second-order system: G(s) = omega_n^2 / (s^2 + 2*zeta*omega_n*s + omega_n^2)
    num = [omega_n**2]
    den = [1, 2*zeta*omega_n, omega_n**2]
    system = TransferFunction(num, den)

    # Create random step-wise input: change value every 1 second
    steps = int(1 / dt)
    u = np.repeat(np.random.uniform(-1, 1, len(t) // steps + 1), steps)[:len(t)]

    # Simulate response
    tout, y, _ = lsim(system, U=u, T=t)
    return t, u, y
