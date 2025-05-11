# simulate_experiment.py

import numpy as np
from scipy.signal import lsim, TransferFunction

def simulate_experiment(G, duration=30, dt=0.01, seed=42):
    """
    Simulate the response of a system G(s) to random step-like inputs.

    Args:
        G: scipy.signal.TransferFunction
            Continuous-time system to simulate.
        duration: total time in seconds
        dt: time step
        seed: random seed for reproducibility

    Returns:
        t: time array
        u: input signal (random step-wise)
        y: output signal (system response)
    """
    np.random.seed(seed)
    t = np.arange(0, duration, dt)

    # Create random step-wise input: change value every 1 second
    steps = int(1 / dt)
    u = np.repeat(np.random.uniform(-1, 1, len(t) // steps + 1), steps)[:len(t)]

    # Simulate system response to input
    tout, y, _ = lsim(G, U=u, T=t)

    return t, u, y
