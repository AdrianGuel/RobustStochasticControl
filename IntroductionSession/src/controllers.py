# controllers.py
import numpy as np
from typing import Callable


class Controllers:
    """
    A collection of feedback control strategies for the 1D Cart system.
    """

    def __init__(self):
        # PID internal state
        self.integral_error = 0.0
        self.prev_error = 0.0

    def lqr(self, x_hat: np.ndarray, reference: float) -> float:
        """
        LQR feedback controller: u = K * (reference - estimated state)

        Args:
            x_hat: Estimated state [position; velocity] (2x1).
            reference: Desired position (float).

        Returns:
            Control force (float).
        """
        K = np.array([[5.5, 1.2]])  # gains: position, velocity
        error = reference - x_hat[0, 0]
        control = K[0, 0] * error - K[0, 1] * x_hat[1, 0]
        return control

    def pid(self, x_hat: np.ndarray, reference: float, dt: float = 0.01) -> float:
        """
        PID controller using estimated position and velocity.

        Args:
            x_hat: Estimated state [position; velocity] (2x1).
            reference: Desired position (float).
            dt: Time step (for integral and derivative terms).

        Returns:
            Control force (float).
        """
        Kp = 10.0
        Ki = 0.5
        Kd = 2.5

        error = reference - x_hat[0, 0]
        derivative = (error - self.prev_error) / dt
        self.integral_error += error * dt
        self.prev_error = error

        control = Kp * error + Ki * self.integral_error + Kd * (-x_hat[1, 0])
        return control

    def get(self, name: str) -> Callable[[np.ndarray, float], float]:
        """
        Retrieves a controller by name.

        Args:
            name: Name of the controller ('lqr' or 'pid').

        Returns:
            A controller function.
        """
        if name == "lqr":
            return self.lqr
        elif name == "pid":
            return lambda x_hat, ref: self.pid(x_hat, ref)
        else:
            raise ValueError(f"Unknown controller '{name}'. Available: ['lqr', 'pid']")
