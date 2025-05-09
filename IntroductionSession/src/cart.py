# cart.py

import numpy as np
from typing import Callable, Optional, Tuple, Union

class Cart:
    """
    Simulates a 1D cart system with velocity-dependent friction,
    controlled via state feedback and estimated with a Kalman filter.
    """

    def __init__(
        self,
        mass: float = 1.0,
        friction: float = 0.1,
        dt: float = 0.01,
        total_time: float = 10.0,
        initial_state: Optional[np.ndarray] = None,
        reference: Union[float, Callable[[float], float]] = 0.0
    ):
        """
        Initialize the Cart object.

        Args:
            mass: Mass of the cart (kg).
            friction: Viscous friction coefficient (N·s/m).
            dt: Simulation time step (s).
            total_time: Total simulation duration (s).
            initial_state: Initial [position; velocity] column vector (2x1).
            reference: Desired position (scalar or time-varying function of time).
        """
        self.mass = mass
        self.friction = friction
        self.dt = dt
        self.total_time = total_time
        self.n_steps = int(total_time / dt)
        self.reference = reference
        self.initial_state = initial_state if initial_state is not None else np.zeros((2, 1))
        self.reset()

        # Discrete-time system model (for Kalman filter and controller)
        self.A = np.array([[1, dt], [0, 1 - friction * dt / mass]])
        self.B = np.array([[0], [dt / mass]])
        self.H = np.array([[1.0, 0.0]])  # Only position is measured

        # Kalman filter parameters
        self.Q = 0.001 * np.eye(2)       # Process noise covariance
        self.R = np.array([[0.3]])       # Measurement noise covariance
        self.P = 0.5 * np.eye(2)         # Initial state covariance
        self.x_hat = np.zeros((2, 1))    # Initial state estimate

    def reset(self) -> None:
        """Resets the simulation state and history."""
        self.state = self.initial_state.copy()
        self.history = []

    def dynamics(self, t: float, state: np.ndarray, control_force: float = 0.0) -> np.ndarray:
        """
        Computes the continuous-time dynamics of the system.

        Args:
            t: Current time (not used, included for compatibility).
            state: Current [position; velocity] (2x1 array).
            control_force: Control input force (N).

        Returns:
            Derivative of the state (2x1 array).
        """
        x, v = state.flatten()
        dxdt = v
        dvdt = (control_force - self.friction * v) / self.mass
        return np.array([dxdt, dvdt], dtype=float).reshape(2, 1)

    def rk4_step(self, t: float, state: np.ndarray, control_force: float) -> np.ndarray:
        """
        Performs one Runge-Kutta 4th order integration step.

        Args:
            t: Current time (s).
            state: Current state (2x1 array).
            control_force: Control input (N).

        Returns:
            New state after dt (2x1 array).
        """
        dt = self.dt
        f = lambda t, s: self.dynamics(t, s, control_force)
        k1 = f(t, state)
        k2 = f(t + dt / 2, state + dt * k1 / 2)
        k3 = f(t + dt / 2, state + dt * k2 / 2)
        k4 = f(t + dt, state + dt * k3)
        return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def sensor_reading(self) -> np.ndarray:
        """
        Simulates a noisy measurement of the state.

        Returns:
            Noisy [position; velocity] (2x1 array).
        """
        noise = np.random.normal(0, 0.2, (2, 1))  # stddev of 0.2
        return self.state + noise

    def kalman_update(self, z: np.ndarray, u: float) -> np.ndarray:
        """
        Updates the Kalman filter state estimate.

        Args:
            z: Measurement (2x1 array).
            u: Control input (scalar).

        Returns:
            Updated state estimate (2x1 array).
        """
        # Predict step
        x_pred = self.A @ self.x_hat + self.B * u
        P_pred = self.A @ self.P @ self.A.T + self.Q

        # Measurement update (only position used)
        y = z[0] - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        self.x_hat = x_pred + K @ y
        self.P = (np.eye(2) - K @ self.H) @ P_pred
        return self.x_hat

    def simulate(self, control_func: Callable[[np.ndarray, float], float]) -> None:
        """
        Runs the simulation with feedback control and Kalman filtering.

        Args:
            control_func: A function that computes control input from (estimated_state, reference).
        """
        t = 0.0
        for _ in range(self.n_steps):
            ref = self.reference(t) if callable(self.reference) else self.reference
            u = float(control_func(self.x_hat, ref))
            self.state = self.rk4_step(t, self.state, u)
            z = self.sensor_reading()
            self.kalman_update(z, u)
            self.history.append((t, self.state.copy(), self.x_hat.copy(), u, ref))
            t += self.dt

    def get_results(self) -> Tuple[list[float], list[float], list[float], list[float], list[float], list[float], list[float]]:
        """
        Returns the simulation history for plotting or analysis.

        Returns:
            Tuple of lists: (time, true_pos, true_vel, est_pos, est_vel, control, reference)
        """
        t_vals = [entry[0] for entry in self.history]
        x_true = [entry[1][0, 0] for entry in self.history]
        v_true = [entry[1][1, 0] for entry in self.history]
        x_hat = [entry[2][0, 0] for entry in self.history]
        v_hat = [entry[2][1, 0] for entry in self.history]
        u_vals = [entry[3] for entry in self.history]
        ref_vals = [entry[4] for entry in self.history]
        return t_vals, x_true, v_true, x_hat, v_hat, u_vals, ref_vals
