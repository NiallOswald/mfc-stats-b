import matplotlib.pyplot as plt
import numpy as np
from .models import Model
from abc import ABC, abstractmethod


class Filter(ABC):
    """A base class for filters."""

    def __init__(self, model: Model):
        """Initialize the filter."""
        self.model = model

        self.rng = model.rng
        self._model_iter = iter(self.model)

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self):
        pass

    @classmethod
    def plot(cls, model: Model, max_iter: int, *args, save_fig: bool = False):
        model_iter = iter(model)
        cls_filter = cls(model, *args)

        x = np.zeros((model.A.shape[0], max_iter))
        y = np.zeros((model.H.shape[0], max_iter))
        x_est = np.zeros((model.A.shape[0], max_iter))

        for i in range(max_iter):
            x[:, i], y[:, i] = next(model_iter)
            x_est[:, i], _ = next(cls_filter)

        plt.figure()
        plt.plot(x[0, :], x[1, :], "b-", label="true trajectory")
        plt.plot(y[0, :], y[1, :], "r.", alpha=0.3, label="measurements")
        plt.plot(x_est[0, :], x_est[1, :], "g-", label=f"{cls.__name__}")
        plt.legend()

        if save_fig:
            plt.savefig(f"{cls.__name__}.png", dpi=300)

        plt.show()


class KalmanFilter(Filter):
    """A Kalman filter implementation for state space models."""

    def __init__(self, model: Model, mu_0: np.ndarray, V_0: np.ndarray):
        """Initialise the Kalman filter."""
        self.mu = mu_0
        self.V = V_0

        super().__init__(model)

    def __next__(self):
        _, y = next(self._model_iter)

        A, Q, H, R = self.model.A, self.model.Q, self.model.H, self.model.R

        mu_pred = A @ self.mu
        V_pred = A @ self.V @ A.T + Q

        S = H @ V_pred @ H.T + R
        K = V_pred @ H.T @ np.linalg.inv(S)

        self.mu = mu_pred + K @ (y - H @ mu_pred)
        self.V = V_pred - K @ H @ V_pred

        return self.mu, self.V


class ParticleFilter(Filter):
    """A particle filter implementation for state space models."""

    def __init__(self, model: Model, particles_init: np.ndarray):
        """Initialise the particle filter."""
        self.x_particles = particles_init
        self.n_particles = particles_init.shape[1]

        self.log_w = np.zeros(self.n_particles)

        super().__init__(model)

    def __next__(self):
        _, y = next(self._model_iter)

        A, Q, H, R = self.model.A, self.model.Q, self.model.H, self.model.R

        x_particles_pred = (
            A @ self.x_particles
            + self.rng.multivariate_normal(
                np.zeros(self.x_particles.shape[0]), Q, self.n_particles
            ).T
        )

        log_w = -0.5 * np.einsum(
            "ij,jk,ki->i",
            (y[:, np.newaxis] - H @ x_particles_pred).T,
            np.linalg.inv(R),
            y[:, np.newaxis] - H @ x_particles_pred,
        )

        W = np.exp(log_w - np.max(log_w))

        w = W / np.sum(W)

        x_est = np.sum(w * x_particles_pred, axis=1)

        self.x_particles = x_particles_pred[
            :, self.rng.choice(self.n_particles, self.n_particles, p=w)
        ]

        return x_est, self.x_particles
