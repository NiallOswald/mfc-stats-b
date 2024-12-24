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
