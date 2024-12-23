import matplotlib.pyplot as plt
import numpy as np
from .models import Model
from abc import ABC, abstractmethod


class Filter(ABC):
    """A base class for filters."""

    def __init__(self, model: Model, mu_0: np.ndarray, V_0: np.ndarray):
        """Initialize the filter with a model and initial state."""
        self.model = model
        self.mu = mu_0
        self.V = V_0

        self._model_iter = iter(self.model)

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self):
        pass

    @classmethod
    def plot(cls, model: Model, mu_0: np.ndarray, V_0: np.ndarray, max_iter: int):
        model_iter = iter(model)
        cls_filter = cls(model, mu_0, V_0)

        x = np.zeros((model.A.shape[0], max_iter))
        y = np.zeros((model.H.shape[0], max_iter))
        mu = np.zeros((model.A.shape[0], max_iter))
        V = np.zeros((*model.Q.shape, max_iter))

        x[:, 0] = model.x_0
        mu[:, 0] = mu_0
        V[:, :, 0] = V_0

        for i in range(1, max_iter):
            x[:, i], y[:, i] = next(model_iter)
            mu[:, i], V[:, :, i] = next(cls_filter)

        plt.plot(x[0, :], x[1, :], "b-", label="true trajectory")
        plt.plot(y[0, :], y[1, :], "r.", alpha=0.3, label="measurements")
        plt.plot(mu[0, :], mu[1, :], "g-", label=f"{cls.__name__}")
        plt.legend()
        plt.show()


class KalmanFilter(Filter):
    """A Kalman filter implementation for state space models."""

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
