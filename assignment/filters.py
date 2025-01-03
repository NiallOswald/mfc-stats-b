import autograd.numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from .models import Model, ModelGen, ModelGenAuto
from .utils import MultivaraiateNormal
from abc import ABC, abstractmethod
from autograd import grad


class Filter(ABC):
    """A base class for filters."""

    def __init__(self, model: Model):
        """Initialize the filter."""
        self.model = model

        self._model_iter = iter(self.model)

    def __iter__(self) -> "Filter":
        return self

    @abstractmethod
    def __next__(self) -> tuple[np.ndarray, np.ndarray]:
        """Advance to the next observation."""
        pass

    @classmethod
    def plot(cls, model: Model, max_iter: int, *args, save_fig: bool = False):
        """Plot the trajectory, observations, and filter estimate."""
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

    @classmethod
    def likelihood(cls, true_model: Model, model: Model, max_iter: int, *args) -> float:
        """Compute the log likelihood."""
        true_model_iter = iter(true_model)
        cls_filter = cls(model, *args)

        log_li = 0
        for i in range(max_iter):
            _, y = next(true_model_iter)  # Observation at current time step
            log_li = log_li + cls_filter._marginal(
                y
            )  # Marginal based on last time step
            next(cls_filter)

        return log_li

    @abstractmethod
    def _marginal(self, y: np.ndarray) -> float:
        """Return the likelihood of the observation y at the next time step."""
        pass

    @classmethod
    def fit(
        cls,
        data_model: Model,
        model_gen: ModelGen,
        params: np.ndarray,
        filter_args: tuple,
        model_iter: int = 1000,
        use_autograd: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """Fit a model to the data."""

        if not isinstance(model_gen, ModelGenAuto) and use_autograd:
            raise ValueError("Autograd can only be used with AutoModelGen.")

        def f(x):
            return -cls.likelihood(data_model, model_gen(x), model_iter, *filter_args)

        if use_autograd:
            fit_params = sp.optimize.minimize(f, params, jac=grad(f), **kwargs)
        else:
            fit_params = sp.optimize.minimize(f, params, **kwargs)

        if not fit_params.success:
            raise sp.optimize.OptimizeWarning(fit_params.message)

        return fit_params.x


class KalmanFilter(Filter):
    """A Kalman filter implementation for state space models."""

    def __init__(self, model: Model, mu_0: np.ndarray, V_0: np.ndarray):
        """Initialise the Kalman filter."""
        self.mu = mu_0
        self.V = V_0

        super().__init__(model)

    def __next__(self) -> tuple[np.ndarray, np.ndarray]:
        _, y = next(self._model_iter)

        A, Q, H, R = self.model.A, self.model.Q, self.model.H, self.model.R

        mu_pred = A @ self.mu
        V_pred = A @ self.V @ A.T + Q

        S = H @ V_pred @ H.T + R
        K = V_pred @ H.T @ np.linalg.inv(S)

        self.mu = mu_pred + K @ (y - H @ mu_pred)
        self.V = V_pred - K @ H @ V_pred

        return self.mu, self.V

    def _marginal(self, y: np.ndarray) -> float:
        A, Q, H, R = self.model.A, self.model.Q, self.model.H, self.model.R

        mu_li = H @ A @ self.mu
        var_li = (H @ A) @ self.V @ (H @ A).T + H @ Q @ H.T + R

        return MultivaraiateNormal.logpdf(y, mu_li, var_li)


class ParticleFilter(Filter):
    """A particle filter implementation for state space models."""

    def __init__(self, model: Model, particles_init: np.ndarray, seed: int = None):
        """Initialise the particle filter."""
        self.x_particles = particles_init
        self.n_particles = particles_init.shape[1]

        self.log_w = np.zeros(self.n_particles)

        self.rng = np.random.default_rng(seed)

        super().__init__(model)

    def __next__(self) -> tuple[np.ndarray, np.ndarray]:
        _, y = next(self._model_iter)

        A, Q, H, R = self.model.A, self.model.Q, self.model.H, self.model.R

        # Sample
        x_particles_pred = (
            A @ self.x_particles
            + self.rng.multivariate_normal(np.zeros(Q.shape[0]), Q, self.n_particles).T
        )

        # Calculate weights
        log_w = MultivaraiateNormal.logpdf(y, H @ x_particles_pred, R)

        W = np.exp(log_w - np.max(log_w))

        w = W / np.sum(W)

        x_est = np.sum(w * x_particles_pred, axis=1)

        # Resample
        self.x_particles = x_particles_pred[
            :, self.rng.choice(self.n_particles, self.n_particles, p=w)
        ]  # Not compatible with autograd

        return x_est, self.x_particles

    def _marginal(self, y: np.ndarray) -> float:
        A, Q, H, R = self.model.A, self.model.Q, self.model.H, self.model.R

        x_particles_pred = (
            A @ self.x_particles
            + self.rng.multivariate_normal(np.zeros(Q.shape[0]), Q, self.n_particles).T
        )

        # Take the mean of the likelihoods (unstable but works well)
        return np.log(np.mean(MultivaraiateNormal.pdf(y, H @ x_particles_pred, R)))
