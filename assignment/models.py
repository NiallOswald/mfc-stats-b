"""State space models for Kalman and particle filters."""

import autograd.numpy as np
from dataclasses import dataclass
from typing import Callable


@dataclass
class Model:
    """A linear Gaussian state space model."""

    x_0: np.ndarray
    A: np.ndarray
    H: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    seed: int = 1234

    @property
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)

    def __iter__(self):
        return ModelIterator(self)


class ModelIterator:
    """An iterator that generates state and observations from a state space model."""

    def __init__(self, model: Model):
        self.model = model

        self.next_x = model.x_0
        self.rng = model.rng

    def __iter__(self) -> "ModelIterator":
        return self

    def __next__(self) -> tuple[np.ndarray, np.ndarray]:
        x = self.next_x
        y = self.model.H @ x + self.rng.multivariate_normal(
            np.zeros(self.model.R.shape[0]), self.model.R
        )

        # Update the iterator
        self.next_x = self.model.A @ x + self.rng.multivariate_normal(
            np.zeros(self.model.Q.shape[0]), self.model.Q
        )

        return x, y


@dataclass
class ModelGen:
    """A parameterized linear Gaussian state space model."""

    x_0: Callable[[np.ndarray], np.ndarray]
    A: Callable[[np.ndarray], np.ndarray]
    H: Callable[[np.ndarray], np.ndarray]
    Q: np.ndarray
    R: np.ndarray
    seed: int = 1234

    def __call__(self, params: np.ndarray) -> Model:
        """Generate a model with the given parameters."""
        return Model(
            self.x_0(params),
            self.A(params),
            self.H(params),
            self.Q,
            self.R,
            self.seed,
        )

    def __iter__(self):
        raise NotImplementedError("Cannot iterate over a model generator.")


def k_ballistic_model(k: float) -> ModelGen:
    """A parameterized linear ballistic model."""
    return ModelGen(
        lambda params: np.zeros(4),  # x_0
        lambda params: np.array(  # A
            [
                [1.0, 0.0, params[0], 0.0],
                [0.0, 1.0, 0.0, params[0]],
                [0.0, 0.0, 0.99, 0.0],
                [0.0, 0.0, 0.0, 0.99],
            ]
        ),
        lambda params: np.array(  # H
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        ),
        np.array(  # Q
            [
                [k**3 / 3, 0.0, k**2 / 2, 0.0],
                [0.0, k**3 / 3, 0.0, k**2 / 2],
                [k**2 / 2, 0.0, k, 0.0],
                [0.0, k**2 / 2, 0.0, k],
            ]
        ),
        np.array(  # R
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
    )
