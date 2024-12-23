"""State space models for Kalman and particle filters."""

import numpy as np
from dataclasses import dataclass


@dataclass
class Model:
    """A linear Gaussian state space model."""

    x_0: np.ndarray
    A: np.ndarray
    Q: np.ndarray
    H: np.ndarray
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

        self.x = model.x_0
        self.rng = model.rng

    def __iter__(self) -> "ModelIterator":
        return self

    def __next__(self) -> tuple[np.ndarray, np.ndarray]:
        self.x = self.model.A @ self.x + self.rng.multivariate_normal(
            np.zeros_like(self.x), self.model.Q
        )
        y = self.model.H @ self.x + self.rng.multivariate_normal(
            np.zeros_like(self.model.H @ self.x), self.model.R
        )

        return self.x, y
