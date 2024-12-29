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
    Q: Callable[[np.ndarray], np.ndarray]
    R: Callable[[np.ndarray], np.ndarray]
    seed: int = 1234

    def __call__(self, params: np.ndarray) -> Model:
        """Generate a model with the given parameters."""
        return Model(
            self.x_0(params),
            self.A(params),
            self.H(params),
            self.Q(params),
            self.R(params),
            self.seed,
        )

    def __iter__(self):
        raise NotImplementedError("Cannot iterate over a model generator.")


class ModelGenAuto(ModelGen):
    """An autograd-compatible linear Gaussian state space model generator."""

    def __init__(
        self,
        x_0: Callable,
        A: Callable,
        H: Callable,
        Q: np.ndarray,
        R: np.ndarray,
        seed: int = 1234,
    ):
        super().__init__(x_0, A, H, lambda x: Q, lambda x: R, seed)

    @classmethod
    def from_model_gen(cls, model_gen: ModelGen, params: np.ndarray) -> "ModelGenAuto":
        """Generate an autograd-compatible model generator from a model generator."""
        return cls(
            model_gen.x_0,
            model_gen.A,
            model_gen.H,
            model_gen.Q(params),
            model_gen.R(params),
            model_gen.seed,
        )


k_ballistic_model = ModelGen(
    lambda x: np.zeros(4),  # x_0
    lambda x: np.array(  # A
        [
            [1.0, 0.0, x[0], 0.0],
            [0.0, 1.0, 0.0, x[0]],
            [0.0, 0.0, 0.99, 0.0],
            [0.0, 0.0, 0.0, 0.99],
        ]
    ),
    lambda x: np.array(  # H
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    ),
    lambda x: np.array(  # Q
        [
            [x[0] ** 3 / 3, 0.0, x[0] ** 2 / 2, 0.0],
            [0.0, x[0] ** 3 / 3, 0.0, x[0] ** 2 / 2],
            [x[0] ** 2 / 2, 0.0, x[0], 0.0],
            [0.0, x[0] ** 2 / 2, 0.0, x[0]],
        ]
    ),
    lambda x: np.array(  # R
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    ),
)  # x = [k]

simple_ballistic_model = ModelGenAuto.from_model_gen(
    k_ballistic_model, np.array([0.04])
)

resonator_model = ModelGen(
    lambda x: np.zeros(2),  # x_0
    lambda x: np.array(  # A
        [
            [np.cos(x[0]), np.sin(x[0]) / x[0]],
            [-x[0] * np.sin(x[0]), np.cos(x[0])],
        ]
    ),
    lambda x: np.array([[1.0, 0.0]]),  # H
    lambda x: np.array(  # Q
        [
            [
                x[1] * (x[0] - np.cos(x[0]) * np.sin(x[0])) / (2 * x[0] ** 3),
                x[1] * np.sin(x[0]) ** 2 / (2 * x[0] ** 2),
            ],
            [
                x[1] * np.sin(x[0]) ** 2 / (2 * x[0] ** 2),
                x[1] * (x[0] + np.cos(x[0]) * np.sin(x[0])) / (2 * x[0]),
            ],
        ]
    ),
    lambda x: np.array([[1.0]]),  # R
)  # x = [omega, q^c]
