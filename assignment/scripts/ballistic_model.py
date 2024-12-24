"""A linear ballistic model."""

import numpy as np
from assignment.models import Model
from assignment.filters import KalmanFilter, ParticleFilter

SEED = 1234
MAX_ITER = 1000
N_PARTICLES = 1000


def main():
    k = 0.04

    A = np.block(
        [
            [np.eye(2), k * np.eye(2)],
            [np.zeros((2, 2)), 0.99 * np.eye(2)],
        ]
    )
    Q = np.block(
        [
            [k**3 / 3 * np.eye(2), k**2 / 2 * np.eye(2)],
            [k**2 / 2 * np.eye(2), k * np.eye(2)],
        ]
    )
    H = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )
    R = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    model = Model(np.zeros(4), A, Q, H, R, SEED)
    KalmanFilter.plot(model, MAX_ITER, 3 * np.ones(4), 10 * np.eye(4), save_fig=True)
    ParticleFilter.plot(
        model,
        MAX_ITER,
        np.random.multivariate_normal(
            np.array([0.0, 0.0, -10.0, -10.0]),
            0.1 * np.eye(4),
            N_PARTICLES,
        ).T,
        save_fig=True,
    )


if __name__ == "__main__":
    main()
