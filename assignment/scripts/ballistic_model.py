"""A linear ballistic model."""

import numpy as np
from assignment.models import Model
from assignment.filters import KalmanFilter

SEED = 1234


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
    KalmanFilter.plot(model, 3 * np.ones(4), 10 * np.eye(4), 1000)


if __name__ == "__main__":
    main()
