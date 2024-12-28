"""A linear ballistic model."""

import numpy as np
from assignment.models import k_ballistic_model
from assignment.filters import KalmanFilter, ParticleFilter

MAX_ITER = 1000
N_PARTICLES = 1000
K = 0.04


def main():
    model = k_ballistic_model(K)([K])
    KalmanFilter.plot(model, MAX_ITER, 3 * np.ones(4), 10 * np.eye(4), save_fig=True)
    ParticleFilter.plot(
        model,
        MAX_ITER,
        np.random.multivariate_normal(
            model.x_0,
            0.1 * np.eye(4),
            N_PARTICLES,
        ).T,
        save_fig=True,
    )


if __name__ == "__main__":
    main()
