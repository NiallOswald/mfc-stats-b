"""Likelihood plotting for a parameterized ballistic model."""

import numpy as np
from alive_progress import alive_it
from assignment.models import Model
from assignment.filters import KalmanFilter, ParticleFilter
from matplotlib import pyplot as plt

SEED = 1234
MAX_ITER = 1000
N_PARTICLES = 1000
TRUE_K = 0.04
r = 5


def A(k):
    return np.block(
        [
            [np.eye(2), k * np.eye(2)],
            [np.zeros((2, 2)), 0.99 * np.eye(2)],
        ]
    )


def Q(k):
    return np.block(
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

R = r * np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
    ]
)


def k_model(k):
    return Model(np.zeros(4), A(k), Q(k), H, R, SEED)


def main():
    true_model = k_model(TRUE_K)

    k_vals = np.linspace(0.02, 0.06, 101)

    kalman_li, particle_li = np.zeros_like(k_vals), np.zeros_like(k_vals)
    for i, k in enumerate(alive_it(k_vals, title="Calculating likelihoods...")):
        kalman_li[i] = KalmanFilter.likelihood(
            true_model, k_model(k), MAX_ITER, 3 * np.ones(4), 10 * np.eye(4)
        )
        particle_li[i] = ParticleFilter.likelihood(
            true_model,
            k_model(k),
            MAX_ITER,
            np.random.multivariate_normal(
                true_model.x_0,
                0.1 * np.eye(4),
                N_PARTICLES,
            ).T,
        )

    plt.figure()
    plt.plot(k_vals, kalman_li, label="Kalman filter")
    plt.plot(k_vals, particle_li, label="Particle filter")
    plt.xlabel(r"$k$")
    plt.ylabel("log likelihood")
    plt.legend()
    plt.savefig("likelihood.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
