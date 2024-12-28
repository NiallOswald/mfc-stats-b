"""Likelihood plotting for a parameterized ballistic model."""

import numpy as np
from alive_progress import alive_it
from assignment.models import k_ballistic_model
from assignment.filters import KalmanFilter, ParticleFilter
from matplotlib import pyplot as plt

MAX_ITER = 1000
N_PARTICLES = 100
TRUE_K = 0.04
K_VALS = 100

base_model = k_ballistic_model(TRUE_K)


def main():
    true_model = base_model([TRUE_K])

    k_vals = np.linspace(0.5 * TRUE_K, 1.5 * TRUE_K, K_VALS)

    kalman_li, particle_li = np.zeros_like(k_vals), np.zeros_like(k_vals)
    for i, k in enumerate(alive_it(k_vals, title="Calculating likelihoods...")):
        kalman_li[i] = KalmanFilter.likelihood(
            true_model,
            base_model([k]),
            MAX_ITER,
            3 * np.ones(4),
            10 * np.eye(4),
        )
        particle_li[i] = ParticleFilter.likelihood(
            true_model,
            base_model([k]),
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
