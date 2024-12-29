"""Likelihood plotting for a parameterized resonator model."""

import numpy as np
from alive_progress import alive_it
from assignment.models import resonator_model
from assignment.filters import KalmanFilter, ParticleFilter
from matplotlib import pyplot as plt

MAX_ITER = 500
N_PARTICLES = 100
TRUE_PARAMS = [0.50, 0.010]
PARAM_LABELS = [r"$\omega$", r"$q^{c}$"]
SEARCH_SPACE = [
    (0.45, 0.55),
    (0.0025, 0.0175),
]
SEARCH_POINTS = 100


def main():
    fig, axs = plt.subplots(1, len(TRUE_PARAMS), figsize=(6 * len(TRUE_PARAMS), 6))

    true_model = resonator_model(TRUE_PARAMS)

    for search_param in range(len(TRUE_PARAMS)):
        search_space = np.linspace(*SEARCH_SPACE[search_param], SEARCH_POINTS)
        kalman_li, particle_li = np.zeros_like(search_space), np.zeros_like(
            search_space
        )

        for i, param in enumerate(
            alive_it(search_space, title="Calculating likelihoods...")
        ):
            params = [
                param if search_param == j else TRUE_PARAMS[j]
                for j in range(len(TRUE_PARAMS))
            ]

            kalman_li[i] = KalmanFilter.likelihood(
                true_model,
                resonator_model(params),
                MAX_ITER,
                np.zeros(2),
                1.0 * np.eye(2),
            )
            particle_li[i] = ParticleFilter.likelihood(
                true_model,
                resonator_model(params),
                MAX_ITER,
                np.random.multivariate_normal(
                    true_model.x_0,
                    1.0 * np.eye(2),
                    N_PARTICLES,
                ).T,
            )

        axs[search_param].plot(search_space, kalman_li, label="Kalman filter")
        axs[search_param].plot(search_space, particle_li, label="Particle filter")
        axs[search_param].set_xlabel(PARAM_LABELS[search_param])
        axs[search_param].set_ylabel("log likelihood")
        axs[search_param].set_title(
            f"Log likelihood for different values of {PARAM_LABELS[search_param]}"
        )

    plt.tight_layout()
    plt.legend()
    plt.savefig("likelihood.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
