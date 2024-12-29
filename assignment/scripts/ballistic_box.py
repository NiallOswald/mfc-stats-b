"""MLE box plots for a parameterized ballistic model."""

import numpy as np
from alive_progress import alive_it
from assignment.models import k_ballistic_model
from assignment.filters import ParticleFilter
from matplotlib import pyplot as plt

MAX_ITER = 1000
N_PARTICLES = 1000
N_FILTERS = 10
TRUE_K = 0.04


def main():
    true_model = k_ballistic_model([TRUE_K])

    particle_args = (
        np.random.multivariate_normal(true_model.x_0, 0.1 * np.eye(4), N_PARTICLES).T,
    )

    k_estimates = np.zeros((N_FILTERS, 1))
    for i in alive_it(range(N_FILTERS), title="Running filters..."):
        k_estimates[i] = ParticleFilter.fit(
            true_model,
            k_ballistic_model,  # Full model may encounter warnings
            np.array([0.03]),
            particle_args,
            MAX_ITER,
            use_autograd=False,
            bounds=[(0.02, 0.06)],  # ParticleFilter requires some assistance
            method="Powell",  # Recommended for ParticleFilter to mitigate noise
        )

    plt.boxplot(k_estimates, tick_labels=[r"$k$"])
    plt.ylabel("Parameter estimate")
    plt.title("MLE estimates using ParticleFilter")
    plt.tight_layout()
    plt.savefig("mle_box.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
