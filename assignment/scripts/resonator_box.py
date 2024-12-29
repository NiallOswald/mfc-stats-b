"""MLE box plots for a parameterized resonator model."""

import numpy as np
from alive_progress import alive_it
from assignment.models import resonator_model
from assignment.filters import ParticleFilter
from matplotlib import pyplot as plt

MAX_ITER = 500
N_PARTICLES = 1000
N_FILTERS = 10
TRUE_PARAMS = [0.50, 0.010]
PARAM_LABELS = [r"$\omega$", r"$q^{c}$"]
SEARCH_SPACE = [
    (0.49, 0.51),  # Tight bounds are needed for omega
    (0.0025, 0.0175),
]
PARAM_INIT = np.array([0.495, 0.005])


def main():
    true_model = resonator_model(TRUE_PARAMS)

    particle_args = (
        np.random.multivariate_normal(true_model.x_0, 1.0 * np.eye(2), N_PARTICLES).T,
    )

    param_estimates = np.zeros((N_FILTERS, len(TRUE_PARAMS)))
    for i in alive_it(range(N_FILTERS), title="Running filters..."):
        param_estimates[i] = ParticleFilter.fit(
            true_model,
            resonator_model,  # Full model may encounter warnings
            PARAM_INIT,
            particle_args,
            MAX_ITER,
            use_autograd=False,
            bounds=SEARCH_SPACE,  # ParticleFilter requires some assistance
            method="Powell",  # Recommended for ParticleFilter to mitigate noise
        )

    fig, axs = plt.subplots(1, len(TRUE_PARAMS), figsize=(8, 6))
    for i in range(len(TRUE_PARAMS)):
        axs[i].boxplot(param_estimates[:, i], tick_labels=[PARAM_LABELS[i]])
        axs[i].set_ylim(SEARCH_SPACE[i])
        axs[i].set_ylabel("Parameter estimate")
        axs[i].set_title(f"MLE estimates for {PARAM_LABELS[i]}")

    plt.tight_layout()
    plt.savefig("mle_box.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
