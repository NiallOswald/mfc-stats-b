"""Maximum likelihood estimation for a ballistic model."""

import numpy as np
from assignment.models import simple_ballistic_model
from assignment.filters import KalmanFilter, ParticleFilter

MAX_ITER = 1000
N_PARTICLES = 1000
TRUE_K = 0.04


def main():
    true_model = simple_ballistic_model([TRUE_K])

    kalman_args = (3 * np.ones(4), 10 * np.eye(4))
    particle_args = (
        np.random.multivariate_normal(true_model.x_0, 0.1 * np.eye(4), N_PARTICLES).T,
    )

    k = KalmanFilter.fit(
        true_model,
        simple_ballistic_model,
        np.array([0.03]),
        kalman_args,
        MAX_ITER,
    )
    print("Estimated k using Kalman filter with autograd:", k)

    k = KalmanFilter.fit(
        true_model,
        simple_ballistic_model,
        np.array([0.03]),
        kalman_args,
        MAX_ITER,
        use_autograd=False,
        bounds=[(0.00, 0.10)],
    )

    print("Estimated k using Kalman filter without autograd:", k)

    k = ParticleFilter.fit(
        true_model,
        simple_ballistic_model,
        np.array([0.03]),
        particle_args,
        MAX_ITER,
        use_autograd=False,
        bounds=[(0.02, 0.06)],  # ParticleFilter requires some assistance
        method="Powell",  # Recommended for ParticleFilter to mitigate noise
    )

    print("Estimated k using Particle filter without autograd:", k)


if __name__ == "__main__":
    main()
