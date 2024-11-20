import numpy as np
import matplotlib.pyplot as plt

SEED = 42
N = 100000

rng = np.random.default_rng(SEED)

# Generate N random samples in the unit square
samples = rng.random((N, 2))


def transformation_sampler():
    while True:
        x, y = rng.random(2)
        yield (np.sqrt(x) * np.cos(2 * np.pi * y), np.sqrt(x) * np.sin(2 * np.pi * y))


def rejection_sampler():
    while True:
        x, y = 2 * rng.random(2) - 1
        if np.linalg.norm((x, y)) < 1:
            yield (x, y)


transformation_vals = np.array([next(transformation_sampler()) for _ in range(N)])
rejection_vals = np.array([next(rejection_sampler()) for _ in range(N)])

# Plot the samples
plt.scatter(
    transformation_vals[:, 0], transformation_vals[:, 1], label="Transformation sampler"
)
plt.scatter(rejection_vals[:, 0], rejection_vals[:, 1], label="Rejection sampler")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.legend()
plt.show()

# Plot the distribution of the samples
plt.hist2d(transformation_vals[:, 0], transformation_vals[:, 1], bins=50)
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.title("Transformation sampler")
plt.colorbar(label="Frequency")
plt.show()

plt.hist2d(rejection_vals[:, 0], rejection_vals[:, 1], bins=50)
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.title("Rejection sampler")
plt.colorbar(label="Frequency")
plt.show()

# Plot the distribution of the x values
plt.hist(transformation_vals[:, 0], bins=50, alpha=0.5, label="Transformation sampler")
plt.hist(rejection_vals[:, 0], bins=50, alpha=0.5, label="Rejection sampler")
plt.xlabel(r"$x$")
plt.ylabel("Frequency")
plt.title(r"$x$-marginal the samples")
plt.legend()
plt.show()
