import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

SEED = 42
N = 100000
THRESHOLD = 4

TRUE_VAL = 1 - stats.norm.cdf(THRESHOLD)


def monte_carlo(n):
    rng = np.random.default_rng(SEED)
    data = rng.normal(0, 1, n)

    return np.cumsum(data > THRESHOLD) / np.arange(1, n + 1)


def importance_sampler(n):
    rng = np.random.default_rng(SEED)
    data = rng.normal(THRESHOLD, 1, n)

    return np.cumsum(
        (data > THRESHOLD)
        * (stats.norm.pdf(data) / stats.norm.pdf(data, loc=THRESHOLD))
    ) / np.arange(1, n + 1)


plt.plot([1, N], [TRUE_VAL, TRUE_VAL], "b--", label="True Value", alpha=1, linewidth=2)
plt.plot(np.arange(1, N + 1), monte_carlo(N), "k-", label="MC estimate")
plt.plot(np.arange(1, N + 1), importance_sampler(N), "r-", label="IS estimate")
plt.yscale("log")
plt.xlabel("Number of samples")
plt.ylabel("Estimate")
plt.legend()
plt.tight_layout()
plt.show()
