import numpy as np
import matplotlib.pyplot as plt


def phi(x):
    return np.sqrt((1 - x**2))


I = np.pi / 4  # true value

N_max = 10000  # go up to 10,000 samples

U = np.random.uniform(0, 1, N_max)
I_est = np.zeros(N_max - 1)  # this is longer than we need
I_var = np.zeros(N_max - 1)

fig = plt.figure(figsize=(10, 5))

k = 0

K = np.array([])

# We are not computing for every N for efficiency

for N in range(1, N_max, 5):

    I_est[k] = 0  # Your mean estimate here
    I_var[k] = 0  # Your variance estimate here

    k = k + 1  # We index estimators with k as we jump N by 5
    K = np.append(K, N)

plt.plot(K, I_est[0:k], "k-", label="MC estimate")
plt.plot(K, I_est[0:k] + np.sqrt(I_var[0:k]), "r", label=r"$\sigma$", alpha=1)
plt.plot(K, I_est[0:k] - np.sqrt(I_var[0:k]), "r", alpha=1)
plt.plot([0, N_max], [I, I], "b--", label="True Value", alpha=1, linewidth=2)
plt.legend()
plt.xlabel("Number of samples")
plt.ylabel("Estimate")
plt.xlim([0, N_max])
plt.show()
