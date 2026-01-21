# From Sutton & Barto (2018)
#
# Initialization:
#   Q(a) <- 0
#   N(a) <- 0
#
# Loop:
#   A <- argmax_a Q(a)  ...with probability 1-r
#   A <- random a       ...with probability r
#   R <- bandit(A)
#   N(A) <- N(A) + 1
#   Q(A) <- Q(A) + 1/N(A) * (R - Q(A))
#
# For non stationary problems the update function for Q(A)
# is changed to the following one:
#   Q(A) <- Q(A) + alpha * (R - Q(A))
#   with constant alpha, for example alpha=0.1

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

K_BANDITS = 10
STEPS = 10_000
SIMULATIONS = 1000


def generate_bandits(k: int = 10):
    return [(np.random.normal(loc=0, scale=1, size=None), 1) for _ in range(k)]


def calc_bandit(bandits: list[tuple[float, int]], bandit_idx: int):
    mean, sd = bandits[bandit_idx]
    return np.random.normal(loc=mean, scale=sd, size=None)


def select_bandit(Q: NDArray[np.float64], epsilon: float):
    if np.random.random() > epsilon:
        return int(np.argmax(Q))
    else:
        return np.random.randint(0, K_BANDITS)


def execute_simulation(bandits: list[tuple[float, int]], epsilon: float, alpha: float):
    Q_sample_avg = np.zeros(K_BANDITS)
    N_sample_avg = np.zeros(K_BANDITS, dtype=np.uint64)

    Q_constant = np.zeros(K_BANDITS)

    rewards_sample_avg = np.empty(STEPS)
    rewards_constant = np.empty(STEPS)

    for step in range(STEPS):
        # Sample avg update function tracing
        bandit_idx = select_bandit(Q_sample_avg, epsilon)
        v = calc_bandit(bandits, bandit_idx)

        N_sample_avg[bandit_idx] += 1
        Q_sample_avg[bandit_idx] += (1.0 / (N_sample_avg[bandit_idx])) * (
            v - Q_sample_avg[bandit_idx]
        )

        rewards_sample_avg[step] = v

        # Constant update function tracing
        bandit_idx = select_bandit(Q_constant, epsilon)
        v = calc_bandit(bandits, bandit_idx)

        Q_constant[bandit_idx] += alpha * (v - Q_constant[bandit_idx])

        rewards_constant[step] = v

        # Random walking
        deltas = np.random.normal(loc=0, scale=0.1, size=10)
        for i in range(K_BANDITS):
            bandit_mean, bandit_sd = bandits[i]
            bandits[i] = (bandit_mean + deltas[i], bandit_sd)

    return rewards_sample_avg, rewards_constant


def execute_experiment(epsilon: float, alpha: float):
    accum_rewards_sample_avg = np.zeros(STEPS)
    accum_rewards_constant = np.zeros(STEPS)

    for simulation in range(SIMULATIONS):
        if simulation % 100 == 0:
            print(f"Simulation {simulation} of {SIMULATIONS}")

        bandits = generate_bandits()
        rewards_sample_avg, rewards_constant = execute_simulation(
            bandits, epsilon, alpha
        )
        accum_rewards_sample_avg += rewards_sample_avg
        accum_rewards_constant += rewards_constant

    return (
        accum_rewards_sample_avg / SIMULATIONS,
        accum_rewards_constant / SIMULATIONS,
    )


def main():
    res_sample_avg, res_constant = execute_experiment(0.1, 0.1)

    _, ax = plt.subplots()
    ax.plot(np.arange(STEPS), res_sample_avg, label="sample avg")
    ax.plot(np.arange(STEPS), res_constant, label="constant with alpha=0.1")
    ax.set_xlabel("steps")
    ax.set_ylabel("avg rewards")
    ax.legend()
    plt.savefig("results")


main()
