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
# We use bias from high initial values to
# converge faster:
#   e.g. Q(a) <- 5

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

K_BANDITS = 10
STEPS = 1000
SIMULATIONS = 2000


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


def execute_simulation(bandits: list[tuple[float, int]], epsilon: float):
    Q_std = np.zeros(K_BANDITS)
    N_std = np.zeros(K_BANDITS, dtype=np.uint64)

    Q_optimistic = np.array([5.0 for _ in range(K_BANDITS)])
    N_optimistic = np.zeros(K_BANDITS, dtype=np.uint64)

    rewards_std = np.empty(STEPS)
    rewards_optimistic = np.empty(STEPS)

    for step in range(STEPS):
        # Standard simple way
        bandit_idx = select_bandit(Q_std, epsilon)
        v = calc_bandit(bandits, bandit_idx)

        N_std[bandit_idx] += 1
        Q_std[bandit_idx] += (1.0 / (N_std[bandit_idx])) * (v - Q_std[bandit_idx])

        rewards_std[step] = v

        # Optimistic initial values
        bandit_idx = select_bandit(Q_optimistic, epsilon)
        v = calc_bandit(bandits, bandit_idx)

        N_optimistic[bandit_idx] += 1
        Q_optimistic[bandit_idx] += (1.0 / (N_optimistic[bandit_idx])) * (
            v - Q_optimistic[bandit_idx]
        )

        rewards_optimistic[step] = v

    return rewards_std, rewards_optimistic


def execute_experiment(epsilon: float):
    accum_rewards_std = np.zeros(STEPS)
    accum_rewards_optimistic = np.zeros(STEPS)

    for simulation in range(SIMULATIONS):
        if simulation % 100 == 0:
            print(f"Simulation {simulation} of {SIMULATIONS}")

        bandits = generate_bandits()
        rewards_std, rewards_optimistic = execute_simulation(bandits, epsilon)
        accum_rewards_std += rewards_std
        accum_rewards_optimistic += rewards_optimistic

    return (accum_rewards_std / SIMULATIONS, accum_rewards_optimistic / SIMULATIONS)


def main():
    result_std, result_optimistic = execute_experiment(0.1)

    _, ax = plt.subplots()
    ax.plot(np.arange(STEPS), result_std, label="simple")
    ax.plot(np.arange(STEPS), result_optimistic, label="optimistic initial values")
    ax.set_xlabel("steps")
    ax.set_ylabel("avg rewards")
    ax.legend()
    plt.savefig("results")


main()
