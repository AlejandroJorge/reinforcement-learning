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
    Q = np.zeros(K_BANDITS)
    N = np.zeros(K_BANDITS, dtype=np.uint64)

    rewards = np.empty(STEPS)

    for step in range(STEPS):
        bandit_idx = select_bandit(Q, epsilon)
        v = calc_bandit(bandits, bandit_idx)

        N[bandit_idx] += 1
        Q[bandit_idx] += (1.0 / (N[bandit_idx])) * (v - Q[bandit_idx])

        rewards[step] = v

    return rewards

def execute_experiment(epsilon: float):
    accum_rewards = np.zeros(STEPS)

    for simulation in range(SIMULATIONS):
        if simulation % 100 == 0:
            print(f"Simulation {simulation} of {SIMULATIONS}")

        bandits = generate_bandits()
        rewards = execute_simulation(bandits, epsilon)
        accum_rewards += rewards

    return accum_rewards / SIMULATIONS

def main():
    result_1 = execute_experiment(0.1)
    result_2 = execute_experiment(0.01)
    result_3 = execute_experiment(0.001)

    _, ax = plt.subplots()
    ax.plot(np.arange(STEPS), result_1, label="epsilon=0.1")
    ax.plot(np.arange(STEPS), result_2, label="epsilon=0.01")
    ax.plot(np.arange(STEPS), result_3, label="epsilon=0.001")
    ax.set_xlabel("steps")
    ax.set_ylabel("avg rewards")
    ax.legend()
    plt.savefig("results")

main()
