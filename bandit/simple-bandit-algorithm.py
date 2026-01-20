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

K_BANDITS = 10
EPSILON = 0.1
STEPS = 10_000

# k=5 bandits
Q = np.zeros(K_BANDITS)
N = np.zeros(K_BANDITS, dtype=np.uint64)

# Bandit distributions initialization
means = np.array(np.random.normal(loc=0, scale=5, size=10))
sd = 0.5


# Bandit function: 1..5 -> R
def bandit(action_idx: int) -> float:
    return np.random.normal(loc=means[action_idx], scale=sd, size=None)


def get_action_idx() -> int:
    if np.random.random() > EPSILON:
        return np.argmax(Q, keepdims=True)
    else:
        return np.random.randint(0, K_BANDITS)


def main():
    for _ in range(STEPS):
        action_idx = get_action_idx()
        v = bandit(action_idx)
        N[action_idx] += 1
        Q[action_idx] += (1.0 / (N[action_idx])) * (v - Q[action_idx])

    print(f"Distribution means: {means}")
    print(f"Final values of Q: {Q}")


main()
