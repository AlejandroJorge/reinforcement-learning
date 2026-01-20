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

# k=5 bandits
Q = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
N = np.array([0, 0, 0, 0, 0])

epsilon = 0.1

# Bandit distributions initialization
means = np.array([0, 1, -1, 0.5, -0.5])
sd = 0.1

# Bandit function: 1..5 -> R
def bandit(action: int) -> float:
    return np.random.normal(loc=means[action-1], scale=sd, size=None)

def get_action(epsilon: float) -> int:
    if np.random.random() < epsilon:
        return np.argmax(Q, keepdims=True) + 1
    else:
        return np.random.randint(1,5 + 1)

def main():
    # print(f"Means: {means}, sd: {sd}")
    #
    # for _ in range(10):
    #     action = randint(1,5)
    #     b = bandit(action)
    #     print(f"For action {action} the bandit resulted in: {b}")

    steps = 1000
    for _ in range(steps):
        action = get_action(epsilon)
        v = bandit(action)
        N[action-1] += 1
        Q[action-1] += (1.0 / (N[action-1])) * (v - Q[action-1])

    print(f"Distribution means: {means}")
    print(f"Final values of Q: {Q}")

main()
