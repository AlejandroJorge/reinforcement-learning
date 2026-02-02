import math
import random
from typing import Literal, final


action = Literal["up"] | Literal["down"] | Literal["right"] | Literal["left"]
state = tuple[int, int]


@final
class GridWorldMDP:
    """An MDP defined by the following p(s' | s,a) and rewards based on reached state: s -> r"""

    def __init__(
        self,
        transition_matrix: dict[state, dict[action, dict[state, float]]],
        reward_matrix: dict[state, float],
        terminal_states: set[state],
        gamma: float,
    ) -> None:
        self.transition_matrix = transition_matrix
        self.reward_matrix = reward_matrix
        self.terminal_states = terminal_states
        self.gamma = gamma

    def get_reward(self, s: state) -> float:
        return self.reward_matrix[s]

    def get_p(self, s: state, a: action, s_: state) -> float:
        return self.transition_matrix[s][a].get(s_, 0.0)

    def get_states(self) -> set[state]:
        return set(self.transition_matrix.keys())

    def get_terminal_states(self) -> set[state]:
        return set(self.terminal_states)

    def get_non_terminal_states(self) -> set[state]:
        return self.get_states() - self.get_terminal_states()


def generate_mdp_params(grid_size: int):
    transition_matrix: dict[state, dict[action, dict[state, float]]] = {}
    actions: list[action] = ["up", "down", "left", "right"]

    for r in range(1, grid_size + 1):
        for c in range(1, grid_size + 1):
            s = (r, c)
            transition_matrix[s] = {}

            for a in actions:
                next_r, next_c = r, c

                if a == "up":
                    next_r = r - 1
                elif a == "down":
                    next_r = r + 1
                elif a == "left":
                    next_c = c - 1
                elif a == "right":
                    next_c = c + 1

                if not (1 <= next_r <= grid_size):
                    next_r = r
                if not (1 <= next_c <= grid_size):
                    next_c = c

                s_ = (next_r, next_c)

                transition_matrix[s][a] = {s_: 1.0}

    reward_matrix: dict[state, float] = {}
    for i in range(1, grid_size + 1):
        for j in range(1, grid_size + 1):
            reward_matrix[(i, j)] = -1
    reward_matrix[(1, 1)] = 0
    reward_matrix[(grid_size, grid_size)] = 0

    terminal_states = set([(1, 1), (grid_size, grid_size)])
    gamma = 0.9

    return transition_matrix, reward_matrix, terminal_states, gamma


policy = dict[state, action]
policy_char_mapping: dict[action, str] = {
    "left": "<",
    "right": ">",
    "up": "^",
    "down": "v",
}


def generate_policy(mdp: GridWorldMDP) -> policy:
    pi: policy = {}
    valid_actions: list[action] = ["up", "down", "left", "right"]

    for s in mdp.get_states():
        pi[s] = random.choice(valid_actions)

    return pi


def get_optimal_policy_from_val(mdp: GridWorldMDP, values: dict[state, float]):
    pi: policy = {}
    for s in mdp.get_states():
        max_q = -math.inf
        actions: list[action] = ["up", "down", "right", "left"]
        for a in actions:
            curr_q = sum(
                mdp.get_p(s, a, s_) * (mdp.get_reward(s) + mdp.gamma * values[s_])
                for s_ in mdp.get_states()
            )
            if curr_q > max_q:
                max_q = curr_q
                pi[s] = a

    for s in mdp.get_terminal_states():
        pi[s] = "up"  # doesn't matter

    return pi


def print_policy(pi: policy):
    policy_grid: dict[int, dict[int, str]] = {}

    grid_height, grid_width = 0, 0
    for i, j in pi.keys():
        if i not in policy_grid:
            policy_grid[i] = {}

        policy_grid[i][j] = policy_char_mapping[pi[(i, j)]]
        grid_height = max(grid_height, i)
        grid_width = max(grid_width, j)

    for i in range(1, grid_height + 1):
        for j in range(1, grid_width + 1):
            print(f"{policy_grid[i][j]}", end=" ")
        print()


def print_rewards(mdp: GridWorldMDP):
    grid_height, grid_width = 0, 0
    for i, j in mdp.reward_matrix.keys():
        grid_height = max(grid_height, i)
        grid_width = max(grid_width, j)

    for i in range(1, grid_height + 1):
        for j in range(1, grid_width + 1):
            print(f"{mdp.reward_matrix[(i, j)]:7.3f}", end=" ")
        print()


def print_values(values: dict[state, float]):
    grid_height, grid_width = 0, 0
    for i, j in values.keys():
        grid_height = max(grid_height, i)
        grid_width = max(grid_width, j)

    for i in range(1, grid_height + 1):
        for j in range(1, grid_width + 1):
            print(f"{values[(i, j)]:7.3f}", end=" ")
        print()


# From Sutton & Barto p75 (2018)
def iterative_policy_evaluation(mdp: GridWorldMDP, pi: policy, theta: float = 0.1):
    # Initialize V(s) for all S+ arbitrarily except for V(terminal) = 0
    values: dict[state, float] = {}
    for state in mdp.get_terminal_states():
        values[state] = 0
    for state in mdp.get_non_terminal_states():
        values[state] = random.random() * 10

    while True:
        delta = 0.0
        for state in mdp.get_non_terminal_states():  # S, not S+
            v = values[state]

            # Adapted for a deterministic policy
            values[state] = sum(
                mdp.get_p(state, pi[state], s_)
                * (mdp.get_reward(state) + mdp.gamma * values[s_])
                for s_ in mdp.get_states()
            )

            delta = max(delta, abs(v - values[state]))

        if delta < theta:
            break

    return values


# From Sutton & Barto p80 (2018)
def policy_iteration(mdp: GridWorldMDP) -> policy:
    pi = generate_policy(mdp)
    values: dict[state, float] = {}

    while True:
        values = iterative_policy_evaluation(mdp, pi)

        policy_stable = True
        for state in mdp.get_non_terminal_states():
            old_action = pi[state]

            # Adapted
            max_q = -math.inf
            actions: list[action] = ["up", "down", "right", "left"]
            for a in actions:
                curr_q = sum(
                    mdp.get_p(state, a, s_)
                    * (mdp.get_reward(state) + mdp.gamma * values[s_])
                    for s_ in mdp.get_states()
                )
                if curr_q > max_q:
                    pi[state] = a
                    max_q = curr_q

            if old_action != pi[state]:
                policy_stable = False

        if policy_stable:
            break

    return pi


# From Sutton & Barto p83 (2018)
def value_iteration(mdp: GridWorldMDP, theta: float = 0.1) -> dict[state, float]:
    values: dict[state, float] = {}
    for state in mdp.get_terminal_states():
        values[state] = 0
    for state in mdp.get_non_terminal_states():
        values[state] = random.random() * 10

    while True:
        delta = 0.0

        for state in mdp.get_non_terminal_states():
            v = values[state]

            actions: list[action] = ["up", "down", "left", "right"]
            values[state] = max(
                sum(
                    mdp.get_p(state, a, s_)
                    * (mdp.get_reward(state) + mdp.gamma * values[s_])
                    for s_ in mdp.get_states()
                )
                for a in actions
            )

            delta = max(delta, abs(v - values[state]))

        if delta < theta:
            break

    return values


def main():
    t, r, terminal, gamma = generate_mdp_params(8)
    mdp = GridWorldMDP(t, r, terminal, gamma)

    print("We have the following rewards for the MDP:")
    print_rewards(mdp)

    pi = generate_policy(mdp)

    print("Arbitrarily generated policy:")
    print_policy(pi)

    print("Evaluating policy through iterative_policy_evaluation")
    values = iterative_policy_evaluation(mdp, pi)

    print("Values of corresponding policy:")
    print_values(values)

    print("---------------------------------------------------------")

    print("Optimizing policy through policy_iteration")
    pi = policy_iteration(mdp)

    print("Evaluating policy through iterative_policy_evaluation")
    values = iterative_policy_evaluation(mdp, pi)

    print("Optimized policy:")
    print_policy(pi)

    print("Values of optimized policy:")
    print_values(values)

    print("---------------------------------------------------------")

    print("Optimizing values through value_iteration")
    values = value_iteration(mdp)

    print("Values of optimized policy:")
    print_values(values)

    print("Calculating policy on optimized values")
    pi = get_optimal_policy_from_val(mdp, values)

    print("Optimized policy:")
    print_policy(pi)


main()
