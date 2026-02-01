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
        initial_state: state,
        gamma: float,
    ) -> None:
        self.transition_matrix = transition_matrix
        self.reward_matrix = reward_matrix
        self.terminal_states = terminal_states
        self.initial_state = initial_state
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
    initial_state = (grid_size // 2, grid_size // 2)
    gamma = 0.9

    return transition_matrix, reward_matrix, terminal_states, initial_state, gamma


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
                * (mdp.get_reward(s_) + mdp.gamma * values[s_])
                for s_ in mdp.get_states()
            )

            delta = max(delta, abs(v - values[state]))

        if delta < theta:
            break

    return values


def main():
    t, r, terminal, initial, gamma = generate_mdp_params(4)
    mdp = GridWorldMDP(t, r, terminal, initial, gamma)

    print("We have the following rewards for the MDP:")
    print_rewards(mdp)

    pi = generate_policy(mdp)

    print("Arbitrarily generated policy:")
    print_policy(pi)

    print("Evaluating policy through iterative_policy_evaluation")
    values = iterative_policy_evaluation(mdp, pi)

    print("Values of corresponding policy:")
    print_values(values)


main()
