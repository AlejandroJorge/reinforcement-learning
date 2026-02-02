import random
from typing import Literal, final

State = tuple[int, int]
Action = Literal["up"] | Literal["down"] | Literal["right"] | Literal["left"]
Policy = dict[State, Action]

valid_actions: list[Action] = ["up", "down", "right", "left"]


@final
class GridEnvironment:
    def __init__(
        self,
        rewards_matrix: dict[State, float],
        terminal_states: list[State],
        grid_width: int,
        grid_height: int,
        gamma: float,
    ) -> None:
        self.rewards_matrix = rewards_matrix
        self.terminal_states = terminal_states
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.gamma = gamma

    def act(self, state: State, action: Action) -> tuple[State, float, bool]:
        state_: State
        match action:
            case "up":
                state_ = random.choices(
                    [
                        (state[0] - 1, state[1]),
                        (state[0], state[1] - 1),
                        (state[0] + 1, state[1]),
                    ],
                    [0.1, 0.8, 0.1],
                    k=1,
                )[0]
            case "down":
                state_ = random.choices(
                    [
                        (state[0] + 1, state[1]),
                        (state[0], state[1] + 1),
                        (state[0] - 1, state[1]),
                    ],
                    [0.1, 0.8, 0.1],
                    k=1,
                )[0]
            case "right":
                state_ = random.choices(
                    [
                        (state[0], state[1] - 1),
                        (state[0] + 1, state[1]),
                        (state[0], state[1] + 1),
                    ],
                    [0.1, 0.8, 0.1],
                    k=1,
                )[0]
            case "left":
                state_ = random.choices(
                    [
                        (state[0], state[1] + 1),
                        (state[0] - 1, state[1]),
                        (state[0], state[1] - 1),
                    ],
                    [0.1, 0.8, 0.1],
                    k=1,
                )[0]

        state_ = (
            min(max(0, state_[0]), self.grid_height - 1),
            min(max(0, state_[1]), self.grid_width - 1),
        )
        reward = self.rewards_matrix[state_]
        is_terminal = state_ in self.terminal_states

        return state_, reward, is_terminal

    def generate_episode(
        self, policy: Policy, initial_state: State = (0, 0)
    ) -> list[Episode]:
        episodes: list[Episode] = []

        curr_state_ = initial_state
        while True:
            curr_state = curr_state_
            curr_action = policy[curr_state]

            curr_state_, curr_reward, is_terminal = self.act(curr_state, curr_action)

            episodes.append(Episode(curr_state, curr_action, curr_reward, curr_state_))

            if is_terminal:
                break

        return episodes


def generate_gridenv_parameters():
    grid_width = 5
    grid_height = 4
    gamma = 0.9
    terminal_states = [
        (grid_height - 1, grid_width - 1),
        (0, grid_width - 1),
        (grid_height - 1, 0),
    ]

    rewards_matrix: dict[State, float] = {}
    for i in range(grid_height):
        for j in range(grid_width):
            rewards_matrix[(i, j)] = -1.0
    rewards_matrix[(grid_height // 2, grid_width // 2)] = 5.0

    return rewards_matrix, terminal_states, grid_width, grid_height, gamma


def generate_policy(env: GridEnvironment):
    policy: dict[State, Action] = {}
    for i in range(env.grid_height):
        for j in range(env.grid_width):
            policy[(i, j)] = random.choice(valid_actions)

    return policy


@final
class Episode:
    def __init__(
        self, state: State, action: Action, reward: float, state_: State
    ) -> None:
        self.state = state
        self.action = action
        self.reward = reward
        self.state_ = state_


# From Sutton & Barto p92 (2018)
def first_visit_mc_prediction(
    env: GridEnvironment, policy: Policy, episodes: int = 1_000
) -> dict[State, float]:
    values: dict[State, float] = {}
    returns: dict[State, list[float]] = {}

    for _ in range(episodes):
        curr_episode = env.generate_episode(policy)
        episode_states = [step.state for step in curr_episode]
        g = 0
        curr_episode.reverse()
        for idx, step in enumerate(curr_episode):
            g = env.gamma * g + step.reward
            if step.state in episode_states[:idx]:
                continue

            if not returns.get(step.state):
                returns[step.state] = []

            returns[step.state].append(g)
            values[step.state] = sum(returns[step.state]) / len(returns[step.state])

    return values


def print_values(values: dict[State, float], grid_height: int, grid_width: int):
    for i in range(grid_height):
        for j in range(grid_width):
            print(f"{values.get((i, j), 0.0):7.3f}", end=" ")
        print()


def print_rewards_matrix(
    rewards_matrix: dict[State, float], grid_height: int, grid_width: int
):
    for i in range(grid_height):
        for j in range(grid_width):
            print(f"{rewards_matrix.get((i, j), 0.0):7.3f}", end=" ")
        print()


policy_char_mapping: dict[Action, str] = {
    "left": "<",
    "right": ">",
    "up": "^",
    "down": "v",
}


def print_policy(pi: Policy, grid_height: int, grid_width: int):
    policy_grid: dict[int, dict[int, str]] = {}

    for i, j in pi.keys():
        if i not in policy_grid:
            policy_grid[i] = {}

        policy_grid[i][j] = policy_char_mapping[pi[(i, j)]]

    for i in range(grid_height):
        for j in range(grid_width):
            print(f"{policy_grid[i][j]}", end=" ")
        print()


def main():
    rewards_matrix, terminal_states, grid_width, grid_height, gamma = (
        generate_gridenv_parameters()
    )
    environment = GridEnvironment(
        rewards_matrix, terminal_states, grid_width, grid_height, gamma
    )
    print("Reward matrix of environment:")
    print_rewards_matrix(
        environment.rewards_matrix, environment.grid_height, environment.grid_width
    )

    pi = generate_policy(environment)
    print("Policy to evaluate:")
    print_policy(pi, environment.grid_height, environment.grid_width)

    values = first_visit_mc_prediction(environment, pi)
    print("Values from evaluating previous policy:")
    print_values(values, environment.grid_height, environment.grid_width)


main()
