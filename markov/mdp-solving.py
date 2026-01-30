from abc import ABC, abstractmethod
from typing import final, override


class MDP(ABC):
    """
    Base class for an MDP defined by p(s',r|s,a) and rewards defined by r(s,a,s')
    """

    gamma: float = 0.9

    @abstractmethod
    def get_p(self, s_: str, r: float, s: str, a: str) -> float:
        pass

    @abstractmethod
    def get_r(self, s: str, a: str, s_: str) -> float:
        pass

    @abstractmethod
    def get_states(self) -> set[str]:
        pass

    @abstractmethod
    def get_terminal_states(self) -> set[str]:
        pass


@final
class ExampleMDP(MDP):
    """
    Simple implementation of this abstraction with an MDP which rewards depend only on the state
    """

    _transition_matrix = {
        "cell_0_0": {
            "move_right": {"cell_0_1": 0.8, "cell_0_0": 0.1, "cell_1_0": 0.1},
            "move_down": {"cell_1_0": 0.8, "cell_0_0": 0.1, "cell_0_1": 0.1},
        },
        "cell_0_1": {
            "move_right": {"cell_0_2": 0.7, "cell_0_1": 0.3},
            "move_left": {"cell_0_0": 0.9, "cell_0_1": 0.1},
            "move_down": {"lava_pit": 0.8, "cell_0_1": 0.2},
        },
        "cell_1_0": {
            "move_up": {"cell_0_0": 0.9, "cell_1_0": 0.1},
            "move_right": {"lava_pit": 0.8, "cell_1_0": 0.2},
        },
        "lava_pit": {
            "stay_and_mine": {"lava_pit": 0.4, "death": 0.6},
            "escape": {"cell_0_0": 0.5, "death": 0.5},
        },
        "cell_0_2": {
            "move_down": {"goal": 0.9, "cell_0_2": 0.1},
            "move_left": {"cell_0_1": 1.0},
        },
        "goal": {},
        "death": {},
    }

    _rewards_matrix = {
        "cell_0_0": -1,
        "cell_0_1": -1,
        "cell_1_0": -1,
        "lava_pit": -5,
        "cell_0_2": -1,
        "goal": 100,
        "death": -200,
    }

    _initial_state = "cell_0_0"

    _terminal_states = ["goal", "death"]

    @override
    def get_p(self, s_: str, r: float, s: str, a: str) -> float:
        s_state_transitions = self._transition_matrix.get(s)
        if s_state_transitions is None:
            return 0.0

        s_a_state_transitions = s_state_transitions.get(a)
        if s_a_state_transitions is None:
            return 0.0

        p = s_a_state_transitions.get(s_)
        if p is None:
            return 0.0

        return p

    @override
    def get_r(self, s: str, a: str, s_: str) -> float:
        return self._rewards_matrix.get(s, 0.0)

    @override
    def get_states(self) -> set[str]:
        return set(self._transition_matrix.keys())

    @override
    def get_terminal_states(self) -> set[str]:
        return set(self._terminal_states)


def iterative_policy_evaluation(policy: dict[str, str], mdp: MDP, theta: float = 0.01):
    delta = 0.0
    values: dict[str, float] = {}

    for s in mdp.get_states():
        values[s] = 1.0

    for s in mdp.get_terminal_states():
        values[s] = 0.0

    while delta < theta:
        for s in mdp.get_states() - mdp.get_terminal_states():
            curr_v = values[s]
            values[s] = sum(
                mdp.get_r(s, policy[s], s_) + mdp.gamma * values[s_]
                for s_ in mdp.get_states()
            )
            delta = max(delta, abs(curr_v - values[s]))

    return values


def main():
    mdp = ExampleMDP()

    policy = {
        "cell_0_0": "move_right",
        "cell_0_1": "move_right",
        "cell_1_0": "move_up",
        "lava_pit": "escape",
        "cell_0_2": "move_down",
    }

    values = iterative_policy_evaluation(policy, mdp)
    print(f"Values after iterative policy evaluation: {values}")


main()
