"""Two-player cooperative matrix games in JAX.

A port of the PyMARL matrix games used by RAC (``matrix_game_climbing``,
``matrix_game_penalty``, ``matrix_game_partially_stochastic``,
``matrix_game_fully_stochastic`` and ``matrix_game_repeated``): two agents pick
one of three actions simultaneously and both receive the payoff of the joint
action. Every cell lists equally likely payoffs, so a stochastic cell pays one
of them uniformly at random.

* ``climbing`` (Claus & Boutilier, 1998):
    [[11, -30, 0], [-30, 7, 6], [0, 0, 5]]; optimum (0, 0) = 11, but (0, x) and
    (x, 0) are heavily punished, so independent learners settle on (1, 1) or
    (2, 2) (relative overgeneralisation).
* ``penalty`` (penalty ``k`` < 10, default -15):
    [[10, 0, k], [0, 2, 0], [k, 0, 10]]; two optima (0, 0) and (2, 2) and a
    miscoordination penalty k between them.
* ``partially_stochastic_climbing`` (Kapetanakis & Kudenko, 2002): climbing
    with (1, 1) paying 14 or 0 (expected 7).
* ``fully_stochastic_climbing``: every cell has two equally likely payoffs
    (PyMARL's ``matrix_game_fully_stochastic``), expected values as climbing.

The game is repeated for ``episode_limit`` steps (1 = the one-shot game). The
observation of each agent is the current step index, as in PyMARL, and the
episode return is the sum of the payoffs.
"""

from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct
from jaxtyping import PRNGKeyArray

from jaxmarl.environments.multi_agent_env import (
    Actions,
    AvailActions,
    Dones,
    Infos,
    MultiAgentEnv,
    Observations,
    Rewards,
)
from jaxmarl.environments.multi_agent_env import State as BaseState
from jaxmarl.environments.spaces import Box, Discrete


def _payoff_tables(k: float):
    """Payoff cells: lists of equally likely payoffs per joint action."""
    return {
        "climbing": [[[11], [-30], [0]], [[-30], [7], [6]], [[0], [0], [5]]],
        "penalty": [[[10], [0], [k]], [[0], [2], [0]], [[k], [0], [10]]],
        "partially_stochastic_climbing": [
            [[11], [-30], [0]],
            [[-30], [14, 0], [6]],
            [[0], [0], [5]],
        ],
        "fully_stochastic_climbing": [
            [[10, 12], [-65, 5], [8, -8]],
            [[-65, 5], [14, 0], [12, 0]],
            [[8, -8], [12, 0], [10, 0]],
        ],
    }


@struct.dataclass
class State(BaseState):
    """From BaseState: done (bool), step (int)."""


class MatrixGame(MultiAgentEnv):
    def __init__(
        self,
        game: str = "climbing",
        episode_limit: int = 1,
        penalty_k: float = -15.0,
    ):
        """
        Args:
            game: "climbing", "penalty", "partially_stochastic_climbing" or
                "fully_stochastic_climbing".
            episode_limit: number of times the game is played per episode.
            penalty_k: miscoordination penalty of the penalty game (< 10).
        """
        super().__init__(num_agents=2)
        tables = _payoff_tables(float(penalty_k))
        if game not in tables:
            raise ValueError(f"Unknown matrix game {game!r}; choose from {sorted(tables)}.")
        if game == "penalty" and not penalty_k < 10:
            raise ValueError(f"penalty_k must be < 10, got {penalty_k}.")
        self.game = game
        self.episode_limit = int(episode_limit)
        assert self.episode_limit >= 1
        cells = tables[game]
        width = max(len(c) for row in cells for c in row)
        self.payoffs = jnp.array(
            [[c + [0] * (width - len(c)) for c in row] for row in cells], jnp.float32
        )  # (3, 3, width)
        self.n_outcomes = jnp.array([[len(c) for c in row] for row in cells], jnp.int32)
        # expected payoff of every joint action
        self.expected = self.payoffs.sum(-1) / self.n_outcomes

        self.n_actions = 3
        self.observation_spaces = {
            a: Box(0.0, float(self.episode_limit), (1,), jnp.float32) for a in self.agents
        }
        self.action_spaces = {a: Discrete(self.n_actions) for a in self.agents}

    def reset(self, key: PRNGKeyArray) -> Tuple[Observations, State]:
        state = State(done=jnp.array(False), step=jnp.array(0, jnp.int32))
        return self.get_obs(state), state

    def step_env(
        self, key: PRNGKeyArray, state: State, actions: Actions
    ) -> Tuple[Observations, State, Rewards, Dones, Infos]:
        a0 = jnp.asarray(actions[self.agents[0]]).astype(jnp.int32).reshape(())
        a1 = jnp.asarray(actions[self.agents[1]]).astype(jnp.int32).reshape(())
        idx = jax.random.randint(key, (), 0, self.n_outcomes[a0, a1])
        reward = self.payoffs[a0, a1, idx]
        step = state.step + 1
        done = step >= self.episode_limit
        state = state.replace(step=step, done=done)
        rewards = {a: reward for a in self.agents}
        dones = {a: done for a in self.agents}
        dones["__all__"] = done
        return self.get_obs(state), state, rewards, dones, {}

    def get_obs(self, state: State) -> Observations:
        obs = jnp.array([state.step], jnp.float32)
        return {a: obs for a in self.agents}

    def get_avail_actions(self, state: State) -> AvailActions:
        return {a: jnp.ones((self.n_actions,), jnp.int32) for a in self.agents}

    @property
    def name(self) -> str:
        return "MatrixGame"

    @property
    def agent_classes(self) -> dict:
        return {"agents": self.agents}
