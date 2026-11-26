"""Level-Based Foraging (LBF) in JAX.

A port of ``lbforaging`` v3 (``ForagingEnv`` with ``grid_observation=False``),
the version behind EPyMARL's ``Foraging-*-v3`` tasks. Rules follow
``lbforaging/foraging/environment.py``:

* Players spawn on random empty cells with a level in
  ``[min_player_level, max_player_level]``. Foods spawn on interior cells (not on
  the border) with no food in the 3x3 neighbourhood or within distance 2 along
  the row and column. The food level is the sum of the three lowest player
  levels (``max_food_level=None``); ``force_coop`` makes every food exactly
  that level, so all players must load it together.
* Actions: 0 NONE, 1 NORTH, 2 SOUTH, 3 WEST, 4 EAST, 5 LOAD. An invalid action
  (moving into food or off the grid, loading with no adjacent food) becomes
  NONE. Players whose targets coincide all stay. Loading players adjacent to
  the same food load it together when their level sum reaches the food's
  level; each gets ``level * food_level / (level_sum * food_spawned)`` with
  ``normalize_reward`` (failed loads cost ``penalty``).
* The episode ends when all food is collected or after ``max_episode_steps``.
* Observation (per agent, ``3 * max_num_food + 3 * num_agents``): the foods
  visible within ``sight`` as (row, col, level) relative to the agent's clipped
  view window in row-major order, then the visible players as (row, col,
  level), self first; missing entries are (-1, -1, 0).

Differences: rejection sampling of spawn cells is replaced by sampling
uniformly among the valid cells (the same distribution); several loading
groups in one step are resolved in agent order (lbforaging uses set order).
With ``shared_rewards`` (default) every agent receives the team reward, the
sum of the agents' rewards, as EPyMARL's and PyMARL's LBF wrappers do; the
team return of an episode is then in [0, 1].
"""

from functools import partial
from typing import Optional, Tuple

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

NONE, NORTH, SOUTH, WEST, EAST, LOAD = range(6)
_MOVES = jnp.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]], dtype=jnp.int32)


@struct.dataclass
class State(BaseState):
    """LBF state.

    From BaseState:
        done: bool
        step: int
    """

    agent_pos: chex.Array  # (N, 2) int32, (row, col)
    agent_level: chex.Array  # (N,) int32
    food_pos: chex.Array  # (F, 2) int32
    food_level: chex.Array  # (F,) int32, 0 once collected or never spawned
    food_spawned: chex.Array  # () float32, sum of the spawned food levels


class LBF(MultiAgentEnv):
    def __init__(
        self,
        num_agents: int = 2,
        field_size: int = 8,
        max_num_food: int = 2,
        sight: Optional[int] = None,
        max_episode_steps: int = 50,
        force_coop: bool = True,
        min_player_level: int = 1,
        max_player_level: int = 2,
        min_food_level: int = 1,
        max_food_level: Optional[int] = None,
        penalty: float = 0.0,
        normalize_reward: bool = True,
        shared_rewards: bool = True,
    ):
        """
        Args:
            num_agents: number of players (lbforaging ``players``).
            field_size: side of the square grid.
            max_num_food: number of food items to spawn.
            sight: view radius; ``None`` = ``field_size`` (full observation,
                lbforaging's default), 2 for the ``-2s`` tasks.
            max_episode_steps: episode length (EPyMARL ``time_limit``).
            force_coop: ``-coop`` tasks, every food needs all players.
            min_player_level / max_player_level: player level range.
            min_food_level / max_food_level: food level range; ``None`` for the
                maximum uses the sum of the three lowest player levels.
            penalty: reward lost by players of a failed load (``-pen`` tasks).
            normalize_reward: divide rewards so the team return is at most 1.
            shared_rewards: give every agent the sum of the agents' rewards.
        """
        super().__init__(num_agents=num_agents)
        assert field_size >= 3, "LBF needs field_size >= 3 (foods spawn off the border)."
        self.field_size = int(field_size)
        self.max_num_food = int(max_num_food)
        self.sight = int(field_size if sight is None else sight)
        self.max_steps = int(max_episode_steps)
        self.force_coop = bool(force_coop)
        self.min_player_level = int(min_player_level)
        self.max_player_level = int(max_player_level)
        self.min_food_level = int(min_food_level)
        self.max_food_level = None if max_food_level is None else int(max_food_level)
        self.penalty = float(penalty)
        self.normalize_reward = bool(normalize_reward)
        self.shared_rewards = bool(shared_rewards)

        obs_food_level = (
            self.max_food_level
            if self.max_food_level is not None
            else min(num_agents, 3) * self.max_player_level
        )
        s = self.field_size
        low = jnp.array(
            [-1, -1, 0] * self.max_num_food + [-1, -1, 0] * num_agents, dtype=jnp.float32
        )
        high = jnp.array(
            [s - 1, s - 1, obs_food_level] * self.max_num_food
            + [s - 1, s - 1, self.max_player_level] * num_agents,
            dtype=jnp.float32,
        )
        self.obs_dim = 3 * self.max_num_food + 3 * num_agents
        self.observation_spaces = {
            a: Box(low, high, (self.obs_dim,), jnp.float32) for a in self.agents
        }
        self.action_spaces = {a: Discrete(6) for a in self.agents}

    # ------------------------------------------------------------------ grid
    def _field(self, food_pos, food_level):
        """(S, S) grid of food levels."""
        field = jnp.zeros((self.field_size, self.field_size), jnp.int32)
        return field.at[food_pos[:, 0], food_pos[:, 1]].add(food_level)

    def _occupied(self, pos, mask):
        """(S, S) bool grid of the cells in ``pos`` selected by ``mask``."""
        grid = jnp.zeros((self.field_size, self.field_size), jnp.int32)
        return grid.at[pos[:, 0], pos[:, 1]].add(mask.astype(jnp.int32)) > 0

    def _food_spawn_mask(self, field, agents_grid):
        """Cells where lbforaging's ``spawn_food`` may place a food."""
        S = self.field_size
        has = jnp.pad((field > 0).astype(jnp.int32), 2)  # (S+4, S+4)
        # 3x3 neighbourhood (diagonals included) and the distance-2 cross
        box = sum(
            has[2 + dr : 2 + dr + S, 2 + dc : 2 + dc + S]
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
        )
        cross = sum(has[2 + d : 2 + d + S, 2 : 2 + S] for d in range(-2, 3)) + sum(
            has[2 : 2 + S, 2 + d : 2 + d + S] for d in range(-2, 3)
        )
        r = jnp.arange(S)
        interior = ((r >= 1) & (r <= S - 2))[:, None] & ((r >= 1) & (r <= S - 2))[None]
        return interior & (box == 0) & (cross == 0) & ~agents_grid & (field == 0)

    def _sample_cell(self, key, mask):
        """Uniform cell among ``mask`` (row, col) and whether one exists."""
        logits = jnp.where(mask.reshape(-1), 0.0, -jnp.inf)
        idx = jax.random.categorical(key, logits)
        return jnp.stack([idx // self.field_size, idx % self.field_size]), mask.any()

    # ----------------------------------------------------------------- reset
    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: PRNGKeyArray) -> Tuple[Observations, State]:
        N, F = self.num_agents, self.max_num_food
        k_lvl, k_pos, k_food, k_flvl = jax.random.split(key, 4)

        agent_level = jax.random.randint(
            k_lvl, (N,), self.min_player_level, self.max_player_level + 1
        )
        agent_pos = jnp.zeros((N, 2), jnp.int32)
        for i, k in enumerate(jax.random.split(k_pos, N)):
            taken = self._occupied(agent_pos, jnp.arange(N) < i)
            cell, _ = self._sample_cell(k, ~taken)
            agent_pos = agent_pos.at[i].set(cell)

        if self.max_food_level is None:
            max_food = jnp.sort(agent_level)[:3].sum()
        else:
            max_food = jnp.asarray(self.max_food_level)
        min_food = max_food if self.force_coop else jnp.asarray(self.min_food_level)
        levels = jax.random.randint(k_flvl, (F,), min_food, max_food + 1)

        agents_grid = self._occupied(agent_pos, jnp.ones(N, bool))
        food_pos = jnp.zeros((F, 2), jnp.int32)
        food_level = jnp.zeros((F,), jnp.int32)
        for j, k in enumerate(jax.random.split(k_food, F)):
            field = self._field(food_pos, food_level)
            cell, ok = self._sample_cell(k, self._food_spawn_mask(field, agents_grid))
            food_pos = food_pos.at[j].set(cell)
            food_level = food_level.at[j].set(jnp.where(ok, levels[j], 0))

        state = State(
            agent_pos=agent_pos,
            agent_level=agent_level,
            food_pos=food_pos,
            food_level=food_level,
            food_spawned=food_level.sum().astype(jnp.float32),
            done=jnp.array(False),
            step=jnp.array(0, jnp.int32),
        )
        return self.get_obs(state), state

    # ------------------------------------------------------------------ step
    def _valid_actions(self, state: State) -> chex.Array:
        """(N, 6) bool, lbforaging's ``_is_valid_action``."""
        S = self.field_size
        field = self._field(state.food_pos, state.food_level)
        r, c = state.agent_pos[:, 0], state.agent_pos[:, 1]
        at = lambda rr, cc: field[jnp.clip(rr, 0, S - 1), jnp.clip(cc, 0, S - 1)]
        north = (r > 0) & (at(r - 1, c) == 0)
        south = (r < S - 1) & (at(r + 1, c) == 0)
        west = (c > 0) & (at(r, c - 1) == 0)
        east = (c < S - 1) & (at(r, c + 1) == 0)
        # adjacent_food uses clipped indices, i.e. the agent's own (empty) cell
        adjacent = at(r - 1, c) + at(r + 1, c) + at(r, c - 1) + at(r, c + 1)
        load = adjacent > 0
        none = jnp.ones_like(load)
        return jnp.stack([none, north, south, west, east, load], axis=-1)

    def step_env(
        self, key: PRNGKeyArray, state: State, actions: Actions
    ) -> Tuple[Observations, State, Rewards, Dones, Infos]:
        N, S = self.num_agents, self.field_size
        acts = jnp.array([actions[a] for a in self.agents]).reshape(N).astype(jnp.int32)
        valid = self._valid_actions(state)
        acts = jnp.where(valid[jnp.arange(N), acts], acts, NONE)

        # movement: players whose targets coincide all stay where they are
        target = state.agent_pos + _MOVES[acts]
        same = (target[:, None, :] == target[None, :, :]).all(-1).sum(-1)
        agent_pos = jnp.where((same == 1)[:, None], target, state.agent_pos)

        # loading, resolved in agent order
        food_level = state.food_level
        rewards = jnp.zeros((N,), jnp.float32)
        pending = acts == LOAD
        lvl = state.agent_level
        for i in range(N):
            field = self._field(state.food_pos, food_level)
            r, c = agent_pos[i, 0], agent_pos[i, 1]
            at = lambda rr, cc: field[jnp.clip(rr, 0, S - 1), jnp.clip(cc, 0, S - 1)]
            # adjacent_food_location: N, S, W, E with lbforaging's bounds
            cands = jnp.array([[r - 1, c], [r + 1, c], [r, c - 1], [r, c + 1]])
            has = jnp.array(
                [
                    (r > 1) & (at(r - 1, c) > 0),
                    (r < S - 1) & (at(r + 1, c) > 0),
                    (c > 1) & (at(r, c - 1) > 0),
                    (c < S - 1) & (at(r, c + 1) > 0),
                ]
            )
            loc = cands[jnp.argmax(has)]
            active = pending[i] & has.any()
            food_idx = jnp.argmax(
                (state.food_pos == loc[None]).all(-1) & (food_level > 0)
            )
            food = food_level[food_idx]
            d = jnp.abs(agent_pos - loc[None])
            adjacent = ((d[:, 0] == 1) & (d[:, 1] == 0)) | ((d[:, 1] == 1) & (d[:, 0] == 0))
            group = active & pending & (adjacent | (jnp.arange(N) == i))
            pending = pending & ~group
            level_sum = jnp.sum(jnp.where(group, lvl, 0))
            loaded = level_sum >= food
            gain = (lvl * food).astype(jnp.float32)
            if self.normalize_reward:
                gain = gain / jnp.maximum(level_sum * state.food_spawned, 1e-8)
            rewards = jnp.where(
                group, jnp.where(loaded, gain, rewards - self.penalty), rewards
            )
            food_level = food_level.at[food_idx].set(
                jnp.where(active & loaded, 0, food_level[food_idx])
            )

        step = state.step + 1
        done = (food_level.sum() == 0) | (step >= self.max_steps)
        state = state.replace(
            agent_pos=agent_pos, food_level=food_level, step=step, done=done
        )
        if self.shared_rewards:
            rewards = jnp.full((N,), rewards.sum())
        reward_dict = {a: rewards[i] for i, a in enumerate(self.agents)}
        dones = {a: done for a in self.agents}
        dones["__all__"] = done
        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            reward_dict,
            dones,
            {},
        )

    # ------------------------------------------------------------------- obs
    def get_obs(self, state: State) -> Observations:
        N, F, S, s = self.num_agents, self.max_num_food, self.field_size, self.sight
        food_alive = state.food_level > 0
        food_key = state.food_pos[:, 0] * S + state.food_pos[:, 1]

        def one(i):
            center = state.agent_pos[i]
            origin = jnp.maximum(center - s, 0)
            upper = jnp.minimum(center + s + 1, S)
            # foods inside the clipped window, row-major
            f_in = food_alive & (state.food_pos >= origin).all(-1) & (
                state.food_pos < upper
            ).all(-1)
            order = jnp.argsort(jnp.where(f_in, food_key, S * S + jnp.arange(F)))
            f_rel = (state.food_pos - origin)[order]
            f_ok = f_in[order]
            foods = jnp.where(
                f_ok[:, None],
                jnp.concatenate([f_rel, state.food_level[order][:, None]], -1),
                jnp.array([-1, -1, 0]),
            )
            # players with relative position in [0, 2 * sight], self first
            p_rel = state.agent_pos - origin
            p_in = (p_rel >= 0).all(-1) & (p_rel <= 2 * s).all(-1)
            rank = jnp.where(jnp.arange(N) == i, -1, jnp.arange(N))
            order = jnp.argsort(jnp.where(p_in, rank, N + jnp.arange(N)))
            p_ok = p_in[order]
            players = jnp.where(
                p_ok[:, None],
                jnp.concatenate([p_rel[order], state.agent_level[order][:, None]], -1),
                jnp.array([-1, -1, 0]),
            )
            return jnp.concatenate([foods.reshape(-1), players.reshape(-1)]).astype(
                jnp.float32
            )

        obs = jax.vmap(one)(jnp.arange(N))
        return {a: obs[i] for i, a in enumerate(self.agents)}

    def get_avail_actions(self, state: State) -> AvailActions:
        valid = self._valid_actions(state)
        return {a: valid[i] for i, a in enumerate(self.agents)}

    @property
    def name(self) -> str:
        return "LBF"

    @property
    def agent_classes(self) -> dict:
        return {"agents": self.agents}
