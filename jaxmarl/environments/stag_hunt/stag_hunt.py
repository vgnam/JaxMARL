"""Predator-prey (stag hunt) grid world in JAX (env id "StagHunt").

A port of the ``stag_hunt`` environment of DCG / PyMARL
(github.com/wendelinboehmer/dcg, ``src/envs/stag_hunt.py``) for its square,
bounded, agent-centric configurations, in particular PyMARL's
``pred_prey_punish`` (RAC's predator-prey with miscapture punishment, the
default) and ``pred_prey_original`` (``capture_action=False``).

World: an ``world_shape`` grid with ``n_agents`` predators, ``n_stags`` stags and
``n_hare`` hares on distinct cells, placed uniformly at random (agents first).

Step (one environment step):

1. Agents move one after another in a random order. A move into a wall, or onto
   a cell holding an entity of a type in ``agent_move_block`` (0 agents, 1 stags,
   2 hares), is a collision and the agent stays; ``stay`` and ``catch`` count as
   collisions with the agent itself, exactly as in the original. Frozen agents
   do not move. Each collision of a free agent adds ``reward_collision``.
2. Prey are processed one after another in a random order. With
   ``capture_action`` a prey is captured when at least
   ``capture_action_conditions[type]`` free agents orthogonally adjacent to it
   play ``catch``; if some but too few do, the team gets
   ``miscapture_punishment`` (per prey). Without it a prey is captured when it
   has at most ``capture_conditions[type]`` free neighbouring cells. With
   ``prevent_cannibalism`` a capture also needs an agent next to the prey. A
   captured prey is removed and pays ``reward_stag`` / ``reward_hare``; with
   ``capture_freezes`` the adjacent (catching) agents freeze, and with
   ``remove_frozen`` frozen agents leave the grid. A prey that is not captured
   rests with probability ``p_stags_rest`` / ``p_hare_rest`` (only if no agent
   is on its cell) or when it cannot move, and otherwise moves to a uniformly
   random free neighbouring cell.
3. Every step pays ``reward_time``. The episode ends when every prey is
   captured, every agent is frozen, a capture happens with
   ``capture_terminal``, or after ``episode_limit`` steps.

All agents receive the same (team) reward. Observation (per agent): the
``(2 * agent_obs[0] + 1) x (2 * agent_obs[1] + 1) x 3`` window centred on the
agent, with channels (free agents, alive stags, alive hares), 1 where present,
flattened row-major; with ``observe_walls`` cells outside the grid are -1 in the
agent channel. Frozen agents see zeros when ``remove_frozen``. Available
actions: frozen agents can only ``stay``; moves off the grid are unavailable;
``catch`` (``capture_action``) only next to a prey.

Not ported: toroidal worlds, directed observations, mountains, ghosts and the
graph / intersection interfaces. The random agent and prey orders are drawn
with JAX's PRNG, so trajectories match the original in distribution, not draw
for draw.

Actions: 0 right, 1 down, 2 left, 3 up, 4 stay, 5 catch (only with
``capture_action``). Positions are (row, column).
"""

from typing import Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
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

RIGHT, DOWN, LEFT, UP, STAY, CATCH = range(6)
_MOVES = jnp.array(
    [[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0], [0, 0]], dtype=jnp.int32
)


@struct.dataclass
class State(BaseState):
    """From BaseState: done (bool), step (int)."""

    agent_pos: chex.Array  # (N, 2) int32
    frozen: chex.Array  # (N,) bool
    prey_pos: chex.Array  # (P, 2) int32
    prey_alive: chex.Array  # (P,) bool
    prey_type: chex.Array  # (P,) int32, 1 = stag, 2 = hare


class StagHunt(MultiAgentEnv):
    def __init__(
        self,
        n_agents: int = 8,
        n_stags: int = 8,
        n_hare: int = 0,
        world_shape: Sequence[int] = (10, 10),
        agent_obs: Sequence[int] = (2, 2),
        episode_limit: int = 200,
        capture_action: bool = True,
        capture_conditions: Sequence[int] = (0, 1),
        capture_action_conditions: Sequence[int] = (2, 1),
        miscapture_punishment: float = -2.0,
        capture_freezes: bool = True,
        remove_frozen: bool = True,
        capture_terminal: bool = False,
        prevent_cannibalism: bool = True,
        agent_move_block: Sequence[int] = (0, 1, 2),
        p_stags_rest: float = 0.0,
        p_hare_rest: float = 0.0,
        reward_stag: float = 10.0,
        reward_hare: float = 1.0,
        reward_time: float = 0.0,
        reward_collision: float = 0.0,
        observe_walls: bool = False,
    ):
        """Defaults are PyMARL's ``pred_prey_punish`` (10x10, 8 agents, 8
        stags, 5x5 views, two catchers per stag, -2 per miscapture, +10 per
        stag, catchers freeze and leave the grid, 200 steps)."""
        super().__init__(num_agents=n_agents)
        self.n_stags, self.n_hare = int(n_stags), int(n_hare)
        self.n_prey = self.n_stags + self.n_hare
        assert self.n_prey >= 1, "PredatorPrey needs at least one prey."
        self.world_shape = tuple(int(s) for s in world_shape)
        assert n_agents + self.n_prey <= self.world_shape[0] * self.world_shape[1]
        self.agent_obs = tuple(int(s) for s in agent_obs)
        self.episode_limit = int(episode_limit)
        self.capture_action = bool(capture_action)
        self.capture_conditions = jnp.array(capture_conditions, jnp.int32)
        self.capture_action_conditions = jnp.array(capture_action_conditions, jnp.int32)
        self.miscapture_punishment = float(miscapture_punishment)
        self.capture_freezes = bool(capture_freezes)
        self.remove_frozen = bool(remove_frozen)
        self.capture_terminal = bool(capture_terminal)
        self.prevent_cannibalism = bool(prevent_cannibalism)
        block = np.zeros(3, bool)
        block[list(agent_move_block)] = True
        self.agent_move_block = jnp.array(block)
        self.p_rest = jnp.array([0.0, p_stags_rest, p_hare_rest], jnp.float32)
        self.prey_reward = jnp.array([0.0, reward_stag, reward_hare], jnp.float32)
        self.reward_time = float(reward_time)
        self.reward_collision = float(reward_collision)
        self.observe_walls = bool(observe_walls)

        self.n_actions = 6 if self.capture_action else 5
        h, w = 2 * self.agent_obs[0] + 1, 2 * self.agent_obs[1] + 1
        self.obs_dim = h * w * 3
        self.observation_spaces = {
            a: Box(-1.0, 1.0, (self.obs_dim,), jnp.float32) for a in self.agents
        }
        self.action_spaces = {a: Discrete(self.n_actions) for a in self.agents}
        self.prey_type_init = jnp.array(
            [1] * self.n_stags + [2] * self.n_hare, jnp.int32
        )

    # ------------------------------------------------------------------ grid
    def _bound(self, pos):
        hi = jnp.array(self.world_shape, jnp.int32) - 1
        return jnp.clip(pos, 0, hi)

    def _grid(self, state: State):
        """(H, W, 3) int32 occupancy: agents on the grid, alive stags, hares."""
        grid = jnp.zeros(self.world_shape + (3,), jnp.int32)
        on_grid = ~(state.frozen & self.remove_frozen)
        grid = grid.at[state.agent_pos[:, 0], state.agent_pos[:, 1], 0].add(
            on_grid.astype(jnp.int32)
        )
        grid = grid.at[
            state.prey_pos[:, 0], state.prey_pos[:, 1], state.prey_type
        ].add(state.prey_alive.astype(jnp.int32))
        return grid

    @staticmethod
    def _sample_cell(key, free):
        """A uniformly random cell among the True cells of ``free`` (H, W)."""
        h, w = free.shape
        logits = jnp.where(free.reshape(-1), 0.0, -jnp.inf)
        idx = jax.random.categorical(key, logits)
        return jnp.array([idx // w, idx % w], jnp.int32)

    # ----------------------------------------------------------------- reset
    def reset(self, key: PRNGKeyArray) -> Tuple[Observations, State]:
        free = jnp.ones(self.world_shape, bool)
        positions = []
        for k in jax.random.split(key, self.num_agents + self.n_prey):
            pos = self._sample_cell(k, free)
            free = free.at[pos[0], pos[1]].set(False)
            positions.append(pos)
        positions = jnp.stack(positions)
        state = State(
            done=jnp.array(False),
            step=jnp.array(0, jnp.int32),
            agent_pos=positions[: self.num_agents],
            frozen=jnp.zeros((self.num_agents,), bool),
            prey_pos=positions[self.num_agents :],
            prey_alive=jnp.ones((self.n_prey,), bool),
            prey_type=self.prey_type_init,
        )
        return self.get_obs(state), state

    # ------------------------------------------------------------------ step
    def step_env(
        self, key: PRNGKeyArray, state: State, actions: Actions
    ) -> Tuple[Observations, State, Rewards, Dones, Infos]:
        N, P = self.num_agents, self.n_prey
        acts = jnp.array([actions[a] for a in self.agents]).reshape(N).astype(jnp.int32)
        k_agents, k_prey = jax.random.split(key)
        reward = jnp.array(self.reward_time, jnp.float32)
        grid = self._grid(state)

        # 1. agents move sequentially in a random order
        def agent_move(i, carry):
            grid, pos, reward = carry
            a = order_a[i]
            free_agent = ~state.frozen[a]
            new = self._bound(pos[a] + _MOVES[acts[a]])
            collide = jnp.any((grid[new[0], new[1]] > 0) & self.agent_move_block)
            move = free_agent & ~collide
            reward = reward + jnp.where(free_agent & collide, self.reward_collision, 0.0)
            grid = grid.at[pos[a, 0], pos[a, 1], 0].add(-move.astype(jnp.int32))
            grid = grid.at[new[0], new[1], 0].add(move.astype(jnp.int32))
            pos = pos.at[a].set(jnp.where(move, new, pos[a]))
            return grid, pos, reward

        order_a = jax.random.permutation(k_agents, N)
        grid, agent_pos, reward = jax.lax.fori_loop(
            0, N, agent_move, (grid, state.agent_pos, reward)
        )

        # 2. prey are processed sequentially in a random order
        k_order, k_prey = jax.random.split(k_prey)
        order_p = jax.random.permutation(k_order, P)
        prey_keys = jax.random.split(k_prey, P)
        catching = (acts == CATCH) if self.capture_action else jnp.zeros((N,), bool)
        # movement actions an agent checks when it freezes (original: n_actions - 1)
        n_freeze_moves = self.n_actions - 1

        def prey_step(i, carry):
            grid, prey_pos, alive, frozen, reward, terminal = carry
            p = order_p[i]
            pos, t, live = prey_pos[p], state.prey_type[p], alive[p]
            k_rest, k_move = jax.random.split(prey_keys[i])
            nbrs = jax.vmap(self._bound)(pos[None] + _MOVES[:4])  # (4, 2)
            cells = grid[nbrs[:, 0], nbrs[:, 1]]  # (4, 3)
            possible = ~jnp.any(cells > 0, axis=-1)  # (4,)
            next_to_agent = jnp.any(cells[:, 0] > 0)
            adjacent = jnp.any(
                (agent_pos[:, None, :] == (pos[None] + _MOVES[:4])[None]).all(-1), -1
            )  # (N,) orthogonal neighbours of the prey (unbounded)
            if self.capture_action:
                n_catch = jnp.sum(catching & ~frozen & adjacent)
                captured = n_catch >= self.capture_action_conditions[t - 1]
                reward = reward + jnp.where(
                    live & (n_catch > 0) & ~captured, self.miscapture_punishment, 0.0
                )
            else:
                captured = possible.sum() <= self.capture_conditions[t - 1]
            if self.prevent_cannibalism:
                captured = captured & next_to_agent
            captured = captured & live

            # capture: remove the prey, pay, freeze the adjacent (catching) agents
            reward = reward + jnp.where(captured, self.prey_reward[t], 0.0)
            terminal = terminal | (captured & self.capture_terminal)
            grid = grid.at[pos[0], pos[1], t].add(-captured.astype(jnp.int32))
            alive = alive.at[p].set(live & ~captured)
            if self.capture_freezes:
                near = jnp.any(
                    (
                        (agent_pos[:, None, :] + _MOVES[None, :n_freeze_moves])
                        == pos[None, None]
                    ).all(-1),
                    -1,
                )
                freeze = captured & ~frozen & near
                if self.capture_action:
                    freeze = freeze & catching
                if self.remove_frozen:
                    grid = grid.at[agent_pos[:, 0], agent_pos[:, 1], 0].add(
                        -freeze.astype(jnp.int32)
                    )
                frozen = frozen | freeze

            # otherwise the prey rests or moves to a random free neighbour
            rest = (
                (grid[pos[0], pos[1], 0] == 0)
                & (jax.random.uniform(k_rest) < self.p_rest[t])
            ) | ~possible.any()
            move = live & ~captured & ~rest
            u = jax.random.categorical(k_move, jnp.where(possible, 0.0, -jnp.inf))
            new = nbrs[u]
            grid = grid.at[pos[0], pos[1], t].add(-move.astype(jnp.int32))
            grid = grid.at[new[0], new[1], t].add(move.astype(jnp.int32))
            prey_pos = prey_pos.at[p].set(jnp.where(move, new, pos))
            return grid, prey_pos, alive, frozen, reward, terminal

        grid, prey_pos, prey_alive, frozen, reward, terminal = jax.lax.fori_loop(
            0,
            P,
            prey_step,
            (grid, state.prey_pos, state.prey_alive, state.frozen, reward, jnp.array(False)),
        )

        step = state.step + 1
        done = (
            terminal
            | ~prey_alive.any()
            | frozen.all()
            | (step >= self.episode_limit)
        )
        state = state.replace(
            agent_pos=agent_pos,
            frozen=frozen,
            prey_pos=prey_pos,
            prey_alive=prey_alive,
            step=step,
            done=done,
        )
        rewards = {a: reward for a in self.agents}
        dones = {a: done for a in self.agents}
        dones["__all__"] = done
        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            rewards,
            dones,
            {},
        )

    # ------------------------------------------------------------------- obs
    def get_obs(self, state: State) -> Observations:
        (H, W), (oh, ow) = self.world_shape, self.agent_obs
        view = jnp.zeros((H + 2 * oh, W + 2 * ow, 3), jnp.float32)
        if self.observe_walls:
            inside = jnp.zeros((H + 2 * oh, W + 2 * ow), bool)
            inside = inside.at[oh : oh + H, ow : ow + W].set(True)
            view = view.at[..., 0].set(jnp.where(inside, 0.0, -1.0))
        view = view.at[state.agent_pos[:, 0] + oh, state.agent_pos[:, 1] + ow, 0].add(
            (~state.frozen).astype(jnp.float32)
        )
        view = view.at[
            state.prey_pos[:, 0] + oh, state.prey_pos[:, 1] + ow, state.prey_type
        ].add(state.prey_alive.astype(jnp.float32))

        def one(i):
            win = jax.lax.dynamic_slice(
                view,
                (state.agent_pos[i, 0], state.agent_pos[i, 1], 0),
                (2 * oh + 1, 2 * ow + 1, 3),
            ).reshape(-1)
            if self.capture_freezes and self.remove_frozen:
                win = jnp.where(state.frozen[i], 0.0, win)
            return win

        obs = jax.vmap(one)(jnp.arange(self.num_agents))
        return {a: obs[i] for i, a in enumerate(self.agents)}

    def get_avail_actions(self, state: State) -> AvailActions:
        H, W = self.world_shape
        grid = self._grid(state)
        target = state.agent_pos[:, None, :] + _MOVES[None, : self.n_actions]
        inside = (
            (target[..., 0] >= 0)
            & (target[..., 0] < H)
            & (target[..., 1] >= 0)
            & (target[..., 1] < W)
        )  # (N, A); stay and catch are always inside
        avail = inside
        if self.capture_action:
            nbrs = jax.vmap(jax.vmap(self._bound))(
                state.agent_pos[:, None, :] + _MOVES[None, :4]
            )  # (N, 4, 2)
            prey_next = jnp.any(grid[nbrs[..., 0], nbrs[..., 1], 1:] > 0, axis=(-1, -2))
            avail = avail.at[:, CATCH].set(prey_next)
        only_stay = jnp.zeros((self.n_actions,), bool).at[STAY].set(True)
        avail = jnp.where(state.frozen[:, None], only_stay[None], avail)
        avail = avail.astype(jnp.int32)
        return {a: avail[i] for i, a in enumerate(self.agents)}

    @property
    def name(self) -> str:
        return "StagHunt"

    @property
    def agent_classes(self) -> dict:
        return {"agents": self.agents}
