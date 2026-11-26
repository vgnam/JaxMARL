"""Tests for the predator-prey (stag hunt) port of PyMARL / DCG's stag_hunt."""

import jax
import jax.numpy as jnp
import numpy as np

from jaxmarl import make
from jaxmarl.environments.stag_hunt.stag_hunt import (
    CATCH,
    DOWN,
    LEFT,
    RIGHT,
    STAY,
    UP,
    State,
)


def _state(agent_pos, prey_pos, frozen=None, prey_type=None, step=0):
    n, p = len(agent_pos), len(prey_pos)
    return State(
        done=jnp.array(False),
        step=jnp.array(step, jnp.int32),
        agent_pos=jnp.array(agent_pos, jnp.int32),
        frozen=jnp.zeros((n,), bool) if frozen is None else jnp.array(frozen),
        prey_pos=jnp.array(prey_pos, jnp.int32),
        prey_alive=jnp.ones((p,), bool),
        prey_type=jnp.ones((p,), jnp.int32) if prey_type is None else jnp.array(prey_type),
    )


def _step(env, state, acts, key=0):
    actions = {a: jnp.array(u) for a, u in zip(env.agents, acts)}
    return env.step_env(jax.random.PRNGKey(key), state, actions)


def _punish_env(**kw):
    # static prey (p_rest = 1) so the outcome does not depend on prey moves
    base = dict(n_agents=2, n_stags=2, p_stags_rest=1.0)
    base.update(kw)
    return make("StagHunt", **base)


def test_capture_needs_two_catchers_and_freezes_them():
    env = _punish_env()
    s = _state([[4, 5], [5, 4]], [[5, 5], [9, 9]])
    obs, s2, rew, dones, _ = _step(env, s, [CATCH, CATCH])
    assert float(rew[env.agents[0]]) == 10.0
    assert np.array_equal(np.asarray(s2.prey_alive), [False, True])
    assert np.all(np.asarray(s2.frozen))
    # every agent frozen -> episode over
    assert bool(dones["__all__"])
    # frozen agents (remove_frozen) see nothing
    assert np.all(np.asarray(obs[env.agents[0]]) == 0)


def test_miscapture_punishment():
    env = _punish_env()
    s = _state([[4, 5], [0, 0]], [[5, 5], [9, 9]])
    _, s2, rew, dones, _ = _step(env, s, [CATCH, STAY])
    assert float(rew[env.agents[0]]) == -2.0
    assert bool(s2.prey_alive[0]) and not bool(s2.frozen[0])
    assert not bool(dones["__all__"])
    # one catcher next to each of two stags: two punishments
    s = _state([[4, 5], [8, 9]], [[5, 5], [9, 9]])
    _, _, rew, _, _ = _step(env, s, [CATCH, CATCH])
    assert float(rew[env.agents[0]]) == -4.0


def test_catch_by_non_adjacent_agent_does_nothing():
    env = _punish_env()
    s = _state([[0, 0], [5, 4]], [[5, 5], [9, 9]])
    _, s2, rew, _, _ = _step(env, s, [CATCH, CATCH])
    # only one adjacent catcher
    assert float(rew[env.agents[0]]) == -2.0
    assert bool(s2.prey_alive[0])


def test_original_rules_capture_by_surrounding():
    env = make(
        "StagHunt",
        n_agents=2,
        n_stags=1,
        capture_action=False,
        capture_freezes=False,
        miscapture_punishment=0.0,
        reward_stag=50.0,
        reward_time=-0.1,
    )
    # stag in the corner, both free neighbours taken by agents
    s = _state([[0, 1], [1, 0]], [[0, 0]])
    _, s2, rew, dones, _ = _step(env, s, [STAY, STAY])
    assert np.isclose(float(rew[env.agents[0]]), 49.9)
    assert not bool(s2.prey_alive[0])
    assert bool(dones["__all__"])  # every prey captured
    # one free neighbour left: the stag escapes and the step only costs time
    s = _state([[0, 1], [5, 5]], [[0, 0]])
    _, s2, rew, _, _ = _step(env, s, [STAY, STAY])
    assert np.isclose(float(rew[env.agents[0]]), -0.1)
    assert bool(s2.prey_alive[0])
    assert np.array_equal(np.asarray(s2.prey_pos[0]), [1, 0])


def test_moves_collisions_and_walls():
    env = _punish_env(n_agents=3)
    s = _state([[0, 0], [5, 5], [5, 6]], [[9, 9], [9, 0]])
    _, s2, _, _, _ = _step(env, s, [UP, RIGHT, DOWN])
    pos = np.asarray(s2.agent_pos)
    assert np.array_equal(pos[0], [0, 0])  # wall
    assert np.array_equal(pos[2], [6, 6])
    # agent 1 is blocked by agent 2 unless agent 2 moved first (random order)
    assert pos[1].tolist() in ([5, 5], [5, 6])
    # moving into a stag is blocked
    s = _state([[8, 9], [0, 0], [3, 3]], [[9, 9], [9, 0]])
    _, s2, _, _, _ = _step(env, s, [DOWN, STAY, STAY])
    assert np.array_equal(np.asarray(s2.agent_pos[0]), [8, 9])


def test_avail_actions():
    env = _punish_env()
    s = _state([[0, 0], [4, 5]], [[5, 5], [9, 9]], frozen=[False, False])
    avail = env.get_avail_actions(s)
    # corner: no up, no left, no catch (no prey adjacent)
    assert np.asarray(avail[env.agents[0]]).tolist() == [1, 1, 0, 0, 1, 0]
    # next to the stag: catch available
    assert np.asarray(avail[env.agents[1]]).tolist() == [1, 1, 1, 1, 1, 1]
    s = _state([[0, 0], [4, 5]], [[5, 5], [9, 9]], frozen=[False, True])
    avail = env.get_avail_actions(s)
    assert np.asarray(avail[env.agents[1]]).tolist() == [0, 0, 0, 0, 1, 0]


def test_observation_window():
    env = _punish_env(observe_walls=True)
    s = _state([[0, 0], [1, 2]], [[2, 1], [9, 9]])
    obs = env.get_obs(s)
    o = np.asarray(obs[env.agents[0]]).reshape(5, 5, 3)
    assert o.shape == (5, 5, 3) and env.obs_dim == 75
    assert o[2, 2, 0] == 1  # self in the centre
    assert o[3, 4, 0] == 1  # agent 1 at (+1, +2)
    assert o[4, 3, 1] == 1  # stag at (+2, +1)
    assert o[9 % 5, 9 % 5, 1] == 0  # far stag not visible
    assert o[0, 0, 0] == -1 and o[1, 2, 0] == -1  # walls above / left
    assert o[2:, 2:, 0].min() >= 0
    # without walls the outside is 0
    env = _punish_env()
    o = np.asarray(env.get_obs(s)[env.agents[0]]).reshape(5, 5, 3)
    assert o.min() == 0


def test_episode_limit_and_random_rollout():
    env = make("StagHunt", episode_limit=20)
    key = jax.random.PRNGKey(3)
    obs, state = env.reset(key)
    pos = np.concatenate([np.asarray(state.agent_pos), np.asarray(state.prey_pos)])
    assert len({tuple(p) for p in pos}) == len(pos)  # distinct cells

    def body(carry, k):
        state = carry
        k_a, k_s = jax.random.split(k)
        avail = env.get_avail_actions(state)
        acts = {
            a: jax.random.categorical(
                jax.random.fold_in(k_a, i), jnp.where(avail[a] > 0, 0.0, -jnp.inf)
            )
            for i, a in enumerate(env.agents)
        }
        obs, state, rew, dones, _ = env.step(k_s, state, acts)
        occupied = jnp.concatenate([state.agent_pos, state.prey_pos])
        return state, (rew[env.agents[0]], dones["__all__"], occupied)

    _, (rews, dones, occ) = jax.lax.scan(
        jax.jit(body), state, jax.random.split(key, 60)
    )
    dones = np.asarray(dones)
    assert dones.sum() >= 2  # episodes of at most 20 steps
    assert np.all(np.isfinite(np.asarray(rews)))
