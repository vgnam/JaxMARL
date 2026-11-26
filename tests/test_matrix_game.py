"""Tests for the cooperative matrix games (port of PyMARL's matrix games)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxmarl import make

CLIMBING = [[11, -30, 0], [-30, 7, 6], [0, 0, 5]]


def _play(env, a0, a1, key=jax.random.PRNGKey(0), state=None):
    if state is None:
        _, state = env.reset(key)
    acts = {env.agents[0]: jnp.array(a0), env.agents[1]: jnp.array(a1)}
    return env.step_env(key, state, acts)


@pytest.mark.parametrize("a0", range(3))
@pytest.mark.parametrize("a1", range(3))
def test_climbing_payoffs(a0, a1):
    env = make("MatrixGame", game="climbing")
    _, _, rew, dones, _ = _play(env, a0, a1)
    for a in env.agents:
        assert float(rew[a]) == CLIMBING[a0][a1]
    assert bool(dones["__all__"])


def test_penalty_payoffs():
    env = make("MatrixGame", game="penalty", penalty_k=-20.0)
    table = [[10, 0, -20], [0, 2, 0], [-20, 0, 10]]
    for a0 in range(3):
        for a1 in range(3):
            _, _, rew, _, _ = _play(env, a0, a1)
            assert float(rew[env.agents[0]]) == table[a0][a1]
    with pytest.raises(ValueError):
        make("MatrixGame", game="penalty", penalty_k=10.0)


def test_partially_stochastic_cell():
    env = make("MatrixGame", game="partially_stochastic_climbing")
    keys = jax.random.split(jax.random.PRNGKey(1), 4000)
    _, state = env.reset(keys[0])
    acts = {env.agents[0]: jnp.array(1), env.agents[1]: jnp.array(1)}
    rew = jax.vmap(lambda k: env.step_env(k, state, acts)[2][env.agents[0]])(keys)
    rew = np.asarray(rew)
    assert set(np.unique(rew)) == {0.0, 14.0}
    assert abs(rew.mean() - 7.0) < 0.5
    # deterministic cells stay deterministic
    acts = {env.agents[0]: jnp.array(0), env.agents[1]: jnp.array(0)}
    rew = jax.vmap(lambda k: env.step_env(k, state, acts)[2][env.agents[0]])(keys[:50])
    assert np.all(np.asarray(rew) == 11.0)
    assert np.allclose(np.asarray(env.expected), [[11, -30, 0], [-30, 7, 6], [0, 0, 5]])


def test_fully_stochastic_expectations():
    env = make("MatrixGame", game="fully_stochastic_climbing")
    # PyMARL's fully stochastic table: every cell has two equally likely payoffs
    assert np.allclose(
        np.asarray(env.expected), [[11, -30, 0], [-30, 7, 6], [0, 6, 5]]
    )


def test_repeated_game_episode():
    env = make("MatrixGame", game="climbing", episode_limit=3)
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    assert float(obs[env.agents[0]][0]) == 0.0
    acts = {a: jnp.array(0) for a in env.agents}
    ret = 0.0
    for t in range(3):
        obs, state, rew, dones, _ = env.step(key, state, acts)
        ret += float(rew[env.agents[0]])
        assert bool(dones["__all__"]) == (t == 2)
        # after the last step the env auto-resets to step 0
        assert float(obs[env.agents[0]][0]) == (0.0 if t == 2 else t + 1)
    assert ret == 33.0


def test_unknown_game():
    with pytest.raises(ValueError):
        make("MatrixGame", game="prisoners")
