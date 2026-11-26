"""Level-Based Foraging: rules of lbforaging v3 on hand-built states.

The JAX port was also checked step by step against lbforaging itself on random
rollouts (observations, positions, food, rewards, termination).
"""

import jax
import jax.numpy as jnp
import numpy as np

from jaxmarl import make
from jaxmarl.environments.lbf.lbf import EAST, LOAD, NONE, NORTH, SOUTH, WEST


def _state(env, agent_pos, agent_level, food_pos, food_level, step=0):
    _, st = env.reset(jax.random.PRNGKey(0))
    food_level = jnp.array(food_level, jnp.int32)
    return st.replace(
        agent_pos=jnp.array(agent_pos, jnp.int32),
        agent_level=jnp.array(agent_level, jnp.int32),
        food_pos=jnp.array(food_pos, jnp.int32),
        food_level=food_level,
        food_spawned=food_level.sum().astype(jnp.float32),
        step=jnp.array(step, jnp.int32),
    )


def _step(env, st, acts):
    return env.step_env(
        jax.random.PRNGKey(1), st, {a: jnp.int32(x) for a, x in zip(env.agents, acts)}
    )


def test_reset_spawn_rules():
    env = make("LBF", num_agents=3, field_size=8, max_num_food=3, force_coop=True)
    for k in jax.random.split(jax.random.PRNGKey(0), 50):
        _, st = env.reset(k)
        ap, fp, fl = map(np.asarray, (st.agent_pos, st.food_pos, st.food_level))
        assert len({tuple(p) for p in ap}) == 3
        assert (fl == np.sort(np.asarray(st.agent_level))[:3].sum()).all()  # -coop
        for i, (r, c) in enumerate(fp):
            assert 1 <= r <= 6 and 1 <= c <= 6
            assert (r, c) not in {tuple(p) for p in ap}
            for j, (r2, c2) in enumerate(fp):
                if i != j:
                    assert max(abs(r - r2), abs(c - c2)) > 1
                    assert not (r == r2 and abs(c - c2) <= 2)
                    assert not (c == c2 and abs(r - r2) <= 2)


def test_coop_load_needs_both_and_normalises():
    env = make("LBF", num_agents=2, field_size=8, max_num_food=2, shared_rewards=False)
    st = _state(env, [[3, 2], [2, 3]], [1, 2], [[3, 3], [6, 6]], [3, 3])
    # one player alone fails
    _, st1, rew, _, _ = _step(env, st, [LOAD, NONE])
    assert float(rew["agent_0"]) == 0.0 and int(st1.food_level[0]) == 3
    # both load: reward level * food / (level_sum * food_spawned)
    _, st2, rew, dones, _ = _step(env, st, [LOAD, LOAD])
    assert int(st2.food_level[0]) == 0 and not bool(dones["__all__"])
    np.testing.assert_allclose(
        [float(rew["agent_0"]), float(rew["agent_1"])], [3 / (3 * 6), 6 / (3 * 6)]
    )


def test_shared_reward_and_termination():
    env = make("LBF", num_agents=2, field_size=8, max_num_food=1)
    st = _state(env, [[3, 2], [2, 3]], [1, 1], [[3, 3]], [2])
    _, _, rew, dones, _ = _step(env, st, [LOAD, LOAD])
    assert float(rew["agent_0"]) == float(rew["agent_1"]) == 1.0
    assert bool(dones["__all__"])
    st = _state(env, [[0, 0], [7, 7]], [1, 1], [[3, 3]], [2], step=49)
    _, _, _, dones, _ = _step(env, st, [NONE, NONE])
    assert bool(dones["__all__"])


def test_collisions_and_invalid_moves():
    env = make("LBF", num_agents=2, field_size=8, max_num_food=1, shared_rewards=False)
    # both move into (2, 2): nobody moves
    st = _state(env, [[2, 1], [2, 3]], [1, 1], [[5, 5]], [2])
    _, st1, _, _, _ = _step(env, st, [EAST, WEST])
    assert np.asarray(st1.agent_pos).tolist() == [[2, 1], [2, 3]]
    # moving into food or off the grid is a no-op
    st = _state(env, [[0, 0], [4, 5]], [1, 1], [[5, 5]], [2])
    _, st1, _, _, _ = _step(env, st, [NORTH, SOUTH])
    assert np.asarray(st1.agent_pos).tolist() == [[0, 0], [4, 5]]


def test_partial_observation():
    env = make("LBF", num_agents=2, field_size=8, max_num_food=2, sight=1)
    st = _state(env, [[1, 1], [6, 6]], [1, 2], [[2, 2], [5, 6]], [3, 3])
    obs = env.get_obs(st)
    # agent_0 window rows/cols [0, 2]: food (2, 2) at (2, 2), agent_1 unseen
    np.testing.assert_array_equal(
        obs["agent_0"], [2, 2, 3, -1, -1, 0, 1, 1, 1, -1, -1, 0]
    )
    # agent_1 window rows/cols [5, 7]: food (5, 6) at (0, 1), self at (1, 1)
    np.testing.assert_array_equal(
        obs["agent_1"], [0, 1, 3, -1, -1, 0, 1, 1, 2, -1, -1, 0]
    )
