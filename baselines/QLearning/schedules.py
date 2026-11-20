"""Schedules shared by iql_rnn.py, rac_rnn.py and i2q_rnn.py."""


def eps_decay_updates(config):
    """Number of updates over which epsilon anneals from EPS_START to EPS_FINISH.

    EPS_DECAY_STEPS (env steps) gives a fixed length that does not grow with
    TOTAL_TIMESTEPS. Without it, EPS_DECAY is the fraction of NUM_UPDATES.
    """
    steps = config.get("EPS_DECAY_STEPS")
    if steps:
        per_update = config["NUM_STEPS"] * config["NUM_ENVS"]
        return max(1, int(round(float(steps) / per_update)))
    return config["EPS_DECAY"] * config["NUM_UPDATES"]


def add_shaped_reward(rewards, infos, timesteps, config):
    """Overcooked: add info["shaped_reward"], annealed linearly from 1 to 0 over
    REW_SHAPING_HORIZON env steps, to every agent's reward (as iql_cnn_overcooked).

    Envs without "shaped_reward" in their infos are returned unchanged. The
    logged returned_episode_returns stay the env's own (sparse) reward.
    """
    if "shaped_reward" not in infos:
        return rewards, infos
    import jax.numpy as jnp

    infos = dict(infos)
    shaped = infos.pop("shaped_reward")
    horizon = float(config.get("REW_SHAPING_HORIZON", 0) or 0)
    if horizon <= 0:
        return rewards, infos
    coef = jnp.clip(1.0 - timesteps / horizon, 0.0, 1.0)
    rewards = dict(rewards)
    for agent, r in shaped.items():
        rewards[agent] = rewards[agent] + coef * r
    if "__all__" in rewards:
        # CTRolloutManager's global reward for Overcooked is the first agent's
        rewards["__all__"] = rewards[next(iter(shaped))]
    return rewards, infos


def drop_shaped_reward(infos):
    """Greedy evaluation logs the env's own reward only: drop Overcooked's
    per-agent info["shaped_reward"], whose shape does not match the other infos."""
    if "shaped_reward" not in infos:
        return infos
    return {k: v for k, v in infos.items() if k != "shaped_reward"}
