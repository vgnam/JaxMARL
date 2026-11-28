"""I2Q: A Fully Decentralized Q-Learning Algorithm (Jiang & Lu, NeurIPS 2022).

Ported from the authors' SMAC learner (github.com/jiechuanjiang/I2Q,
``SMAC/src/learners/q_learner.py``). On top of its independent recurrent
Q-network, every agent i learns

  * Q_ss^i(h, h')  regressed to  r + gamma * V^i_target(h'),
  * V^i(h)         the tau-expectile of Q_ss^i_target(h, h'), so with tau > 0.5
                   V approaches the value of the best *reachable* next state, i.e.
                   the "ideal transition" produced by the other agents,

and its Q target blends the two:

  y = lambda * (r + gamma * V^i_target(h')) + (1 - lambda) * (r + gamma * Q^i_target(h', a*)).

h is the agent's own feature (the GRU output of the target Q-network). Q_ss and
V are separate networks per agent while the Q-network shares parameters across
agents, as in the original code. Everything stays local to the agent.

The rollout, buffer and testing code follows ``iql_rnn.py``.
"""

import copy
import os
from functools import partial
from typing import Any

import chex
import flashbax as fbx
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

import wandb

import checkpointing
import schedules
from flat_buffer import FlatTrajectoryBuffer
from jaxmarl import make
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.environments.smax import map_name_to_scenario
from jaxmarl.wrappers.baselines import (
    CTRolloutManager,
    LogWrapper,
    MPELogWrapper,
    SMAXLogWrapper,
)


class ScannedRNN(nn.Module):
    @partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        hidden_size = ins.shape[-1]
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(hidden_size, *ins.shape[:-1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(hidden_size)(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(hidden_size, *batch_size):
        # Use a dummy key since the default state init fn is just zeros.
        return nn.GRUCell(hidden_size, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (*batch_size, hidden_size)
        )


class RNNQNetwork(nn.Module):
    """IQL's recurrent Q-network that also returns its per-step features h."""

    action_dim: int
    hidden_dim: int
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, hidden, obs, dones):
        embedding = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(obs)
        embedding = nn.relu(embedding)
        hidden, features = ScannedRNN()(hidden, (embedding, dones))
        q_vals = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(features)
        return hidden, q_vals, features


class QSS(nn.Module):
    """Q_ss(h, h'): value of reaching h' from h."""

    hidden_dim: int

    @nn.compact
    def __call__(self, h, h_next):
        x = jnp.concatenate([h, h_next], axis=-1)
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        return nn.Dense(1)(x)[..., 0]


class VS(nn.Module):
    """V(h): expectile of Q_ss(h, .) over the next states actually reached."""

    hidden_dim: int

    @nn.compact
    def __call__(self, h):
        x = nn.relu(nn.Dense(self.hidden_dim)(h))
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        return nn.Dense(1)(x)[..., 0]


@chex.dataclass(frozen=True)
class Timestep:
    obs: dict
    actions: dict
    rewards: dict
    dones: dict
    avail_actions: dict


class CustomTrainState(TrainState):
    target_network_params: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


class AuxTrainState(TrainState):
    """Train state of the per-agent Q_ss / V networks (with their targets)."""

    target_params: Any


def make_optimizer(config, lr):
    name = str(config.get("OPTIMIZER", "radam")).lower()
    if name == "radam":
        opt = optax.radam(learning_rate=lr)
    elif name == "adam":
        opt = optax.adam(learning_rate=lr)
    elif name == "rmsprop":
        # PyMARL / the I2Q SMAC code: RMSprop(alpha=0.99, eps=1e-5)
        opt = optax.rmsprop(learning_rate=lr, decay=0.99, eps=1e-5)
    else:
        raise ValueError(f"OPTIMIZER must be radam, adam or rmsprop, got {name!r}.")
    return optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), opt)


def make_train(config, env):

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    lamb = float(config.get("I2Q_LAMBDA", 0.2))
    tau = float(config.get("I2Q_TAU", 0.52))
    i2q_hidden = int(config.get("I2Q_HIDDEN_DIM", 256))

    eps_scheduler = optax.linear_schedule(
        init_value=config["EPS_START"],
        end_value=config["EPS_FINISH"],
        transition_steps=schedules.eps_decay_updates(config),
    )

    def get_greedy_actions(q_vals, valid_actions):
        unavail_actions = 1 - valid_actions
        q_vals = q_vals - (unavail_actions * 1e10)
        return jnp.argmax(q_vals, axis=-1)

    def eps_greedy_exploration(rng, q_vals, eps, valid_actions):
        rng_a, rng_e = jax.random.split(rng)
        greedy_actions = get_greedy_actions(q_vals, valid_actions)

        def get_random_actions(rng, val_action):
            return jax.random.choice(
                rng,
                jnp.arange(val_action.shape[-1]),
                p=val_action * 1.0 / jnp.sum(val_action, axis=-1),
            )

        _rngs = jax.random.split(rng_a, valid_actions.shape[0])
        random_actions = jax.vmap(get_random_actions)(_rngs, valid_actions)
        return jnp.where(
            jax.random.uniform(rng_e, greedy_actions.shape) < eps,
            random_actions,
            greedy_actions,
        )

    def batchify(x: dict):
        return jnp.stack([x[agent] for agent in env.agents], axis=0)

    def unbatchify(x: jnp.ndarray):
        return {agent: x[i] for i, agent in enumerate(env.agents)}

    # OBS_LAST_ACTION: the Q-network also reads the one-hot of the agent's
    # previous action (zeros at the first step of an episode), as PyMARL's
    # obs_last_action; h (its GRU output) then carries it into Q_ss and V.
    # On by default (False: the earlier, PyMARL-incompatible inputs).
    obs_last_action = bool(config.get("OBS_LAST_ACTION", True))

    def with_last_action(obs, last_onehot, resets):
        if not obs_last_action:
            return obs
        return jnp.concatenate(
            [obs, last_onehot * (1.0 - resets[..., None].astype(obs.dtype))], axis=-1
        )

    def train(rng, runner_state=None, num_updates=None):

        original_seed = rng[0]
        rng, _rng = jax.random.split(rng)
        wrapped_env = CTRolloutManager(env, batch_size=config["NUM_ENVS"])
        test_env = CTRolloutManager(env, batch_size=config["TEST_NUM_ENVS"])
        n_agents = len(env.agents)
        H = config["HIDDEN_SIZE"]
        n_actions = wrapped_env.max_action_space

        # INIT NETWORKS AND OPTIMIZERS
        network = RNNQNetwork(action_dim=wrapped_env.max_action_space, hidden_dim=H)
        qss = QSS(hidden_dim=i2q_hidden)
        vs = VS(hidden_dim=i2q_hidden)

        lr_scheduler = optax.linear_schedule(
            init_value=config["LR"],
            end_value=1e-10,
            transition_steps=(config["NUM_EPOCHS"]) * config["NUM_UPDATES"],
        )
        lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]

        def create_agent(rng):
            rng_q, rng_ss, rng_v = jax.random.split(rng, 3)
            init_hs = ScannedRNN.initialize_carry(H, 1)
            q_params = network.init(
                rng_q,
                init_hs,
                jnp.zeros(
                    (1, 1, wrapped_env.obs_size + (n_actions if obs_last_action else 0))
                ),
                jnp.zeros((1, 1)),
            )
            # One Q_ss / V per agent: stacked parameters, vmapped over agents.
            dummy_h = jnp.zeros((1, H))
            qss_params = jax.vmap(qss.init, in_axes=(0, None, None))(
                jax.random.split(rng_ss, n_agents), dummy_h, dummy_h
            )
            vs_params = jax.vmap(vs.init, in_axes=(0, None))(
                jax.random.split(rng_v, n_agents), dummy_h
            )
            q_state = CustomTrainState.create(
                apply_fn=network.apply,
                params=q_params,
                target_network_params=q_params,
                tx=make_optimizer(config, lr),
            )
            qss_state = AuxTrainState.create(
                apply_fn=qss.apply,
                params=qss_params,
                target_params=qss_params,
                tx=make_optimizer(config, lr),
            )
            vs_state = AuxTrainState.create(
                apply_fn=vs.apply,
                params=vs_params,
                target_params=vs_params,
                tx=make_optimizer(config, lr),
            )
            return q_state, qss_state, vs_state

        rng, _rng = jax.random.split(rng)
        train_state, qss_state, vs_state = create_agent(_rng)

        apply_q = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))  # shared params
        apply_qss = jax.vmap(qss.apply, in_axes=(0, 0, 0))  # per-agent params
        apply_vs = jax.vmap(vs.apply, in_axes=(0, 0))

        # INIT BUFFER
        def _env_sample_step(env_state, unused):
            rng, key_a, key_s = jax.random.split(jax.random.PRNGKey(0), 3)
            key_a = jax.random.split(key_a, env.num_agents)
            actions = {
                agent: wrapped_env.batch_sample(key_a[i], agent)
                for i, agent in enumerate(env.agents)
            }
            avail_actions = wrapped_env.get_valid_actions(env_state.env_state)
            obs, env_state, rewards, dones, infos = wrapped_env.batch_step(
                key_s, env_state, actions
            )
            timestep = Timestep(
                obs=obs,
                actions=actions,
                rewards=rewards,
                dones=dones,
                avail_actions=avail_actions,
            )
            return env_state, timestep

        _, _env_state = wrapped_env.batch_reset(rng)
        _, sample_traj = jax.lax.scan(
            _env_sample_step, _env_state, None, config["NUM_STEPS"]
        )
        sample_traj_unbatched = jax.tree.map(lambda x: x[:, 0], sample_traj)
        buffer = fbx.make_trajectory_buffer(
            max_length_time_axis=config["BUFFER_SIZE"] // config["NUM_ENVS"],
            min_length_time_axis=config["BUFFER_BATCH_SIZE"],
            sample_batch_size=config["BUFFER_BATCH_SIZE"],
            add_batch_size=config["NUM_ENVS"],
            sample_sequence_length=1,
            period=1,
        )
        # one flat row per trajectory: keeps the buffer in place on GPU (flat_buffer.py)
        buffer = FlatTrajectoryBuffer(buffer, sample_traj_unbatched)
        buffer_state = buffer.init(sample_traj_unbatched)

        # TRAINING LOOP
        def _update_step(runner_state, unused):

            train_state, qss_state, vs_state, buffer_state, test_state, rng = (
                runner_state
            )

            # SAMPLE PHASE
            def _step_env(carry, _):
                hs, last_obs, last_dones, last_act, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                _obs = with_last_action(
                    batchify(last_obs), last_act, batchify(last_dones)
                )[:, np.newaxis]
                _dones = batchify(last_dones)[:, np.newaxis]
                new_hs, q_vals, _ = apply_q(train_state.params, hs, _obs, _dones)
                q_vals = q_vals.squeeze(axis=1)
                avail_actions = wrapped_env.get_valid_actions(env_state.env_state)
                eps = eps_scheduler(train_state.n_updates)
                _rngs = jax.random.split(rng_a, env.num_agents)
                actions = jax.vmap(eps_greedy_exploration, in_axes=(0, 0, None, 0))(
                    _rngs, q_vals, eps, batchify(avail_actions)
                )
                new_last_act = jax.nn.one_hot(actions, n_actions)
                actions = unbatchify(actions)
                new_obs, new_env_state, rewards, dones, infos = wrapped_env.batch_step(
                    rng_s, env_state, actions
                )
                rewards, infos = schedules.add_shaped_reward(
                    rewards, infos, train_state.timesteps, config
                )
                timestep = Timestep(
                    obs=last_obs,
                    actions=actions,
                    rewards=jax.tree.map(
                        lambda x: config.get("REW_SCALE", 1) * x, rewards
                    ),
                    dones=last_dones,
                    avail_actions=avail_actions,
                )
                return (new_hs, new_obs, dones, new_last_act, new_env_state, rng), (
                    timestep,
                    infos,
                )

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = wrapped_env.batch_reset(_rng)
            init_dones = {
                agent: jnp.zeros((config["NUM_ENVS"]), dtype=bool)
                for agent in env.agents + ["__all__"]
            }
            init_hs = ScannedRNN.initialize_carry(H, n_agents, config["NUM_ENVS"])
            init_last_act = jnp.zeros((n_agents, config["NUM_ENVS"], n_actions))
            rng, _rng = jax.random.split(rng)
            _, (timesteps, infos) = jax.lax.scan(
                _step_env,
                (init_hs, init_obs, init_dones, init_last_act, env_state, _rng),
                None,
                config["NUM_STEPS"],
            )
            train_state = train_state.replace(
                timesteps=train_state.timesteps
                + config["NUM_STEPS"] * config["NUM_ENVS"]
            )

            # BUFFER UPDATE
            buffer_traj_batch = jax.tree.map(
                lambda x: jnp.swapaxes(x, 0, 1)[:, np.newaxis], timesteps
            )
            buffer_state = buffer.add(buffer_state, buffer_traj_batch)

            # NETWORKS UPDATE
            def _learn_phase(carry, _):
                train_state, qss_state, vs_state, rng = carry
                rng, _rng = jax.random.split(rng)
                minibatch = buffer.sample(buffer_state, _rng).experience
                minibatch = jax.tree.map(
                    lambda x: jnp.swapaxes(x[:, 0], 0, 1), minibatch
                )
                init_hs = ScannedRNN.initialize_carry(
                    H, n_agents, config["BUFFER_BATCH_SIZE"]
                )
                # (num_agents, timesteps, batch_size, ...)
                _obs = batchify(minibatch.obs)
                _dones = batchify(minibatch.dones)
                _actions = batchify(minibatch.actions)
                _rewards = batchify(minibatch.rewards)
                _avail = batchify(minibatch.avail_actions)
                if obs_last_action:
                    # previous action of step t = action of step t - 1 (zeros at t = 0)
                    prev = jax.nn.one_hot(_actions, n_actions)
                    prev = jnp.concatenate(
                        [jnp.zeros_like(prev[:, :1]), prev[:, :-1]], axis=1
                    )
                    _obs = with_last_action(_obs, prev, _dones)

                # dones[t] marks obs t as the first obs of a new episode, so
                # transition t is terminal iff dones[t+1].
                rewards = _rewards[:, :-1]
                cont = 1.0 - _dones[:, 1:]

                _, q_next_target, features = apply_q(
                    train_state.target_network_params, init_hs, _obs, _dones
                )
                hid, next_hid = features[:, :-1], features[:, 1:]
                # r + gamma * V(h'): the value of the transition actually reached.
                target_v = rewards + config["GAMMA"] * cont * apply_vs(
                    vs_state.target_params, next_hid
                )

                # Q_ss(h, h') <- r + gamma * V_target(h')
                def _qss_loss(params):
                    q_ss = apply_qss(params, hid, next_hid)
                    return jnp.mean((q_ss - target_v) ** 2), q_ss.mean()

                (loss_qss, qss_mean), grads = jax.value_and_grad(
                    _qss_loss, has_aux=True
                )(qss_state.params)
                qss_state = qss_state.apply_gradients(grads=grads)

                # V(h) <- tau-expectile of Q_ss_target(h, h')
                q_ss_target = apply_qss(qss_state.target_params, hid, next_hid)

                def _vs_loss(params):
                    v = apply_vs(params, hid)
                    w = jnp.where(
                        q_ss_target > jax.lax.stop_gradient(v), tau, 1.0 - tau
                    )
                    return (w * (v - q_ss_target) ** 2).sum() / w.sum(), v.mean()

                (loss_vs, v_mean), grads = jax.value_and_grad(_vs_loss, has_aux=True)(
                    vs_state.params
                )
                vs_state = vs_state.apply_gradients(grads=grads)

                def _loss_fn(params):
                    _, q_vals, _ = apply_q(params, init_hs, _obs, _dones)
                    chosen = jnp.take_along_axis(
                        q_vals, _actions[..., None], axis=-1
                    ).squeeze(-1)[:, :-1]
                    valid_q = q_vals - (1 - _avail) * 1e10
                    q_next = jnp.take_along_axis(
                        q_next_target, jnp.argmax(valid_q, axis=-1)[..., None], axis=-1
                    ).squeeze(-1)[:, 1:]
                    td_target = rewards + config["GAMMA"] * cont * q_next
                    target = lamb * target_v + (1.0 - lamb) * td_target
                    target = jax.lax.stop_gradient(target)
                    loss = jnp.mean((chosen - target) ** 2)
                    return loss, (chosen.mean(), td_target.mean(), target.mean())

                (loss, (qvals, td_mean, y_mean)), grads = jax.value_and_grad(
                    _loss_fn, has_aux=True
                )(train_state.params)
                train_state = train_state.apply_gradients(grads=grads)
                train_state = train_state.replace(grad_steps=train_state.grad_steps + 1)
                aux = {
                    "loss": loss,
                    "qvals": qvals,
                    "i2q_qss_loss": loss_qss,
                    "i2q_vs_loss": loss_vs,
                    "i2q_qss_mean": qss_mean,
                    "i2q_v_mean": v_mean,
                    "i2q_v_target_mean": target_v.mean(),
                    "i2q_td_target_mean": td_mean,
                    "target_mean": y_mean,
                }
                return (train_state, qss_state, vs_state, rng), aux

            def _learn(ts, qs, vs_, r):
                return jax.lax.scan(
                    _learn_phase, (ts, qs, vs_, r), None, config["NUM_EPOCHS"]
                )

            def _no_learn(ts, qs, vs_, r):
                aux_shape = jax.eval_shape(_learn, ts, qs, vs_, r)[1]
                aux = jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), aux_shape)
                return (ts, qs, vs_, r), aux

            rng, _rng = jax.random.split(rng)
            is_learn_time = buffer.can_sample(buffer_state) & (
                train_state.timesteps > config["LEARNING_STARTS"]
            )
            (train_state, qss_state, vs_state, rng), learn_metrics = jax.lax.cond(
                is_learn_time, _learn, _no_learn, train_state, qss_state, vs_state, _rng
            )

            # update target networks (Q, Q_ss and V together)
            do_update = train_state.n_updates % config["TARGET_UPDATE_INTERVAL"] == 0

            def _soft(new, old):
                return jax.tree.map(
                    lambda n, o: jnp.where(do_update, n, o),
                    optax.incremental_update(new, old, config["TAU"]),
                    old,
                )

            train_state = train_state.replace(
                target_network_params=_soft(
                    train_state.params, train_state.target_network_params
                )
            )
            qss_state = qss_state.replace(
                target_params=_soft(qss_state.params, qss_state.target_params)
            )
            vs_state = vs_state.replace(
                target_params=_soft(vs_state.params, vs_state.target_params)
            )

            # UPDATE METRICS
            train_state = train_state.replace(n_updates=train_state.n_updates + 1)
            metrics = {
                "env_step": train_state.timesteps,
                "update_steps": train_state.n_updates,
                "grad_steps": train_state.grad_steps,
            }
            metrics.update(jax.tree.map(lambda x: x.mean(), learn_metrics))
            metrics.update(jax.tree.map(lambda x: x.mean(), infos))
            if config.get("LOG_AGENTS_SEPARATELY", False):
                for i, a in enumerate(env.agents):
                    m = jax.tree.map(lambda x: x[..., i].mean(), infos)
                    metrics.update({k + f"_{a}": v for k, v in m.items()})

            if config.get("TEST_DURING_TRAINING", True):
                rng, _rng = jax.random.split(rng)
                test_state = jax.lax.cond(
                    train_state.n_updates
                    % int(config["NUM_UPDATES"] * config["TEST_INTERVAL"])
                    == 0,
                    lambda _: get_greedy_metrics(_rng, train_state),
                    lambda _: test_state,
                    operand=None,
                )
                metrics.update({"test_" + k: v for k, v in test_state.items()})

            if config["WANDB_MODE"] != "disabled":

                def callback(metrics, original_seed):
                    if config.get("WANDB_LOG_ALL_SEEDS", False):
                        metrics.update(
                            {
                                f"rng{int(original_seed)}/{k}": v
                                for k, v in metrics.items()
                            }
                        )
                    if checkpointing.should_log(metrics):
                        wandb.log(metrics)

                jax.debug.callback(callback, metrics, original_seed)

            runner_state = (
                train_state,
                qss_state,
                vs_state,
                buffer_state,
                test_state,
                rng,
            )
            return runner_state, metrics

        def get_greedy_metrics(rng, train_state):
            """Help function to test greedy policy during training"""
            if not config.get("TEST_DURING_TRAINING", True):
                return None
            params = train_state.params

            def _greedy_env_step(step_state, unused):
                params, env_state, last_obs, last_dones, last_act, hstate, rng = (
                    step_state
                )
                rng, key_s = jax.random.split(rng)
                _obs = with_last_action(
                    batchify(last_obs), last_act, batchify(last_dones)
                )[:, np.newaxis]
                _dones = batchify(last_dones)[:, np.newaxis]
                hstate, q_vals, _ = apply_q(params, hstate, _obs, _dones)
                q_vals = q_vals.squeeze(axis=1)
                valid_actions = test_env.get_valid_actions(env_state.env_state)
                actions = get_greedy_actions(q_vals, batchify(valid_actions))
                last_act = jax.nn.one_hot(actions, n_actions)
                actions = unbatchify(actions)
                obs, env_state, rewards, dones, infos = test_env.batch_step(
                    key_s, env_state, actions
                )
                infos = schedules.drop_shaped_reward(infos)
                step_state = (params, env_state, obs, dones, last_act, hstate, rng)
                return step_state, (rewards, dones, infos)

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = test_env.batch_reset(_rng)
            init_dones = {
                agent: jnp.zeros((config["TEST_NUM_ENVS"]), dtype=bool)
                for agent in env.agents + ["__all__"]
            }
            rng, _rng = jax.random.split(rng)
            hstate = ScannedRNN.initialize_carry(H, n_agents, config["TEST_NUM_ENVS"])
            step_state = (
                params,
                env_state,
                init_obs,
                init_dones,
                jnp.zeros((n_agents, config["TEST_NUM_ENVS"], n_actions)),
                hstate,
                _rng,
            )
            step_state, (rewards, dones, infos) = jax.lax.scan(
                _greedy_env_step, step_state, None, config["TEST_NUM_STEPS"]
            )
            if config.get("LOG_AGENTS_SEPARATELY", False):
                metrics = {}
                for i, a in enumerate(env.agents):
                    m = jax.tree.map(
                        lambda x: jnp.nanmean(
                            jnp.where(
                                infos["returned_episode"][..., i], x[..., i], jnp.nan
                            )
                        ),
                        infos,
                    )
                    metrics.update({k + f"_{a}": v for k, v in m.items()})
            else:
                metrics = jax.tree.map(
                    lambda x: jnp.nanmean(
                        jnp.where(infos["returned_episode"], x, jnp.nan)
                    ),
                    infos,
                )
            return metrics

        rng, _rng = jax.random.split(rng)
        test_state = get_greedy_metrics(_rng, train_state)

        rng, _rng = jax.random.split(rng)
        init_state = (
            train_state,
            qss_state,
            vs_state,
            buffer_state,
            test_state,
            _rng,
        )
        if runner_state is None:
            runner_state = init_state
        if num_updates is None:
            num_updates = config["NUM_UPDATES"]
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, num_updates
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train


def env_from_config(config):
    env_name = config["ENV_NAME"]
    if "smax" in env_name.lower():
        config["ENV_KWARGS"]["scenario"] = map_name_to_scenario(config["MAP_NAME"])
        env_name = f"{config['ENV_NAME']}_{config['MAP_NAME']}"
        env = make(config["ENV_NAME"], **config["ENV_KWARGS"])
        env = SMAXLogWrapper(env)
    elif "overcooked" in env_name.lower():
        env_name = f"{config['ENV_NAME']}_{config['ENV_KWARGS']['layout']}"
        config["ENV_KWARGS"]["layout"] = overcooked_layouts[
            config["ENV_KWARGS"]["layout"]
        ]
        env = make(config["ENV_NAME"], **config["ENV_KWARGS"])
        env = LogWrapper(env)
    elif "mpe" in env_name.lower():
        env = make(config["ENV_NAME"], **config["ENV_KWARGS"])
        env = MPELogWrapper(env)
    else:
        env = make(config["ENV_NAME"], **config["ENV_KWARGS"])
        env = LogWrapper(env)
    return env, env_name


def single_run(config):

    config = {**config, **config["alg"]}  # merge the alg config with the main config
    print("Config:\n", OmegaConf.to_yaml(config))

    alg_name = config.get("ALG_NAME", "i2q_rnn")
    env, env_name = env_from_config(copy.deepcopy(config))

    ckpt_dir = checkpointing.checkpoint_dir(config, alg_name, env_name)
    checkpointing.init_wandb(
        ckpt_dir,
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[alg_name.upper(), env_name.upper(), f"jax_{jax.__version__}"],
        name=f"{alg_name}_{env_name}_s{config['SEED']}",
        job_type=alg_name,  # W&B: group by Group, then Job Type -> one curve per method
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    runner_state = checkpointing.train(
        config, make_train(config, env), rngs, ckpt_dir
    )
    outs = {"runner_state": runner_state}

    if config.get("SAVE_PATH", None) is not None:
        from jaxmarl.wrappers.baselines import save_params

        model_state = outs["runner_state"][0]
        save_dir = os.path.join(config["SAVE_PATH"], env_name)
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(
            config,
            os.path.join(
                save_dir, f"{alg_name}_{env_name}_seed{config['SEED']}_config.yaml"
            ),
        )
        for i, rng in enumerate(rngs):
            params = jax.tree.map(lambda x: x[i], model_state.params)
            save_path = os.path.join(
                save_dir,
                f"{alg_name}_{env_name}_seed{config['SEED']}_vmap{i}.safetensors",
            )
            save_params(params, save_path)


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))
    single_run(config)


if __name__ == "__main__":
    main()
