"""RAC and HC-RAC on Overcooked with the learner of ``iql_cnn_overcooked.py``.

Everything IQL-CNN does is kept: the CNN Q-network, 1-step rollouts from
persistent envs (``NUM_STEPS: 1``), a flat transition buffer sampled
``NUM_EPOCHS`` times per update, a max-over-actions target network, and the
annealed reward shaping. On top of it every agent trains, as in ``rac_rnn.py``,

  * a twin Q(o, c, a) with its own CNN, conditioned on a context c of the
    episode the transition came from (RAC: one-hot bin of the episode return;
    HC-RAC: ``[phi(R) ; u]`` from ``rac_rnn.CoRACLabeler``), and
  * the behaviour Q is pulled towards the twin maximised over contexts through
    ``KL_WEIGHT * KL(softmax(Q) || softmax(Q_teacher))``.

The context needs the whole episode, so transitions wait in a per-env staging
area until their episode ends (Overcooked episodes all last ``max_steps`` = 400
steps and start together) and are then labelled and added to the buffer.
Learning therefore starts after the first full episodes
(400 * NUM_ENVS env steps) instead of after ``LEARNING_STARTS``.

  python baselines/QLearning/rac_cnn_overcooked.py +alg=rac_cnn_overcooked
  python baselines/QLearning/rac_cnn_overcooked.py +alg=hcrac_cnn_overcooked
"""

import copy
import os
from typing import Any

import flashbax as fbx
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import zeros
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

import wandb

from iql_cnn_overcooked import CNN, QNetwork, env_from_config
from jaxmarl.wrappers.baselines import CTRolloutManager
from rac_rnn import (
    CoRACLabeler,
    categorical_kl,
    discounted_return_bounds,
    make_optimizer,
    return_bin_contexts,
    select_teacher_q,
)


class TwinQNetwork(nn.Module):
    """Twin Q(o, c, a): IQL-CNN's trunk with RAC's context head.

    x (B, H, W, C), contexts broadcastable to (B, M, C) -> (B, M, A).
    head = "normal": Q = W [h ; c] + b. head = "hyper": Q = h W(c) + b(c), with
    W, b linear in z(c) (z = c, or relu(Dense(c)) with a zero-initialised
    context part when ``hyper_hidden > 0``), as in ``rac_rnn.TwinRNNQNetwork``.
    """

    action_dim: int
    hidden_size: int = 64
    head: str = "hyper"
    hyper_hidden: int = 0

    @nn.compact
    def __call__(self, x, contexts):
        h = nn.Dense(self.hidden_size)(CNN()(x))  # (B, H)
        if self.head == "normal":
            q_h = nn.Dense(self.action_dim, name="q_hidden")(h)
            q_c = nn.Dense(self.action_dim, use_bias=False, name="q_context")(contexts)
            return q_h[:, None, :] + q_c
        if self.head != "hyper":
            raise ValueError(f"TWIN_HEAD must be 'normal' or 'hyper', got {self.head!r}.")
        H, A = self.hidden_size, self.action_dim
        if self.hyper_hidden > 0:
            z = nn.relu(nn.Dense(self.hyper_hidden, name="hyper_in")(contexts))
            delta_init = zeros
            bound = 1.0 / np.sqrt(H)
            base_init = lambda key, shape, dtype=jnp.float32: jax.random.uniform(
                key, shape, dtype, -bound, bound
            )
        else:
            z = contexts
            delta_init = nn.initializers.lecun_normal(in_axis=0, out_axis=(1, 2))
            base_init = zeros
        K = z.shape[-1]
        w_delta = self.param("w_delta", delta_init, (K, H, A))
        w_base = self.param("w_base", base_init, (H, A))
        b_delta = self.param("b_delta", zeros, (K, A))
        b_base = self.param("b_base", zeros, (A,))
        hk = jnp.einsum("bh,kha->bka", h, w_delta)  # (B, K, A)
        q = jnp.matmul(z, hk)  # (B, M, A)
        return q + (h @ w_base + b_base)[:, None, :] + z @ b_delta


class CustomTrainState(TrainState):
    target_network_params: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


def make_train(config, env):

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    ep_len = int(env.max_steps)
    if ep_len % config["NUM_STEPS"] != 0:
        raise ValueError(f"NUM_STEPS must divide the episode length {ep_len}.")

    use_corac = str(config.get("CONTEXT_LABELER", "return_bins")).lower() == "corac"
    teacher_mode = str(config.get("TEACHER_CONTEXT_SELECT", "action_max"))
    teacher_topk = int(config.get("TEACHER_TOPK", 4))
    twin_clamp = bool(config.get("TWIN_TARGET_CLAMP", False))
    kl_temp_frac = float(config.get("KL_LOGIT_TEMPERATURE_FRAC", 0.0) or 0.0)
    slot_number = int(config.get("SLOT_NUMBER", 10))
    return_min = float(config.get("RETURN_MIN", 0.0))
    return_max = float(config.get("RETURN_MAX", 1.0))
    twin_head = str(config.get("TWIN_HEAD", "hyper"))
    twin_hyper_hidden = int(config.get("TWIN_HYPER_HIDDEN", 0) or 0)
    if use_corac and twin_head != "hyper":
        raise ValueError("HC-RAC reads continuous contexts: set TWIN_HEAD=hyper.")
    kl_start = float(config.get("KL_WEIGHT", 1.0))
    kl_final = config.get("KL_WEIGHT_FINAL", None)
    kl_begin = float(config.get("KL_ANNEAL_START", 0))
    kl_end = float(config.get("KL_ANNEAL_END", kl_begin))

    def kl_weight_at(t_env):
        if kl_final is None:
            return jnp.asarray(kl_start, jnp.float32)
        frac = jnp.clip((t_env - kl_begin) / max(kl_end - kl_begin, 1e-8), 0.0, 1.0)
        return kl_start + frac * (float(kl_final) - kl_start)

    eps_scheduler = optax.linear_schedule(
        config["EPS_START"],
        config["EPS_FINISH"],
        config["EPS_DECAY"] * config["NUM_UPDATES"],
    )
    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.0, end_value=0.0, transition_steps=config["REW_SHAPING_HORIZON"]
    )

    def get_greedy_actions(q_vals, valid_actions):
        return jnp.argmax(q_vals - (1 - valid_actions) * 1e10, axis=-1)

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

    def train(rng):

        original_seed = rng[0]
        rng, _rng = jax.random.split(rng)
        wrapped_env = CTRolloutManager(env, batch_size=config["NUM_ENVS"], preprocess_obs=False)
        test_env = CTRolloutManager(env, batch_size=config["TEST_NUM_ENVS"], preprocess_obs=False)
        n_agents = len(env.agents)
        n_actions = wrapped_env.max_action_space
        E = config["NUM_ENVS"]
        # Overcooked's observation_space says (W, H, C) but obs are (H, W, C):
        # take the shape from a real observation.
        _obs0, _ = wrapped_env.batch_reset(jax.random.PRNGKey(0))
        obs_shape = tuple(_obs0[env.agents[0]].shape[1:])
        obs_dim = int(np.prod(obs_shape))

        labeler = None
        if use_corac:
            labeler = CoRACLabeler(
                config, obs_dim=obs_dim, n_actions=n_actions, n_agents=n_agents,
                batch_size=E, n_steps=ep_len,
            )
            context_dim = labeler.context_dim
        else:
            context_dim = slot_number

        network = QNetwork(action_dim=n_actions, hidden_size=config["HIDDEN_SIZE"])
        twin = TwinQNetwork(
            action_dim=n_actions, hidden_size=config["HIDDEN_SIZE"],
            head=twin_head, hyper_hidden=twin_hyper_hidden,
        )
        apply_q = jax.vmap(network.apply, in_axes=(None, 0))
        apply_twin = jax.vmap(twin.apply, in_axes=(None, 0, 0))

        def create_agent(rng):
            rng_a, rng_t = jax.random.split(rng)
            init_x = jnp.zeros((1, *obs_shape))
            params = {
                "agent": network.init(rng_a, init_x),
                "twin": twin.init(rng_t, init_x, jnp.zeros((1, 1, context_dim))),
            }
            lr_scheduler = optax.linear_schedule(
                config["LR"], 1e-10, (config["NUM_EPOCHS"]) * config["NUM_UPDATES"]
            )
            lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]
            return CustomTrainState.create(
                apply_fn=network.apply, params=params, target_network_params=params,
                tx=make_optimizer(config, lr),
            )

        rng, _rng = jax.random.split(rng)
        train_state = create_agent(_rng)
        rng, _rng = jax.random.split(rng)
        ctx_state = labeler.init_state(_rng) if use_corac else {}

        # BUFFER: flat transitions, one row per env, labelled when their episode ends
        item = {
            "obs": jnp.zeros((n_agents, *obs_shape)),
            "actions": jnp.zeros((n_agents,), jnp.int32),
            "avail": jnp.zeros((n_agents, n_actions)),
            "rewards": jnp.zeros((n_agents,)),
            "done": jnp.zeros((), jnp.float32),
            "ep_return": jnp.zeros((n_agents,)),
            "rtg_low": jnp.zeros(()),
            "rtg_high": jnp.zeros(()),
        }
        if use_corac:
            item["context"] = jnp.zeros((n_agents, context_dim))
        buffer = fbx.make_flat_buffer(
            max_length=int(config["BUFFER_SIZE"]),
            min_length=int(config["BUFFER_BATCH_SIZE"]),
            sample_batch_size=int(config["BUFFER_BATCH_SIZE"]),
            add_sequences=True,
            add_batch_size=E,
        )
        buffer = buffer.replace(
            init=jax.jit(buffer.init),
            add=jax.jit(buffer.add, donate_argnums=0),
            sample=jax.jit(buffer.sample),
            can_sample=jax.jit(buffer.can_sample),
        )
        buffer_state = buffer.init(item)

        # STAGING: the current episode of every env, (ep_len, N, E, ...)
        staging = {
            "obs": jnp.zeros((ep_len, n_agents, E, *obs_shape)),
            "actions": jnp.zeros((ep_len, n_agents, E), jnp.int32),
            "avail": jnp.zeros((ep_len, n_agents, E, n_actions)),
            "rewards": jnp.zeros((ep_len, n_agents, E)),
            "done": jnp.zeros((ep_len, E)),
        }

        def flush(buffer_state, ctx_state, staging, rng):
            """Label the finished episodes and add them to the buffer."""
            rewards = jnp.swapaxes(staging["rewards"], 0, 1)  # (N, T, E)
            actions = jnp.swapaxes(staging["actions"], 0, 1)
            ep_dones = jnp.zeros((ep_len, E), dtype=bool)  # one episode per row
            ep_return = rewards.sum(axis=1)  # (N, E)
            low, high = discounted_return_bounds(rewards, ep_dones, config["GAMMA"])
            to_rows = lambda x: jnp.swapaxes(x, 0, 2)  # (T, N, E, ...) -> (E, N, T, ...)
            seq = {
                "obs": jnp.swapaxes(to_rows(staging["obs"]), 1, 2),  # (E, T, N, ...)
                "actions": jnp.swapaxes(to_rows(staging["actions"]), 1, 2),
                "avail": jnp.swapaxes(to_rows(staging["avail"]), 1, 2),
                "rewards": jnp.swapaxes(to_rows(staging["rewards"]), 1, 2),
                "done": staging["done"].T,  # (E, T)
                "ep_return": jnp.broadcast_to(ep_return.T[:, None], (E, ep_len, n_agents)),
                "rtg_low": jnp.full((E, ep_len), low),
                "rtg_high": jnp.full((E, ep_len), high),
            }
            stats = {}
            if use_corac:
                obs = jnp.swapaxes(staging["obs"], 0, 1).reshape(n_agents, ep_len, E, obs_dim)
                contexts, ctx_state, stats = labeler.label(
                    rng, ctx_state, obs.astype(jnp.float32), actions, rewards, ep_dones
                )  # (N, T, E, C)
                seq["context"] = jnp.transpose(contexts, (2, 1, 0, 3))  # (E, T, N, C)
                ctx_state = {**{k: v for k, v in ctx_state.items() if k != "stats"}, "stats": stats}
            return buffer.add(buffer_state, seq), ctx_state

        if use_corac:  # zero stats so the carried ctx_state keeps one structure
            stats_shape = jax.eval_shape(
                lambda b, c, s, r: flush(b, c, s, r)[1]["stats"],
                buffer_state, ctx_state, staging, jax.random.PRNGKey(0),
            )
            ctx_state = {**ctx_state, "stats": jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), stats_shape)}

        def _update_step(runner_state, unused):

            train_state, ctx_state, buffer_state, staging, step_in_ep, expl_state, test_state, rng = runner_state

            # SAMPLE PHASE (behaviour Q only, exactly as iql_cnn_overcooked)
            def _step_env(carry, _):
                last_obs, env_state, staging, step_in_ep, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                q_vals = apply_q(train_state.params["agent"], batchify(last_obs))
                avail_actions = wrapped_env.get_valid_actions(env_state.env_state)
                eps = eps_scheduler(train_state.n_updates)
                _rngs = jax.random.split(rng_a, env.num_agents)
                new_action = jax.vmap(eps_greedy_exploration, in_axes=(0, 0, None, 0))(
                    _rngs, q_vals, eps, batchify(avail_actions)
                )
                actions = unbatchify(new_action)
                new_obs, new_env_state, rewards, dones, infos = wrapped_env.batch_step(
                    rng_s, env_state, actions
                )
                shaped_reward = infos.pop("shaped_reward")
                shaped_reward["__all__"] = batchify(shaped_reward).sum(axis=0)
                rewards = jax.tree.map(
                    lambda x, y: x + y * rew_shaping_anneal(train_state.timesteps),
                    rewards,
                    shaped_reward,
                )
                t = step_in_ep
                staging = {
                    "obs": staging["obs"].at[t].set(batchify(last_obs)),
                    "actions": staging["actions"].at[t].set(new_action),
                    "avail": staging["avail"].at[t].set(batchify(avail_actions)),
                    "rewards": staging["rewards"].at[t].set(batchify(rewards)),
                    "done": staging["done"].at[t].set(dones["__all__"].astype(jnp.float32)),
                }
                return (new_obs, new_env_state, staging, (t + 1) % ep_len, rng), infos

            rng, _rng = jax.random.split(rng)
            (obs, env_state, staging, step_in_ep, _), infos = jax.lax.scan(
                _step_env, (*expl_state, staging, step_in_ep, _rng), None, config["NUM_STEPS"]
            )
            expl_state = (obs, env_state)
            train_state = train_state.replace(
                timesteps=train_state.timesteps + config["NUM_STEPS"] * config["NUM_ENVS"]
            )

            # BUFFER UPDATE: once the episodes of all envs are complete
            rng, _rng = jax.random.split(rng)
            buffer_state, ctx_state = jax.lax.cond(
                step_in_ep == 0,
                lambda b, c: flush(b, c, staging, _rng),
                lambda b, c: (b, c),
                buffer_state,
                ctx_state,
            )

            # NETWORKS UPDATE
            def _learn_phase(carry, _):
                train_state, rng = carry
                rng, _rng = jax.random.split(rng)
                mb = buffer.sample(buffer_state, _rng).experience
                first, second = mb.first, mb.second
                sw = lambda x: jnp.swapaxes(x, 0, 1)  # (B, N, ...) -> (N, B, ...)
                obs, next_obs = sw(first["obs"]), sw(second["obs"])
                actions, rewards, avail = sw(first["actions"]), sw(first["rewards"]), sw(first["avail"])
                not_done = 1.0 - first["done"][None]  # (1, B)

                if use_corac:
                    contexts = sw(first["context"])  # (N, B, C)
                    bank = ctx_state["bank"][:, None]  # (N, 1, M, C)
                    bank_valid = labeler.bank_valid(ctx_state)[:, None]  # (N, 1, M)
                    stats = {}
                else:
                    contexts, slots = return_bin_contexts(
                        sw(first["ep_return"]), return_min, return_max, slot_number
                    )
                    bank = jnp.broadcast_to(
                        jnp.eye(slot_number), (n_agents, 1, slot_number, slot_number)
                    )
                    bank_valid = jnp.ones((n_agents, 1, slot_number), dtype=bool)
                    frac = jax.nn.one_hot(slots, slot_number).mean(axis=(0, 1))
                    stats = {f"return_slot_{i}_fraction": frac[i] for i in range(slot_number)}
                contexts = contexts[:, :, None, :]  # (N, B, 1, C)

                ret_low, ret_high = first["rtg_low"].min(), first["rtg_high"].max()
                kl_temperature = (
                    jnp.maximum(1e-3, kl_temp_frac * jnp.maximum(ret_high - ret_low, 1e-3))
                    if kl_temp_frac > 0
                    else 1.0
                )

                # optimistic teacher: twin Q maximised over the contexts (no grad)
                teacher_q = select_teacher_q(
                    apply_twin(train_state.params["twin"], obs, bank),
                    avail, bank_valid, teacher_mode, teacher_topk,
                )
                if twin_clamp:
                    teacher_q = jnp.clip(teacher_q, ret_low, ret_high)
                teacher_q = jax.lax.stop_gradient(teacher_q)

                tp = train_state.target_network_params
                q_next = apply_q(tp["agent"], next_obs).max(-1)  # (N, B)
                twin_next = apply_twin(tp["twin"], next_obs, contexts)[..., 0, :].max(-1)
                target = rewards + not_done * config["GAMMA"] * q_next
                twin_target = rewards + not_done * config["GAMMA"] * twin_next
                if twin_clamp:
                    twin_target = jnp.clip(twin_target, ret_low, ret_high)
                kl_w = kl_weight_at(train_state.timesteps)

                def _loss_fn(params):
                    q_vals = apply_q(params["agent"], obs)  # (N, B, A)
                    twin_q = apply_twin(params["twin"], obs, contexts)[..., 0, :]
                    pick = lambda q: jnp.take_along_axis(q, actions[..., None], axis=-1)[..., 0]
                    chosen, twin_chosen = pick(q_vals), pick(twin_q)
                    td_loss = jnp.mean((chosen - jax.lax.stop_gradient(target)) ** 2)
                    twin_td_loss = jnp.mean((twin_chosen - jax.lax.stop_gradient(twin_target)) ** 2)
                    kl_loss = categorical_kl(q_vals / kl_temperature, teacher_q / kl_temperature).mean()
                    loss = td_loss + twin_td_loss + kl_w * kl_loss
                    return loss, {
                        "td_loss": td_loss,
                        "twin_td_loss": twin_td_loss,
                        "kl_loss": kl_loss,
                        "qvals": chosen.mean(),
                        "twin_qvals": twin_chosen.mean(),
                        "twin_target_mean": twin_target.mean(),
                    }

                (loss, aux), grads = jax.value_and_grad(_loss_fn, has_aux=True)(train_state.params)
                train_state = train_state.apply_gradients(grads=grads)
                train_state = train_state.replace(grad_steps=train_state.grad_steps + 1)
                aux.update(stats)
                aux.update({"loss": loss, "kl_weight": kl_w, "teacher_q_mean": teacher_q.mean()})
                return (train_state, rng), aux

            def _learn(ts, r):
                return jax.lax.scan(_learn_phase, (ts, r), None, config["NUM_EPOCHS"])

            def _no_learn(ts, r):
                aux_shape = jax.eval_shape(_learn, ts, r)[1]
                return (ts, r), jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), aux_shape)

            rng, _rng = jax.random.split(rng)
            is_learn_time = buffer.can_sample(buffer_state) & (
                train_state.timesteps > config["LEARNING_STARTS"]
            )
            (train_state, rng), learn_metrics = jax.lax.cond(
                is_learn_time, _learn, _no_learn, train_state, _rng
            )

            train_state = jax.lax.cond(
                train_state.n_updates % config["TARGET_UPDATE_INTERVAL"] == 0,
                lambda ts: ts.replace(
                    target_network_params=optax.incremental_update(
                        ts.params, ts.target_network_params, config["TAU"]
                    )
                ),
                lambda ts: ts,
                operand=train_state,
            )

            train_state = train_state.replace(n_updates=train_state.n_updates + 1)
            metrics = {
                "env_step": train_state.timesteps,
                "update_steps": train_state.n_updates,
                "grad_steps": train_state.grad_steps,
            }
            metrics.update(jax.tree.map(lambda x: x.mean(), learn_metrics))
            if use_corac:
                metrics.update(ctx_state["stats"])
            metrics.update(jax.tree.map(lambda x: x.mean(), infos))

            if config.get("TEST_DURING_TRAINING", True):
                rng, _rng = jax.random.split(rng)
                test_state = jax.lax.cond(
                    train_state.n_updates % int(config["NUM_UPDATES"] * config["TEST_INTERVAL"]) == 0,
                    lambda _: get_greedy_metrics(_rng, train_state),
                    lambda _: test_state,
                    operand=None,
                )
                metrics.update({"test_" + k: v for k, v in test_state.items()})

            if config["WANDB_MODE"] != "disabled":

                def callback(metrics, original_seed):
                    if config.get("WANDB_LOG_ALL_SEEDS", False):
                        metrics.update({f"rng{int(original_seed)}/{k}": v for k, v in metrics.items()})
                    wandb.log(metrics, step=metrics["update_steps"])

                jax.debug.callback(callback, metrics, original_seed)

            runner_state = (train_state, ctx_state, buffer_state, staging, step_in_ep, expl_state, test_state, rng)
            return runner_state, None

        def get_greedy_metrics(rng, train_state):
            """Greedy behaviour Q, as iql_cnn_overcooked."""

            def _greedy_env_step(step_state, unused):
                last_obs, env_state, rng = step_state
                rng, rng_s = jax.random.split(rng)
                q_vals = apply_q(train_state.params["agent"], batchify(last_obs))
                actions = unbatchify(jnp.argmax(q_vals, axis=-1))
                new_obs, new_env_state, rewards, dones, infos = test_env.batch_step(
                    rng_s, env_state, actions
                )
                return (new_obs, new_env_state, rng), infos

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = test_env.batch_reset(_rng)
            _, infos = jax.lax.scan(
                _greedy_env_step, (init_obs, env_state, rng), None, config["TEST_NUM_STEPS"]
            )
            return {
                "returned_episode_returns": jnp.nanmean(
                    jnp.where(infos["returned_episode"], infos["returned_episode_returns"], jnp.nan)
                )
            }

        rng, _rng = jax.random.split(rng)
        test_state = get_greedy_metrics(_rng, train_state)
        rng, _rng = jax.random.split(rng)
        obs, env_state = wrapped_env.batch_reset(_rng)
        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state, ctx_state, buffer_state, staging, jnp.zeros((), jnp.int32),
            (obs, env_state), test_state, _rng,
        )
        runner_state, _ = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state}

    return train


def single_run(config):

    config = {**config, **config["alg"]}  # merge the alg config with the main config
    print("Config:\n", OmegaConf.to_yaml(config))

    alg_name = config.get("ALG_NAME", "rac_cnn")
    env, env_name = env_from_config(copy.deepcopy(config))

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[alg_name.upper(), env_name.upper(), f"jax_{jax.__version__}"],
        name=f"{alg_name}_{env_name}",
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config, env)))
    outs = jax.block_until_ready(train_vjit(rngs))

    if config.get("SAVE_PATH", None) is not None:
        from jaxmarl.wrappers.baselines import save_params

        model_state = outs["runner_state"][0]
        save_dir = os.path.join(config["SAVE_PATH"], env_name)
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(
            config, os.path.join(save_dir, f"{alg_name}_{env_name}_seed{config['SEED']}_config.yaml")
        )
        for i, _ in enumerate(rngs):
            params = jax.tree.map(lambda x: x[i], model_state.params)
            save_params(
                params,
                os.path.join(save_dir, f"{alg_name}_{env_name}_seed{config['SEED']}_vmap{i}.safetensors"),
            )


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))
    single_run(config)


if __name__ == "__main__":
    main()
