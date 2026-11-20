"""I2Q on Overcooked with the learner of ``iql_cnn_overcooked.py``.

Everything IQL-CNN does is kept: the CNN Q-network, 1-step rollouts from
persistent envs, the flat transition buffer, the max-over-actions target
network and the annealed reward shaping. On top of it every agent i learns, as
in ``i2q_rnn.py`` (Jiang & Lu, NeurIPS 2022),

  * Q_ss^i(h, h')  regressed to  r + gamma * V^i_target(h'),
  * V^i(h)         the tau-expectile of Q_ss^i_target(h, h'),

and its Q target blends the two:

  y = lambda * (r + gamma * V^i_target(h')) + (1 - lambda) * (r + gamma * max_a Q^i_target(o', a)).

h is the agent's feature from the target Q-network (the output of the hidden
Dense layer after the CNN). Q_ss and V are separate networks per agent, the
Q-network shares parameters across agents as in IQL-CNN.

  python baselines/QLearning/i2q_cnn_overcooked.py +alg=i2q_cnn_overcooked
"""

import copy
import os
from typing import Any

import chex
import flashbax as fbx
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

import wandb

from i2q_rnn import QSS, VS, make_optimizer
from iql_cnn_overcooked import CNN, env_from_config
from jaxmarl.wrappers.baselines import CTRolloutManager


class QNetwork(nn.Module):
    """IQL-CNN's Q-network (same layers and parameter names) that also returns h."""

    action_dim: int
    hidden_size: int = 64

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        embedding = CNN()(x)
        h = nn.Dense(self.hidden_size)(embedding)
        return nn.Dense(self.action_dim)(h), h


@chex.dataclass(frozen=True)
class Timestep:
    obs: dict
    actions: dict
    avail_actions: dict
    rewards: dict
    dones: dict


class CustomTrainState(TrainState):
    target_network_params: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


class AuxTrainState(TrainState):
    target_params: Any


def make_train(config, env):

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    lamb = float(config.get("I2Q_LAMBDA", 0.2))
    tau = float(config.get("I2Q_TAU", 0.52))
    i2q_hidden = int(config.get("I2Q_HIDDEN_DIM", 256))

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
        H = config["HIDDEN_SIZE"]

        network = QNetwork(action_dim=wrapped_env.max_action_space, hidden_size=H)
        qss = QSS(hidden_dim=i2q_hidden)
        vs = VS(hidden_dim=i2q_hidden)
        apply_q = jax.vmap(network.apply, in_axes=(None, 0))  # shared params
        apply_qss = jax.vmap(qss.apply, in_axes=(0, 0, 0))  # per-agent params
        apply_vs = jax.vmap(vs.apply, in_axes=(0, 0))

        lr_scheduler = optax.linear_schedule(
            config["LR"], 1e-10, (config["NUM_EPOCHS"]) * config["NUM_UPDATES"]
        )
        lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]

        def create_agent(rng):
            rng_q, rng_ss, rng_v = jax.random.split(rng, 3)
            q_params = network.init(rng_q, jnp.zeros((1, *env.observation_space("agent_0").shape)))
            dummy_h = jnp.zeros((1, H))
            qss_params = jax.vmap(qss.init, in_axes=(0, None, None))(
                jax.random.split(rng_ss, n_agents), dummy_h, dummy_h
            )
            vs_params = jax.vmap(vs.init, in_axes=(0, None))(
                jax.random.split(rng_v, n_agents), dummy_h
            )
            q_state = CustomTrainState.create(
                apply_fn=network.apply, params=q_params, target_network_params=q_params,
                tx=make_optimizer(config, lr),
            )
            qss_state = AuxTrainState.create(
                apply_fn=qss.apply, params=qss_params, target_params=qss_params,
                tx=make_optimizer(config, lr),
            )
            vs_state = AuxTrainState.create(
                apply_fn=vs.apply, params=vs_params, target_params=vs_params,
                tx=make_optimizer(config, lr),
            )
            return q_state, qss_state, vs_state

        rng, _rng = jax.random.split(rng)
        train_state, qss_state, vs_state = create_agent(_rng)

        # INIT BUFFER (as iql_cnn_overcooked)
        buffer = fbx.make_flat_buffer(
            max_length=int(config["BUFFER_SIZE"]),
            min_length=int(config["BUFFER_BATCH_SIZE"]),
            sample_batch_size=int(config["BUFFER_BATCH_SIZE"]),
            add_sequences=False,
            add_batch_size=int(config["NUM_ENVS"] * config["NUM_STEPS"]),
        )
        buffer = buffer.replace(
            init=jax.jit(buffer.init),
            add=jax.jit(buffer.add, donate_argnums=0),
            sample=jax.jit(buffer.sample),
            can_sample=jax.jit(buffer.can_sample),
        )
        _obs, _env_state = wrapped_env.batch_reset(_rng)
        _actions = {agent: wrapped_env.batch_sample(_rng, agent) for agent in env.agents}
        _obs, _, _rewards, _dones, _infos = wrapped_env.batch_step(_rng, _env_state, _actions)
        _timestep = Timestep(
            obs=_obs,
            actions=_actions,
            avail_actions=wrapped_env.get_valid_actions(_env_state.env_state),
            rewards=_rewards,
            dones=_dones,
        )
        buffer_state = buffer.init(jax.tree.map(lambda x: x[0], _timestep))

        def _update_step(runner_state, unused):

            train_state, qss_state, vs_state, buffer_state, expl_state, test_state, rng = runner_state

            # SAMPLE PHASE (as iql_cnn_overcooked)
            def _step_env(carry, _):
                last_obs, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                q_vals, _ = apply_q(train_state.params, batchify(last_obs))
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
                timestep = Timestep(
                    obs=last_obs, actions=actions, avail_actions=avail_actions,
                    rewards=rewards, dones=dones,
                )
                return (new_obs, new_env_state, rng), (timestep, infos)

            rng, _rng = jax.random.split(rng)
            carry, (timesteps, infos) = jax.lax.scan(
                _step_env, (*expl_state, _rng), None, config["NUM_STEPS"]
            )
            expl_state = carry[:2]
            train_state = train_state.replace(
                timesteps=train_state.timesteps + config["NUM_STEPS"] * config["NUM_ENVS"]
            )
            buffer_state = buffer.add(
                buffer_state, jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), timesteps)
            )

            # NETWORKS UPDATE
            def _learn_phase(carry, _):
                train_state, qss_state, vs_state, rng = carry
                rng, _rng = jax.random.split(rng)
                minibatch = buffer.sample(buffer_state, _rng).experience
                obs = batchify(minibatch.first.obs)  # (N, B, ...)
                next_obs = batchify(minibatch.second.obs)
                actions = batchify(minibatch.first.actions)
                rewards = batchify(minibatch.first.rewards)
                cont = 1.0 - batchify(minibatch.first.dones)

                tp = train_state.target_network_params
                _, hid = apply_q(tp, obs)
                q_next_target, next_hid = apply_q(tp, next_obs)
                # r + gamma * V(h'): the value of the transition actually reached
                target_v = rewards + config["GAMMA"] * cont * apply_vs(vs_state.target_params, next_hid)

                def _qss_loss(params):
                    q_ss = apply_qss(params, hid, next_hid)
                    return jnp.mean((q_ss - target_v) ** 2), q_ss.mean()

                (loss_qss, qss_mean), grads = jax.value_and_grad(_qss_loss, has_aux=True)(qss_state.params)
                qss_state = qss_state.apply_gradients(grads=grads)

                q_ss_target = apply_qss(qss_state.target_params, hid, next_hid)

                def _vs_loss(params):
                    v = apply_vs(params, hid)
                    w = jnp.where(q_ss_target > jax.lax.stop_gradient(v), tau, 1.0 - tau)
                    return (w * (v - q_ss_target) ** 2).sum() / w.sum(), v.mean()

                (loss_vs, v_mean), grads = jax.value_and_grad(_vs_loss, has_aux=True)(vs_state.params)
                vs_state = vs_state.apply_gradients(grads=grads)

                td_target = rewards + config["GAMMA"] * cont * q_next_target.max(-1)
                target = jax.lax.stop_gradient(lamb * target_v + (1.0 - lamb) * td_target)

                def _loss_fn(params):
                    q_vals, _ = apply_q(params, obs)
                    chosen = jnp.take_along_axis(q_vals, actions[..., None], axis=-1)[..., 0]
                    return jnp.mean((chosen - target) ** 2), chosen.mean()

                (loss, qvals), grads = jax.value_and_grad(_loss_fn, has_aux=True)(train_state.params)
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
                    "i2q_td_target_mean": td_target.mean(),
                    "target_mean": target.mean(),
                }
                return (train_state, qss_state, vs_state, rng), aux

            def _learn(ts, qs, vs_, r):
                return jax.lax.scan(_learn_phase, (ts, qs, vs_, r), None, config["NUM_EPOCHS"])

            def _no_learn(ts, qs, vs_, r):
                aux_shape = jax.eval_shape(_learn, ts, qs, vs_, r)[1]
                return (ts, qs, vs_, r), jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), aux_shape)

            rng, _rng = jax.random.split(rng)
            is_learn_time = buffer.can_sample(buffer_state) & (
                train_state.timesteps > config["LEARNING_STARTS"]
            )
            (train_state, qss_state, vs_state, rng), learn_metrics = jax.lax.cond(
                is_learn_time, _learn, _no_learn, train_state, qss_state, vs_state, _rng
            )

            # target networks (Q, Q_ss and V together)
            do_update = train_state.n_updates % config["TARGET_UPDATE_INTERVAL"] == 0

            def _soft(new, old):
                return jax.tree.map(
                    lambda n, o: jnp.where(do_update, n, o),
                    optax.incremental_update(new, old, config["TAU"]),
                    old,
                )

            train_state = train_state.replace(
                target_network_params=_soft(train_state.params, train_state.target_network_params)
            )
            qss_state = qss_state.replace(target_params=_soft(qss_state.params, qss_state.target_params))
            vs_state = vs_state.replace(target_params=_soft(vs_state.params, vs_state.target_params))

            train_state = train_state.replace(n_updates=train_state.n_updates + 1)
            metrics = {
                "env_step": train_state.timesteps,
                "update_steps": train_state.n_updates,
                "grad_steps": train_state.grad_steps,
            }
            metrics.update(jax.tree.map(lambda x: x.mean(), learn_metrics))
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

            runner_state = (train_state, qss_state, vs_state, buffer_state, expl_state, test_state, rng)
            return runner_state, None

        def get_greedy_metrics(rng, train_state):
            """Greedy Q, as iql_cnn_overcooked."""

            def _greedy_env_step(step_state, unused):
                last_obs, env_state, rng = step_state
                rng, rng_s = jax.random.split(rng)
                q_vals, _ = apply_q(train_state.params, batchify(last_obs))
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
        runner_state = (train_state, qss_state, vs_state, buffer_state, (obs, env_state), test_state, _rng)
        runner_state, _ = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state}

    return train


def single_run(config):

    config = {**config, **config["alg"]}  # merge the alg config with the main config
    print("Config:\n", OmegaConf.to_yaml(config))

    alg_name = config.get("ALG_NAME", "i2q_cnn")
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
