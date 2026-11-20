"""RAC and HC-RAC: fully decentralized Q-learning with return-conditioned twins.

RAC (Li et al., NeurIPS 2025, "In-Context Fully Decentralized Cooperative
Multi-Agent Reinforcement Learning") trains, for every agent, two Q-networks on
its own local history only:

  * the behaviour Q(tau, a), the one used to act, and
  * a twin Q(tau, c, a) conditioned on a context c of the episode the step came
    from. In RAC c is a one-hot bin of the episode return (``SLOT_NUMBER``
    equal-width bins over [``RETURN_MIN``, ``RETURN_MAX``]).

Both are trained with 1-step double-Q TD, and the behaviour Q is pulled towards
an optimistic teacher, the twin's Q maximised over contexts, through
``KL_WEIGHT * KL(softmax(Q) || softmax(Q_teacher))``. Maximising over contexts
asks "what would I do if my teammates behaved as in the best episodes seen so
far", which lets independent learners escape relative overgeneralisation.

HC-RAC (``CONTEXT_LABELER: corac``, preset ``hcrac_rnn_*``) replaces the one-hot
bins with a continuous context ``c = [phi(R) ; u]``: cosine features of the
episode return plus a trajectory embedding learned with Trajectory Contrastive
Learning (positives: two windows of the same episode; negatives: other episodes
and "outcome-swapped" windows). The twin reads c through a zero-delta
hypernetwork and the teacher maximises over a FIFO bank of recent contexts. The
``hcrac`` presets use a fixed return range and an EMA context encoder (step A of
the HC-RAC stability ladder), a constant kl_weight and RAC's per-action teacher
(``action_max``); ``KL_WEIGHT_FINAL`` and ``TEACHER_CONTEXT_SELECT:
context_argmax`` give ladder steps B and C.

Execution only uses the behaviour Q, exactly as IQL. The rollout, buffer and
testing code follows ``iql_rnn.py``. Episodes are recovered inside every stored
rollout from ``dones["__all__"]``; the return of the last, unfinished episode of
a rollout is the partial return observed so far.
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
from flax.linen.initializers import constant, orthogonal, zeros
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


# --------------------------------------------------------------------------- #
# Networks
# --------------------------------------------------------------------------- #
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
    """Behaviour Q(tau, a): the network used to act (identical to IQL's)."""

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
        hidden, embedding = ScannedRNN()(hidden, (embedding, dones))
        q_vals = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(embedding)
        return hidden, q_vals


class TwinRNNQNetwork(nn.Module):
    """Context-conditioned twin Q(tau, c, a).

    The recurrent core never sees the context, so the Q-values for a whole bank
    of contexts share one pass over the sequence. ``contexts`` has shape
    (..., M, C) broadcastable to (T, B, M, C); the output is (T, B, M, A).

    head = "normal": Q = W [h ; c] + b (RAC's Normal-Net, used on SMAC).
    head = "hyper":  Q = h W(c) + b(c) with W, b linear in z(c). With
        ``hyper_hidden == 0`` z = c (RAC's Hyper-Net); with ``hyper_hidden > 0``
        z = relu(Dense(c)) and the context-dependent part starts at zero, so every
        context initially shares one base head and a max over contexts starts
        without optimism (HC-RAC's zero-delta hypernetwork).
    Because W(c) is linear in z, Q = h W0 + b0 + z (h K) + z Kb, which never
    materialises a (T, B, M, H, A) weight tensor.
    """

    action_dim: int
    hidden_dim: int
    head: str = "hyper"
    hyper_hidden: int = 0
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, hidden, obs, dones, contexts):
        embedding = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(obs)
        embedding = nn.relu(embedding)
        hidden, h = ScannedRNN()(hidden, (embedding, dones))  # (T, B, H)

        if self.head == "normal":
            q_h = nn.Dense(
                self.action_dim,
                kernel_init=orthogonal(self.init_scale),
                bias_init=constant(0.0),
                name="q_hidden",
            )(h)
            q_c = nn.Dense(
                self.action_dim,
                use_bias=False,
                kernel_init=orthogonal(self.init_scale),
                name="q_context",
            )(contexts)
            return hidden, q_h[..., None, :] + q_c

        if self.head != "hyper":
            raise ValueError(
                f"TWIN_HEAD must be 'normal' or 'hyper', got {self.head!r}."
            )

        H, A = self.hidden_dim, self.action_dim
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

        hk = jnp.einsum("tbh,kha->tbka", h, w_delta)  # (T, B, K, A)
        q = jnp.matmul(z, hk)  # (T, B, M, A)
        q = q + (h @ w_base + b_base)[..., None, :] + z @ b_delta
        return hidden, q


class TCLEncoder(nn.Module):
    """Window encoder of HC-RAC's Trajectory Contrastive Learning.

    (..., L, F) -> unit vectors (..., d): per-step MLP, attention pooling over
    the window, linear projection, L2 normalisation.
    """

    hidden_dim: int
    embed_dim: int

    @nn.compact
    def __call__(self, segments):
        x = nn.relu(nn.Dense(self.hidden_dim)(segments))
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        weights = jax.nn.softmax(nn.Dense(1)(x), axis=-2)
        pooled = (weights * x).sum(axis=-2)
        out = nn.Dense(self.embed_dim)(pooled)
        return out / jnp.maximum(jnp.linalg.norm(out, axis=-1, keepdims=True), 1e-12)


# --------------------------------------------------------------------------- #
# Episode bookkeeping inside a stored rollout
# --------------------------------------------------------------------------- #
def episode_segments(ep_dones):
    """Segment id of every step, (T, B) -> (T, B) int32.

    ``ep_dones[t]`` is True when obs t is the first obs of a new episode (the
    auto-reset obs), so the reward of step t belongs to segment ``seg[t]``.
    """
    return jnp.cumsum(ep_dones.astype(jnp.int32), axis=0)


def segment_returns(rewards, seg):
    """Sum of rewards per segment.

    rewards (N, T, B), seg (T, B) -> (N, S, B) with S = T + 1 possible segments.
    """
    T = rewards.shape[1]

    def one(r, s):  # (T,), (T,)
        return jax.ops.segment_sum(r, s, num_segments=T + 1)

    per_b = jax.vmap(one, in_axes=(1, 1), out_axes=1)  # over B
    return jax.vmap(per_b, in_axes=(0, None))(rewards, seg)  # over N


def gather_segments(values, seg):
    """values (N, S, B, ...) at seg (T, B) -> (N, T, B, ...)."""

    def one(v, s):  # (S, ...), (T,)
        return v[s]

    per_b = jax.vmap(one, in_axes=(1, 1), out_axes=1)
    return jax.vmap(per_b, in_axes=(0, None))(values, seg)


def discounted_return_bounds(rewards, ep_dones, gamma):
    """(low, high) of the discounted return-to-go over every step of the batch.

    Q is a discounted value, so its realisable range is that of the discounted
    return-to-go. Both ends include 0 (the value past termination).
    rewards (N, T, B), ep_dones (T, B).
    """
    # cont[t] = 0 when obs t+1 starts a new episode (and past the last step).
    cont = 1.0 - ep_dones[1:].astype(jnp.float32)  # (T-1, B)
    cont = jnp.concatenate([cont, jnp.zeros_like(cont[:1])], axis=0)  # (T, B)

    def step(running, inp):
        r, c = inp  # (N, B), (B,)
        running = r + gamma * c[None] * running
        return running, running

    _, to_go = jax.lax.scan(
        step,
        jnp.zeros_like(rewards[:, 0]),
        (jnp.swapaxes(rewards, 0, 1), cont),
        reverse=True,
    )
    return jnp.minimum(0.0, to_go.min()), jnp.maximum(0.0, to_go.max())


def return_bin_contexts(returns, return_min, return_max, slot_number):
    """RAC's one-hot context: equal-width bins over [return_min, return_max]."""
    edges = jnp.linspace(return_min, return_max, slot_number + 1)
    idx = jnp.clip(jnp.digitize(returns, edges, right=False) - 1, 0, slot_number - 1)
    return jax.nn.one_hot(idx, slot_number), idx


def select_teacher_q(context_q, avail_actions, bank_valid, mode, topk):
    """Teacher Q over actions from Q for every context.

    context_q (..., M, A), avail_actions (..., A), bank_valid broadcastable to
    (..., M). Returns (..., A).

    "action_max" (RAC) maxes over contexts separately for every action.
    "context_argmax" picks one context per step, the one with the highest
    greedy value, and keeps that context's whole action distribution.
    "topk_mean" averages the k best valid contexts per action.
    """
    neg_inf = -1e10
    valid = bank_valid[..., None]  # (..., M, 1)
    if mode == "action_max":
        return jnp.where(valid, context_q, neg_inf).max(axis=-2)
    if mode == "topk_mean":
        k = min(int(topk), context_q.shape[-2])
        if k <= 0:
            raise ValueError(f"TEACHER_TOPK must be positive, got {topk}.")
        masked = jnp.where(valid, context_q, neg_inf)
        top = jax.lax.top_k(jnp.swapaxes(masked, -1, -2), k)[0]  # (..., A, k)
        ok = top > neg_inf / 2
        return (top * ok).sum(-1) / jnp.maximum(ok.sum(-1), 1)
    if mode != "context_argmax":
        raise ValueError(
            "TEACHER_CONTEXT_SELECT must be 'action_max', 'context_argmax' or "
            f"'topk_mean', got {mode!r}."
        )
    available = avail_actions[..., None, :] > 0  # (..., 1, A)
    context_values = jnp.where(available, context_q, neg_inf).max(axis=-1)  # (..., M)
    context_values = jnp.where(bank_valid, context_values, neg_inf)
    best = jnp.argmax(context_values, axis=-1)
    return jnp.take_along_axis(context_q, best[..., None, None], axis=-2)[..., 0, :]


def categorical_kl(logits_p, logits_q):
    """KL(softmax(p) || softmax(q)) over the last axis."""
    log_p = jax.nn.log_softmax(logits_p, axis=-1)
    log_q = jax.nn.log_softmax(logits_q, axis=-1)
    return (jnp.exp(log_p) * (log_p - log_q)).sum(-1)


# --------------------------------------------------------------------------- #
# HC-RAC context labeler (functional port of the CoRAC labeler)
# --------------------------------------------------------------------------- #
class CoRACLabeler:
    """Continuous contexts c = [phi(R) ; u] and the TCL objective.

    All state lives in a dict (``init_state``) carried through the training
    scan; every method is a pure function of it. Arrays use the replay layout
    (N agents, T steps, B trajectories, ...).
    """

    def __init__(self, config, obs_dim, n_actions, n_agents, batch_size, n_steps):
        self.use_return = bool(config.get("CORAC_USE_RETURN", True))
        self.use_trajectory = bool(config.get("CORAC_USE_TRAJECTORY", True))
        if not (self.use_return or self.use_trajectory):
            raise ValueError(
                "HC-RAC needs at least one context half: enable CORAC_USE_RETURN "
                "and/or CORAC_USE_TRAJECTORY."
            )
        self.n_freq = int(config.get("CORAC_N_FREQ", 16)) if self.use_return else 0
        self.embed_dim = (
            int(config.get("CORAC_EMBED_DIM", 32)) if self.use_trajectory else 0
        )
        self.context_dim = self.n_freq + self.embed_dim
        self.segment_len = int(config.get("CORAC_SEGMENT_LEN", 1))
        self.temperature = float(config.get("CORAC_TEMPERATURE", 0.1))
        self.n_hard = int(config.get("CORAC_N_HARD", 4))
        self.stat_momentum = float(config.get("CORAC_STAT_MOMENTUM", 0.01))
        self.min_std_frac = float(config.get("CORAC_MIN_STD_FRAC", 0.0))
        self.target_tau = float(config.get("CORAC_TARGET_TAU", 0.0))
        self.bank_push = int(config.get("CORAC_BANK_PUSH", 0))
        self.bank_size = int(config.get("CORAC_BANK_SIZE", 64))
        return_range = config.get("CORAC_RETURN_RANGE", None)
        self.return_range = None
        if return_range is not None:
            low, high = (float(v) for v in return_range)
            if low >= high:
                raise ValueError(
                    f"CORAC_RETURN_RANGE must be [low, high] with low < high, got {return_range}."
                )
            self.return_range = (low, high)

        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.batch_size = batch_size
        self.n_steps = n_steps
        if self.use_trajectory and n_steps < 2 * self.segment_len + 1:
            raise ValueError(
                f"CORAC_SEGMENT_LEN={self.segment_len} needs NUM_STEPS >= "
                f"{2 * self.segment_len + 1} to form a contrastive pair."
            )
        self.proj_dim = min(obs_dim, int(config.get("CORAC_PROJ_DIM", 32)))
        self.pool_size = min(
            int(config.get("CORAC_POOL_SIZE", 4096)), batch_size * n_steps * n_agents
        )
        # x_t = [o_t, onehot(a_t), r_t, P(o_{t+1} - o_t)]; the preprocessed obs
        # already carries the agent id one-hot.
        self.n_state_features = obs_dim + n_actions
        self.n_outcome_features = 1 + self.proj_dim
        self.feature_dim = self.n_state_features + self.n_outcome_features
        self.encoder = TCLEncoder(
            hidden_dim=int(config.get("CORAC_HIDDEN_DIM", 64)),
            embed_dim=max(self.embed_dim, 1),
        )
        self.tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(float(config.get("CORAC_LR", 5e-4))),
        )

    # ---------------------------------------------------------------- state
    def init_state(self, rng):
        rng_p, rng_d, rng_k = jax.random.split(rng, 3)
        scale = 1.0 / np.sqrt(self.proj_dim)
        state = {
            "delta_proj": jax.random.normal(rng_d, (self.obs_dim, self.proj_dim))
            * scale,
            "key_proj": jax.random.normal(rng_k, (self.obs_dim, self.proj_dim)) * scale,
            "return_mean": jnp.zeros(()),
            "return_var": jnp.ones(()),
            "stats_init": jnp.zeros((), dtype=bool),
            "bank": jnp.zeros((self.n_agents, self.bank_size, self.context_dim)),
            "bank_count": jnp.zeros((), dtype=jnp.int32),
        }
        if self.use_trajectory:
            dummy = jnp.zeros((1, self.segment_len, self.feature_dim))
            params = self.encoder.init(rng_p, dummy)
            state["params"] = params
            state["target_params"] = params
            state["opt_state"] = self.tx.init(params)
        return state

    def bank_valid(self, state):
        """(N, M) mask of filled bank entries."""
        filled = jnp.minimum(state["bank_count"], self.bank_size)
        valid = jnp.arange(self.bank_size) < filled
        return jnp.broadcast_to(valid, (self.n_agents, self.bank_size))

    # ------------------------------------------------------------- features
    def step_features(self, state, obs, actions, rewards, ep_dones):
        """Per-step features (B, T, N, F) and a (B, T) mask of usable steps.

        A step is usable when its next obs exists and belongs to the same episode,
        so P(o_{t+1} - o_t) never spans an auto-reset.
        """
        # (N, T, B, ...) -> (B, T, N, ...)
        obs = jnp.transpose(obs, (2, 1, 0, 3))
        actions = jnp.transpose(actions, (2, 1, 0))
        rewards = jnp.transpose(rewards, (2, 1, 0))
        dones = ep_dones.T  # (B, T)
        next_obs = jnp.concatenate([obs[:, 1:], obs[:, -1:]], axis=1)
        valid = jnp.concatenate(
            [~dones[:, 1:], jnp.zeros_like(dones[:, :1])], axis=1
        )  # (B, T)
        delta = ((next_obs - obs) @ state["delta_proj"]) * valid[..., None, None]
        features = jnp.concatenate(
            [
                obs,
                jax.nn.one_hot(actions, self.n_actions),
                rewards[..., None],
                delta,
            ],
            axis=-1,
        )
        return features, valid

    def _window_starts(self, valid):
        """(B, S) mask of starts t whose window [t, t+L) only has usable steps."""
        L = self.segment_len
        c = jnp.concatenate(
            [jnp.zeros_like(valid[:, :1], dtype=jnp.int32), jnp.cumsum(valid, axis=1)],
            axis=1,
        )
        S = valid.shape[1] - L + 1
        return (c[:, L : L + S] - c[:, :S]) == L

    def _windows(self, features, starts):
        """features (B, T, N, F), starts (B, K) -> (B, K, N, L, F)."""
        L = self.segment_len

        def one(f, t):
            w = jax.lax.dynamic_slice_in_dim(f, t, L, axis=0)  # (L, N, F)
            return jnp.swapaxes(w, 0, 1)

        return jax.vmap(jax.vmap(one, in_axes=(None, 0)))(features, starts)

    # ------------------------------------------------------- TCL objective
    def _sample_pairs(self, rng, window_ok, seg):
        """One (anchor, positive) pair of disjoint windows of the same episode per trajectory."""
        L = self.segment_len
        S = window_ok.shape[1]
        seg_s = seg[:, :S]
        t = jnp.arange(S)
        same = (
            window_ok[:, :, None]
            & window_ok[:, None, :]
            & (seg_s[:, :, None] == seg_s[:, None, :])
            & (jnp.abs(t[:, None] - t[None, :]) >= L)[None]
        )  # (B, S, S)
        can_anchor = same.any(-1)
        has_pair = can_anchor.any(-1)
        rng1, rng2 = jax.random.split(rng)
        g1 = jax.random.gumbel(rng1, can_anchor.shape)
        t1 = jnp.argmax(jnp.where(can_anchor, g1, -jnp.inf), axis=-1)
        partners = jnp.take_along_axis(same, t1[:, None, None], axis=1)[:, 0]
        g2 = jax.random.gumbel(rng2, partners.shape)
        t2 = jnp.argmax(jnp.where(partners, g2, -jnp.inf), axis=-1)
        return t1, t2, has_pair

    def _hard_negatives(self, rng, state, features, valid, anchor):
        """Anchor (o, a) windows with the outcomes of same-action neighbours.

        anchor (B, N, L, F). Returns hard (B*N, k, L, F) and a (B*N, k) mask.
        """
        B, T, N, F = features.shape
        L = self.segment_len
        o0, o1 = self.n_state_features, self.n_state_features + self.n_outcome_features
        a0, a1 = self.obs_dim, self.obs_dim + self.n_actions

        pool = features.reshape(-1, F)
        pool_b = jnp.broadcast_to(jnp.arange(B)[:, None, None], (B, T, N)).reshape(-1)
        pool_n = jnp.broadcast_to(jnp.arange(N)[None, None, :], (B, T, N)).reshape(-1)
        pool_valid = jnp.broadcast_to(valid[:, :, None], (B, T, N)).reshape(-1)
        rng_p, rng_d = jax.random.split(rng)
        # Prefer valid steps when subsampling the candidate pool.
        order = jnp.argsort(
            jnp.where(pool_valid, 0.0, 1.0)
            + jax.random.uniform(rng_p, pool_valid.shape)
        )[: self.pool_size]
        pool, pool_b, pool_n, pool_valid = (
            pool[order],
            pool_b[order],
            pool_n[order],
            pool_valid[order],
        )
        pool_action = pool[:, a0:a1].argmax(-1)
        pool_key = pool[:, : self.obs_dim] @ state["key_proj"]

        flat = anchor.reshape(-1, F)  # (B*N*L, F)
        flat_b = jnp.repeat(jnp.arange(B), N * L)
        flat_n = jnp.tile(jnp.repeat(jnp.arange(N), L), B)
        flat_action = flat[:, a0:a1].argmax(-1)
        flat_key = flat[:, : self.obs_dim] @ state["key_proj"]

        def pdist(x, y):
            d2 = (x**2).sum(-1)[:, None] + (y**2).sum(-1)[None] - 2 * x @ y.T
            return jnp.sqrt(jnp.maximum(d2, 0.0))

        dist = pdist(flat_key, pool_key)
        dist = dist + 1e-3 * jax.random.uniform(rng_d, dist.shape)
        # A neighbour with the anchor's own outcome would not be a negative.
        new_outcome = pdist(flat[:, o0:o1], pool[:, o0:o1]) > 1e-6
        allowed = (
            (flat_action[:, None] == pool_action[None])
            & (flat_n[:, None] == pool_n[None])
            & (flat_b[:, None] != pool_b[None])
            & pool_valid[None]
            & new_outcome
        )
        dist = jnp.where(allowed, dist, jnp.inf)
        k = min(self.n_hard, dist.shape[1])
        neg_dist, nearest = jax.lax.top_k(-dist, k)
        found = jnp.isfinite(neg_dist)

        outcomes = pool[:, o0:o1][nearest]  # (B*N*L, k, O)
        outcomes = jnp.swapaxes(
            outcomes.reshape(B * N, L, k, -1), 1, 2
        )  # (B*N, k, L, O)
        base = jnp.broadcast_to(anchor.reshape(B * N, 1, L, F), (B * N, k, L, F))
        hard = jnp.concatenate([base[..., :o0], outcomes, base[..., o1:]], axis=-1)
        hard_valid = found.reshape(B * N, L, k).all(axis=1)
        return hard, hard_valid

    def contrastive_loss(self, params, rng, state, features, valid, seg):
        B, T, N, F = features.shape
        window_ok = self._window_starts(valid)
        rng_s, rng_h = jax.random.split(rng)
        t1, t2, has_pair = self._sample_pairs(rng_s, window_ok, seg)
        anchor = self._windows(features, t1[:, None])[:, 0]  # (B, N, L, F)
        positive = self._windows(features, t2[:, None])[:, 0]

        u = self.encoder.apply(params, anchor).reshape(B * N, -1)
        u_pos = self.encoder.apply(params, positive).reshape(B * N, -1)
        row_b = jnp.repeat(jnp.arange(B), N)
        row_ok = has_pair[row_b]

        pos_logit = (u * u_pos).sum(-1, keepdims=True) / self.temperature
        easy = u @ u_pos.T / self.temperature
        # Rows of the same trajectory (the positive, the other agents) and rows
        # without a pair are not negatives.
        easy = jnp.where(
            (row_b[:, None] == row_b[None]) | ~row_ok[None], -jnp.inf, easy
        )
        logits = [pos_logit, easy]
        hard_logits = None
        if self.n_hard > 0:
            hard, hard_valid = self._hard_negatives(
                rng_h, state, features, valid, anchor
            )
            u_hard = self.encoder.apply(params, hard)  # (B*N, k, d)
            hard_logits = (u[:, None] * u_hard).sum(-1) / self.temperature
            hard_logits = jnp.where(hard_valid, hard_logits, -jnp.inf)
            logits.append(hard_logits)
        logits = jnp.concatenate(logits, axis=1)
        nll = -jax.nn.log_softmax(logits, axis=1)[:, 0]
        n_ok = row_ok.sum()
        enough = n_ok >= 2 * N  # at least two trajectories with a pair
        loss = jnp.where(enough, (nll * row_ok).sum() / jnp.maximum(n_ok, 1), 0.0)

        acc = ((logits.argmax(1) == 0) * row_ok).sum() / jnp.maximum(n_ok, 1)
        stats = {"corac_nce_accuracy": acc}
        if hard_logits is not None:
            beats = (pos_logit[:, 0] > hard_logits.max(1)) * row_ok
            stats["corac_pos_beats_hard"] = beats.sum() / jnp.maximum(n_ok, 1)
            stats["corac_hard_valid"] = (hard_logits > -jnp.inf).mean()
        return loss, stats

    # ------------------------------------------------------------- contexts
    def return_features(self, state, returns):
        """phi(R): cosine features of the return squashed to [0, 1]."""
        if self.return_range is not None:
            low, high = self.return_range
            squashed = jnp.clip((returns - low) / (high - low), 0.0, 1.0)
        else:
            std = jnp.sqrt(jnp.maximum(state["return_var"], 1e-6))
            if self.min_std_frac > 0.0:
                std = jnp.maximum(
                    std, self.min_std_frac * jnp.abs(state["return_mean"]) + 1e-3
                )
            z = (returns - state["return_mean"]) / std
            squashed = jax.nn.sigmoid(1.702 * z)  # ~ normal CDF, order preserving
        freqs = jnp.pi * jnp.arange(self.n_freq)
        return jnp.cos(squashed[..., None] * freqs)

    def _update_return_stats(self, state, seg_ret, seg_used):
        w = seg_used.astype(jnp.float32)
        n = jnp.maximum(w.sum(), 1.0)
        mean = (seg_ret * w).sum() / n
        var = (((seg_ret - mean) ** 2) * w).sum() / n
        m = self.stat_momentum
        new_mean = jnp.where(
            state["stats_init"], (1 - m) * state["return_mean"] + m * mean, mean
        )
        new_var = jnp.where(
            state["stats_init"],
            (1 - m) * state["return_var"] + m * var,
            jnp.maximum(var, 1e-6),
        )
        return {
            **state,
            "return_mean": new_mean,
            "return_var": new_var,
            "stats_init": jnp.ones((), dtype=bool),
        }

    def segment_embeddings(self, params, features, valid, seg):
        """Embedding per (trajectory, segment, agent): mean of the episode's windows.

        Windows are aligned to the episode's own first step (non-overlapping, as in
        the original labeler). Segments without a full window get a zero vector.
        Returns (B, S, N, d) with S = T + 1.
        """
        B, T, N, F = features.shape
        L = self.segment_len
        window_ok = self._window_starts(valid)  # (B, Sw)
        Sw = window_ok.shape[1]
        t = jnp.arange(T)
        is_start = jnp.concatenate(
            [jnp.ones((B, 1), dtype=bool), seg[:, 1:] != seg[:, :-1]], axis=1
        )
        seg_start = jax.lax.cummax(jnp.where(is_start, t[None], 0), axis=1)
        aligned = (((t[None] - seg_start) % L) == 0)[:, :Sw] & window_ok

        windows = self._windows(features, jnp.broadcast_to(jnp.arange(Sw), (B, Sw)))
        emb = self.encoder.apply(params, windows)  # (B, Sw, N, d)
        emb = emb * aligned[..., None, None]

        def one(e, a, s):  # (Sw, N, d), (Sw,), (Sw,)
            total = jax.ops.segment_sum(e, s, num_segments=T + 1)
            count = jax.ops.segment_sum(a.astype(jnp.float32), s, num_segments=T + 1)
            return total, count

        total, count = jax.vmap(one)(emb, aligned, seg[:, :Sw])
        mean = total / jnp.maximum(
            jnp.linalg.norm(total, axis=-1, keepdims=True), 1e-12
        )
        return mean * (count > 0)[..., None, None]

    def _push_bank(self, rng, state, contexts):
        """contexts (N, B, C) -> FIFO bank (N, M, C)."""
        M = self.bank_size
        B = contexts.shape[1]
        Bw = min(B, M)
        perm = jax.random.permutation(rng, B)[:Bw]
        new = contexts[:, perm]
        full = state["bank_count"] >= M
        n_write = Bw
        if self.bank_push > 0:
            n_write = jnp.where(full, min(self.bank_push, Bw), Bw)
        idx = (state["bank_count"] + jnp.arange(Bw)) % M
        write = (jnp.arange(Bw) < n_write)[None, :, None]
        values = jnp.where(write, new, state["bank"][:, idx])
        bank = state["bank"].at[:, idx].set(values)
        return {**state, "bank": bank, "bank_count": state["bank_count"] + n_write}

    def label(self, rng, state, obs, actions, rewards, ep_dones):
        """Contexts (N, T, B, C), the updated labeler state and stats.

        rewards (N, T, B) are the agents' own rewards; ep_dones (T, B).
        """
        N, T, B = rewards.shape
        seg = episode_segments(ep_dones)  # (T, B)
        seg_ret = segment_returns(rewards, seg)  # (N, S, B)
        seg_used = jax.vmap(
            lambda s: jax.ops.segment_sum(jnp.ones(T), s, num_segments=T + 1) > 0,
            in_axes=1,
            out_axes=1,
        )(seg)  # (S, B)
        state = self._update_return_stats(
            state, seg_ret, jnp.broadcast_to(seg_used, seg_ret.shape)
        )
        rng_l, rng_b, rng_c = jax.random.split(rng, 3)
        stats = {}

        parts = []
        if self.use_return:
            phi = self.return_features(
                state, gather_segments(seg_ret, seg)
            )  # (N, T, B, n_freq)
            parts.append(phi)
            stats["corac_phi_saturated"] = (jnp.abs(phi) > 0.999).mean()
        if self.use_trajectory:
            features, valid = self.step_features(state, obs, actions, rewards, ep_dones)
            seg_bt = seg.T  # (B, T)
            (loss, tcl_stats), grads = jax.value_and_grad(
                self.contrastive_loss, has_aux=True
            )(state["params"], rng_l, state, features, valid, seg_bt)
            stats.update(tcl_stats)
            stats["corac_nce_loss"] = loss
            # Contexts from the (EMA) context encoder, before this update.
            ctx_params = (
                state["target_params"] if self.target_tau > 0 else state["params"]
            )
            emb = self.segment_embeddings(
                ctx_params, features, valid, seg_bt
            )  # (B, S, N, d)
            emb = jnp.transpose(emb, (2, 1, 0, 3))  # (N, S, B, d)
            u = gather_segments(emb, seg)  # (N, T, B, d)
            parts.append(u)
            stats["corac_embed_batch_std"] = u[:, 0].std(axis=1).mean()

            updates, opt_state = self.tx.update(
                grads, state["opt_state"], state["params"]
            )
            params = optax.apply_updates(state["params"], updates)
            if self.target_tau > 0:
                target = optax.incremental_update(
                    params, state["target_params"], self.target_tau
                )
            else:
                target = params
            state = {
                **state,
                "params": params,
                "opt_state": opt_state,
                "target_params": target,
            }

        contexts = jax.lax.stop_gradient(
            parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=-1)
        )

        # One context per trajectory goes to the bank: a random episode that
        # finished inside the rollout (the first one if none did).
        is_start = jnp.concatenate(
            [jnp.ones((1, B), dtype=bool), seg[1:] != seg[:-1]], axis=0
        )  # (T, B)
        complete = seg < seg[-1:]
        cand = is_start & complete
        cand = cand.at[0].set(cand[0] | ~cand.any(0))
        g = jax.random.gumbel(rng_c, cand.shape)
        t_pick = jnp.argmax(jnp.where(cand, g, -jnp.inf), axis=0)  # (B,)
        picked = contexts[:, t_pick, jnp.arange(B)]  # (N, B, C)
        state = self._push_bank(rng_b, state, picked)

        stats["corac_bank_size"] = jnp.minimum(
            state["bank_count"], self.bank_size
        ).astype(jnp.float32)
        stats["corac_return_mean"] = state["return_mean"]
        return contexts, state, stats


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
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


def make_optimizer(config, lr):
    name = str(config.get("OPTIMIZER", "radam")).lower()
    if name == "radam":
        opt = optax.radam(learning_rate=lr)
    elif name == "adam":
        opt = optax.adam(learning_rate=lr)
    elif name == "rmsprop":
        # PyMARL / the RAC paper: RMSprop(alpha=0.99, eps=1e-5)
        opt = optax.rmsprop(learning_rate=lr, decay=0.99, eps=1e-5)
    else:
        raise ValueError(f"OPTIMIZER must be radam, adam or rmsprop, got {name!r}.")
    return optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), opt)


def make_train(config, env):

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    labeler_kind = str(config.get("CONTEXT_LABELER", "return_bins")).lower()
    if labeler_kind not in ("return_bins", "corac"):
        raise ValueError(
            f"CONTEXT_LABELER must be 'return_bins' (RAC) or 'corac' (HC-RAC), got {labeler_kind!r}."
        )
    use_corac = labeler_kind == "corac"
    teacher_mode = str(config.get("TEACHER_CONTEXT_SELECT", "action_max"))
    teacher_topk = int(config.get("TEACHER_TOPK", 4))
    twin_clamp = bool(config.get("TWIN_TARGET_CLAMP", False))
    kl_temp_frac = float(config.get("KL_LOGIT_TEMPERATURE_FRAC", 0.0) or 0.0)
    slot_number = int(config.get("SLOT_NUMBER", 10))
    return_min = float(config.get("RETURN_MIN", 0.0))
    return_max = float(config.get("RETURN_MAX", 1.0))
    if not use_corac and return_min >= return_max:
        raise ValueError(
            f"RETURN_MIN must be < RETURN_MAX, got {return_min} >= {return_max}."
        )
    twin_head = str(config.get("TWIN_HEAD", "hyper"))
    twin_hyper_hidden = int(config.get("TWIN_HYPER_HIDDEN", 0) or 0)
    if use_corac and twin_head != "hyper":
        raise ValueError(
            "HC-RAC (CONTEXT_LABELER=corac) reads continuous contexts: set TWIN_HEAD=hyper."
        )

    kl_start = float(config.get("KL_WEIGHT", 1.0))
    kl_final = config.get("KL_WEIGHT_FINAL", None)
    kl_begin = float(config.get("KL_ANNEAL_START", 0))
    kl_end = float(config.get("KL_ANNEAL_END", kl_begin))
    if kl_final is not None and kl_end < kl_begin:
        raise ValueError(
            f"KL_ANNEAL_END ({kl_end}) must be >= KL_ANNEAL_START ({kl_begin})."
        )

    def kl_weight_at(t_env):
        """KL_WEIGHT, linearly annealed to KL_WEIGHT_FINAL over [start, end] env steps."""
        if kl_final is None:
            return jnp.asarray(kl_start, jnp.float32)
        frac = jnp.clip((t_env - kl_begin) / max(kl_end - kl_begin, 1e-8), 0.0, 1.0)
        return kl_start + frac * (float(kl_final) - kl_start)

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

    def train(rng, runner_state=None, num_updates=None):

        original_seed = rng[0]
        rng, _rng = jax.random.split(rng)
        wrapped_env = CTRolloutManager(env, batch_size=config["NUM_ENVS"])
        test_env = CTRolloutManager(env, batch_size=config["TEST_NUM_ENVS"])
        n_agents = len(env.agents)
        n_actions = wrapped_env.max_action_space

        # INIT NETWORKS AND OPTIMIZER
        network = RNNQNetwork(action_dim=n_actions, hidden_dim=config["HIDDEN_SIZE"])
        labeler = None
        if use_corac:
            labeler = CoRACLabeler(
                config,
                obs_dim=wrapped_env.obs_size,
                n_actions=n_actions,
                n_agents=n_agents,
                batch_size=config["BUFFER_BATCH_SIZE"],
                n_steps=config["NUM_STEPS"],
            )
            context_dim = labeler.context_dim
        else:
            context_dim = slot_number
        twin = TwinRNNQNetwork(
            action_dim=n_actions,
            hidden_dim=config["HIDDEN_SIZE"],
            head=twin_head,
            hyper_hidden=twin_hyper_hidden,
        )

        def create_agent(rng):
            rng_a, rng_t = jax.random.split(rng)
            init_obs = jnp.zeros((1, 1, wrapped_env.obs_size))
            init_dones = jnp.zeros((1, 1))
            init_hs = ScannedRNN.initialize_carry(config["HIDDEN_SIZE"], 1)
            init_ctx = jnp.zeros((1, 1, 1, context_dim))
            params = {
                "agent": network.init(rng_a, init_hs, init_obs, init_dones),
                "twin": twin.init(rng_t, init_hs, init_obs, init_dones, init_ctx),
            }
            lr_scheduler = optax.linear_schedule(
                init_value=config["LR"],
                end_value=1e-10,
                transition_steps=(config["NUM_EPOCHS"]) * config["NUM_UPDATES"],
            )
            lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]
            return CustomTrainState.create(
                apply_fn=network.apply,
                params=params,
                target_network_params=params,
                tx=make_optimizer(config, lr),
            )

        rng, _rng = jax.random.split(rng)
        train_state = create_agent(_rng)
        rng, _rng = jax.random.split(rng)
        ctx_state = labeler.init_state(_rng) if use_corac else {}

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

        apply_agent = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))
        apply_twin = jax.vmap(twin.apply, in_axes=(None, 0, 0, 0, 0))

        # TRAINING LOOP
        def _update_step(runner_state, unused):

            train_state, ctx_state, buffer_state, test_state, rng = runner_state

            # SAMPLE PHASE (behaviour Q only, exactly as IQL)
            def _step_env(carry, _):
                hs, last_obs, last_dones, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                _obs = batchify(last_obs)[:, np.newaxis]
                _dones = batchify(last_dones)[:, np.newaxis]
                new_hs, q_vals = apply_agent(
                    train_state.params["agent"], hs, _obs, _dones
                )
                q_vals = q_vals.squeeze(axis=1)
                avail_actions = wrapped_env.get_valid_actions(env_state.env_state)
                eps = eps_scheduler(train_state.n_updates)
                _rngs = jax.random.split(rng_a, env.num_agents)
                actions = jax.vmap(eps_greedy_exploration, in_axes=(0, 0, None, 0))(
                    _rngs, q_vals, eps, batchify(avail_actions)
                )
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
                return (new_hs, new_obs, dones, new_env_state, rng), (timestep, infos)

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = wrapped_env.batch_reset(_rng)
            init_dones = {
                agent: jnp.zeros((config["NUM_ENVS"]), dtype=bool)
                for agent in env.agents + ["__all__"]
            }
            init_hs = ScannedRNN.initialize_carry(
                config["HIDDEN_SIZE"], n_agents, config["NUM_ENVS"]
            )
            rng, _rng = jax.random.split(rng)
            _, (timesteps, infos) = jax.lax.scan(
                _step_env,
                (init_hs, init_obs, init_dones, env_state, _rng),
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
                train_state, ctx_state, rng = carry
                rng, rng_s, rng_c = jax.random.split(rng, 3)
                minibatch = buffer.sample(buffer_state, rng_s).experience
                minibatch = jax.tree.map(
                    lambda x: jnp.swapaxes(x[:, 0], 0, 1), minibatch
                )
                # (num_agents, timesteps, batch_size, ...)
                B = config["BUFFER_BATCH_SIZE"]
                init_hs = ScannedRNN.initialize_carry(
                    config["HIDDEN_SIZE"], n_agents, B
                )
                _obs = batchify(minibatch.obs)
                _dones = batchify(minibatch.dones)
                _actions = batchify(minibatch.actions)
                _rewards = batchify(minibatch.rewards)
                _avail = batchify(minibatch.avail_actions)
                ep_dones = minibatch.dones["__all__"]  # (T, B)
                T = ep_dones.shape[0]

                # ---- contexts of every step and the teacher's support
                stats = {}
                if use_corac:
                    contexts, ctx_state, stats = labeler.label(
                        rng_c, ctx_state, _obs, _actions, _rewards, ep_dones
                    )
                    bank = ctx_state["bank"][:, None, None]  # (N, 1, 1, M, C)
                    bank_valid = labeler.bank_valid(ctx_state)[
                        :, None, None
                    ]  # (N,1,1,M)
                else:
                    seg = episode_segments(ep_dones)
                    returns = gather_segments(
                        segment_returns(_rewards, seg), seg
                    )  # (N,T,B)
                    contexts, slots = return_bin_contexts(
                        returns, return_min, return_max, slot_number
                    )
                    eye = jnp.eye(slot_number)
                    bank = jnp.broadcast_to(
                        eye, (n_agents, 1, 1, slot_number, slot_number)
                    )
                    bank_valid = jnp.ones((n_agents, 1, 1, slot_number), dtype=bool)
                    slot_frac = jax.nn.one_hot(slots[:, 0], slot_number).mean(
                        axis=(0, 1)
                    )
                    for i in range(slot_number):
                        stats[f"return_slot_{i}_fraction"] = slot_frac[i]
                contexts = contexts[..., None, :]  # (N, T, B, 1, C)

                ret_low, ret_high = discounted_return_bounds(
                    _rewards, ep_dones, config["GAMMA"]
                )
                if kl_temp_frac > 0:
                    kl_temperature = jnp.maximum(
                        1e-3,
                        kl_temp_frac * jnp.maximum(jnp.abs(ret_high - ret_low), 1e-3),
                    )
                else:
                    kl_temperature = 1.0

                # ---- optimistic teacher: twin Q maximised over contexts (no grad)
                _, all_ctx_q = apply_twin(
                    train_state.params["twin"], init_hs, _obs, _dones, bank
                )  # (N, T, B, M, A)
                teacher_q = select_teacher_q(
                    all_ctx_q, _avail, bank_valid, teacher_mode, teacher_topk
                )[:, :-1]
                if twin_clamp:
                    teacher_q = jnp.clip(teacher_q, ret_low, ret_high)
                teacher_q = jax.lax.stop_gradient(teacher_q)

                # ---- targets
                tp = train_state.target_network_params
                _, q_next_target = apply_agent(tp["agent"], init_hs, _obs, _dones)
                _, twin_next_target = apply_twin(
                    tp["twin"], init_hs, _obs, _dones, contexts
                )
                twin_next_target = twin_next_target[..., 0, :]
                kl_w = kl_weight_at(train_state.timesteps)

                def _loss_fn(params):
                    _, q_vals = apply_agent(params["agent"], init_hs, _obs, _dones)
                    _, twin_q = apply_twin(
                        params["twin"], init_hs, _obs, _dones, contexts
                    )
                    twin_q = twin_q[..., 0, :]

                    def td(q, q_target):
                        chosen = jnp.take_along_axis(
                            q, _actions[..., None], axis=-1
                        ).squeeze(-1)
                        valid_q = q - (1 - _avail) * 1e10
                        q_next = jnp.take_along_axis(
                            q_target, jnp.argmax(valid_q, axis=-1)[..., None], axis=-1
                        ).squeeze(-1)
                        # dones[t] marks obs t as the first obs of a new
                        # episode, so transition t is terminal iff dones[t+1].
                        target = (
                            _rewards[:, :-1]
                            + (1 - _dones[:, 1:]) * config["GAMMA"] * q_next[:, 1:]
                        )
                        return chosen[:, :-1], target

                    chosen, target = td(q_vals, q_next_target)
                    twin_chosen, twin_target = td(twin_q, twin_next_target)
                    if twin_clamp:
                        # Bound the twin's bootstrap by the realisable discounted
                        # return so it cannot ratchet on reward-poor contexts.
                        twin_target = jnp.clip(twin_target, ret_low, ret_high)
                    target = jax.lax.stop_gradient(target)
                    twin_target = jax.lax.stop_gradient(twin_target)
                    td_loss = jnp.mean((chosen - target) ** 2)
                    twin_td_loss = jnp.mean((twin_chosen - twin_target) ** 2)

                    kl = categorical_kl(
                        q_vals[:, :-1] / kl_temperature, teacher_q / kl_temperature
                    )
                    kl_loss = kl.mean()
                    loss = td_loss + twin_td_loss + kl_w * kl_loss
                    aux = {
                        "td_loss": td_loss,
                        "twin_td_loss": twin_td_loss,
                        "kl_loss": kl_loss,
                        "qvals": chosen.mean(),
                        "twin_qvals": twin_chosen.mean(),
                        "twin_target_mean": twin_target.mean(),
                    }
                    return loss, aux

                (loss, aux), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
                    train_state.params
                )
                train_state = train_state.apply_gradients(grads=grads)
                train_state = train_state.replace(grad_steps=train_state.grad_steps + 1)
                aux.update(stats)
                aux["loss"] = loss
                aux["kl_weight"] = kl_w
                aux["teacher_q_mean"] = teacher_q.mean()
                aux["return_rtg_low"] = ret_low
                aux["return_rtg_high"] = ret_high
                return (train_state, ctx_state, rng), aux

            def _no_learn(train_state, ctx_state, rng):
                # Same pytree as _learn_phase's outputs, filled with zeros.
                aux_shape = jax.eval_shape(
                    lambda ts, cs, r: jax.lax.scan(
                        _learn_phase, (ts, cs, r), None, config["NUM_EPOCHS"]
                    )[1],
                    train_state,
                    ctx_state,
                    rng,
                )
                aux = jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), aux_shape)
                return (train_state, ctx_state, rng), aux

            rng, _rng = jax.random.split(rng)
            is_learn_time = buffer.can_sample(buffer_state) & (
                train_state.timesteps > config["LEARNING_STARTS"]
            )
            (train_state, ctx_state, rng), learn_metrics = jax.lax.cond(
                is_learn_time,
                lambda ts, cs, r: jax.lax.scan(
                    _learn_phase, (ts, cs, r), None, config["NUM_EPOCHS"]
                ),
                _no_learn,
                train_state,
                ctx_state,
                _rng,
            )

            # update target network
            train_state = jax.lax.cond(
                train_state.n_updates % config["TARGET_UPDATE_INTERVAL"] == 0,
                lambda train_state: train_state.replace(
                    target_network_params=optax.incremental_update(
                        train_state.params,
                        train_state.target_network_params,
                        config["TAU"],
                    )
                ),
                lambda train_state: train_state,
                operand=train_state,
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

            runner_state = (train_state, ctx_state, buffer_state, test_state, rng)
            return runner_state, metrics

        def get_greedy_metrics(rng, train_state):
            """Greedy evaluation of the behaviour Q (the only network used to act)."""
            if not config.get("TEST_DURING_TRAINING", True):
                return None
            params = train_state.params["agent"]

            def _greedy_env_step(step_state, unused):
                params, env_state, last_obs, last_dones, hstate, rng = step_state
                rng, key_s = jax.random.split(rng)
                _obs = batchify(last_obs)[:, np.newaxis]
                _dones = batchify(last_dones)[:, np.newaxis]
                hstate, q_vals = apply_agent(params, hstate, _obs, _dones)
                q_vals = q_vals.squeeze(axis=1)
                valid_actions = test_env.get_valid_actions(env_state.env_state)
                actions = get_greedy_actions(q_vals, batchify(valid_actions))
                actions = unbatchify(actions)
                obs, env_state, rewards, dones, infos = test_env.batch_step(
                    key_s, env_state, actions
                )
                infos = schedules.drop_shaped_reward(infos)
                step_state = (params, env_state, obs, dones, hstate, rng)
                return step_state, (rewards, dones, infos)

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = test_env.batch_reset(_rng)
            init_dones = {
                agent: jnp.zeros((config["TEST_NUM_ENVS"]), dtype=bool)
                for agent in env.agents + ["__all__"]
            }
            rng, _rng = jax.random.split(rng)
            hstate = ScannedRNN.initialize_carry(
                config["HIDDEN_SIZE"], n_agents, config["TEST_NUM_ENVS"]
            )
            step_state = (params, env_state, init_obs, init_dones, hstate, _rng)
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
        init_state = (train_state, ctx_state, buffer_state, test_state, _rng)
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

    alg_name = config.get("ALG_NAME", "rac_rnn")
    env, env_name = env_from_config(copy.deepcopy(config))

    ckpt_dir = checkpointing.checkpoint_dir(config, alg_name, env_name)
    checkpointing.init_wandb(
        ckpt_dir,
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[alg_name.upper(), env_name.upper(), f"jax_{jax.__version__}"],
        name=f"{alg_name}_{env_name}",
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
            # Only the behaviour Q is needed to act.
            params = jax.tree.map(lambda x: x[i], model_state.params["agent"])
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
