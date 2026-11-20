"""Chunked training with checkpoint/resume for the QLearning baselines.

The baselines run the whole training loop as one jitted ``lax.scan``, so a
crash or reboot loses the run. With ``CHECKPOINT_DIR`` set, training runs in
chunks of ``CHECKPOINT_INTERVAL * NUM_UPDATES`` updates and the full runner
state (params, optimizer, target params, replay buffer, test state, rng) is
written to disk after each chunk. Restarting the same command resumes from
the last checkpoint and continues the same W&B run, so the result matches an
uninterrupted run.

``make_train(config, env)`` must return ``train(rng, runner_state=None,
num_updates=None)``: it builds the initial state when ``runner_state`` is None
and runs ``num_updates`` updates (default ``config["NUM_UPDATES"]``).
"""

import json
import os

import jax
import numpy as np
import wandb

STATE_FILE = "state.npz"
META_FILE = "meta.json"

# Updates up to this index were already logged before a crash; their metrics
# are recomputed identically on resume, so they are not logged twice.
_SKIP_LOG_UNTIL = -1


def checkpoint_dir(config, alg_name, env_name):
    root = config.get("CHECKPOINT_DIR")
    if not root:
        return None
    return os.path.join(root, f"{alg_name}_{env_name}_seed{config['SEED']}")


def _load_meta(ckpt_dir):
    path = os.path.join(ckpt_dir, META_FILE)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _write_atomic(path, write):
    tmp = path + ".tmp"
    write(tmp)
    os.replace(tmp, path)


def _save_meta(ckpt_dir, meta):
    def write(tmp):
        with open(tmp, "w") as f:
            json.dump(meta, f, indent=2)

    _write_atomic(os.path.join(ckpt_dir, META_FILE), write)


# Settings that may change between a run and its resume.
_RESUME_FREE_KEYS = {
    "CHECKPOINT_DIR",
    "CHECKPOINT_INTERVAL",
    "SAVE_PATH",
    "WANDB_MODE",
    "ENTITY",
    "PROJECT",
    "WANDB_LOG_ALL_SEEDS",
    # Evaluation frequency only: the test rng is split every update whether or
    # not the test runs, so training itself is unchanged.
    "TEST_INTERVAL",
}


def _config_fingerprint(config):
    def strip(d):
        return {
            k: strip(v) if isinstance(v, dict) else v
            for k, v in d.items()
            if k not in _RESUME_FREE_KEYS
        }

    return json.loads(json.dumps(strip(config), sort_keys=True, default=str))


def _diff_keys(a, b, prefix=""):
    keys = []
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k), b.get(k)
        if isinstance(va, dict) and isinstance(vb, dict):
            keys += _diff_keys(va, vb, f"{prefix}{k}.")
        elif va != vb:
            keys.append(f"{prefix}{k}: {va!r} -> {vb!r}")
    return keys


def init_wandb(ckpt_dir, **wandb_kwargs):
    """``wandb.init`` that reattaches to the run recorded in the checkpoint."""
    global _SKIP_LOG_UNTIL
    if ckpt_dir is None:
        return wandb.init(**wandb_kwargs)
    os.makedirs(ckpt_dir, exist_ok=True)
    meta = _load_meta(ckpt_dir)
    if meta.get("wandb_id"):
        wandb_kwargs.update(id=meta["wandb_id"], resume="allow")
    run = wandb.init(**wandb_kwargs)
    if wandb_kwargs.get("mode") != "disabled" and getattr(run, "id", None):
        last = run.summary.get("update_steps")
        if last is not None:
            _SKIP_LOG_UNTIL = int(last)
        meta["wandb_id"] = run.id
        _save_meta(ckpt_dir, meta)
    return run


def should_log(metrics):
    """False for updates already logged before the run was resumed."""
    if _SKIP_LOG_UNTIL < 0:
        return True
    return int(np.max(np.asarray(metrics["update_steps"]))) > _SKIP_LOG_UNTIL


def _save_state(ckpt_dir, state, updates_done):
    # np.savez converts one array at a time, so host RAM holds one leaf (not the
    # whole replay buffer) at once.
    leaves = jax.tree_util.tree_leaves(state)
    arrays = {f"leaf{i}": x for i, x in enumerate(leaves)}
    arrays["updates_done"] = np.asarray(updates_done)

    def write(tmp):
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)

    _write_atomic(os.path.join(ckpt_dir, STATE_FILE), write)


def _load_state(ckpt_dir, template):
    path = os.path.join(ckpt_dir, STATE_FILE)
    if not os.path.exists(path):
        return None, 0
    leaves, treedef = jax.tree_util.tree_flatten(template)
    with np.load(path) as data:
        n_saved = len([k for k in data.files if k.startswith("leaf")])
        if n_saved != len(leaves):
            raise ValueError(
                f"{path} has {n_saved} arrays but the runner state has "
                f"{len(leaves)}: the config or code changed since it was saved."
            )
        loaded = []
        for i, ref in enumerate(leaves):
            arr = data[f"leaf{i}"]
            if arr.shape != ref.shape or arr.dtype != ref.dtype:
                raise ValueError(
                    f"{path} leaf {i}: saved {arr.shape} {arr.dtype}, expected "
                    f"{ref.shape} {ref.dtype}: the config changed since it was saved."
                )
            loaded.append(jax.device_put(arr))
            del arr
        updates_done = int(data["updates_done"])
    return jax.tree_util.tree_unflatten(treedef, loaded), updates_done


def train(config, train_fn, rngs, ckpt_dir):
    """Run ``train_fn`` vmapped over ``rngs``; returns the final runner state."""
    if ckpt_dir is None:
        outs = jax.jit(jax.vmap(train_fn))(rngs)
        return jax.block_until_ready(outs)["runner_state"]

    total = int(config["NUM_UPDATES"])
    chunk = max(1, int(round(total * config.get("CHECKPOINT_INTERVAL", 0.05))))

    init = jax.jit(jax.vmap(lambda r: train_fn(r, num_updates=0)["runner_state"]))
    has_checkpoint = os.path.exists(os.path.join(ckpt_dir, STATE_FILE))

    meta = _load_meta(ckpt_dir)
    fingerprint = _config_fingerprint(config)
    if has_checkpoint:
        saved = meta.get("config")
        if saved is not None:
            # Strip again: older checkpoints were saved with fewer free keys.
            saved = _config_fingerprint(saved)
        if saved is not None and saved != fingerprint:
            raise ValueError(
                f"{ckpt_dir} was saved with a different config; delete it to start "
                "over or restore these settings:\n  "
                + "\n  ".join(_diff_keys(saved, fingerprint))
            )
    else:
        meta["config"] = fingerprint
        _save_meta(ckpt_dir, meta)

    if has_checkpoint:
        # Shapes only: materialising the initial state as well would hold two
        # replay buffers on the device while loading.
        state, done = _load_state(ckpt_dir, jax.eval_shape(init, rngs))
        print(f"[checkpoint] resumed {ckpt_dir} at update {done}/{total}")
    else:
        state, done = jax.block_until_ready(init(rngs)), 0
        print(f"[checkpoint] new run, saving to {ckpt_dir} every {chunk} updates")

    runners = {}
    while done < total:
        n = min(chunk, total - done)
        if n not in runners:
            runners[n] = jax.jit(
                jax.vmap(
                    lambda r, s, n=n: train_fn(r, s, num_updates=n)["runner_state"]
                ),
                donate_argnums=1,
            )
        state = jax.block_until_ready(runners[n](rngs, state))
        done += n
        _save_state(ckpt_dir, state, done)
        print(f"[checkpoint] saved update {done}/{total}")

    meta = _load_meta(ckpt_dir)
    meta["finished"] = True
    _save_meta(ckpt_dir, meta)
    return state
