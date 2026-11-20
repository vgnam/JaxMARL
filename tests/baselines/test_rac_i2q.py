import os
import subprocess
import sys

import jax.numpy as jnp
import pytest

from baselines.QLearning.rac_rnn import (
    discounted_return_bounds,
    episode_segments,
    gather_segments,
    return_bin_contexts,
    segment_returns,
    select_teacher_q,
)


def run_script(script_path, *args):
    return subprocess.run(
        [sys.executable, script_path, *args], capture_output=True, text=True
    )


@pytest.mark.parametrize(
    "script, alg",
    [
        ("rac_rnn.py", "rac_rnn_mpe"),
        ("rac_rnn.py", "hcrac_rnn_mpe"),
        ("i2q_rnn.py", "i2q_rnn_mpe"),
    ],
)
def test_script_with_arguments(script, alg):
    result = run_script(
        os.path.join("baselines/QLearning", script),
        f"+alg={alg}",
        "alg.TOTAL_TIMESTEPS=2e4",
        "alg.LEARNING_STARTS=2000",  # make sure the learning phase runs
        "alg.TEST_NUM_ENVS=32",
        "WANDB_MODE=disabled",
        "SAVE_PATH=null",
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_episode_returns_within_a_rollout():
    # obs 3 is the first obs of a new episode: episodes are steps [0, 2] and [3, 5]
    dones = jnp.array([[0], [0], [0], [1], [0], [0]], dtype=bool)
    rewards = jnp.array([[1.0], [2.0], [3.0], [10.0], [20.0], [30.0]])[None]
    seg = episode_segments(dones)
    returns = gather_segments(segment_returns(rewards, seg), seg)
    assert returns[0, :, 0].tolist() == [6.0, 6.0, 6.0, 60.0, 60.0, 60.0]

    low, high = discounted_return_bounds(rewards, dones, 0.5)
    assert float(low) == 0.0
    assert float(high) == pytest.approx(20.0 + 0.5 * 30.0)


def test_return_bins():
    _, idx = return_bin_contexts(
        jnp.array([-1.0, 0.0, 0.49, 0.5, 1.0, 5.0]), 0.0, 1.0, 2
    )
    assert idx.tolist() == [0, 0, 0, 1, 1, 1]


def test_teacher_selection():
    q = jnp.array([[1.0, 5.0], [4.0, 0.0], [3.0, 3.0]])  # (contexts, actions)
    avail = jnp.array([1.0, 1.0])
    valid = jnp.array([True, True, False])  # last bank entry not filled yet
    assert select_teacher_q(q, avail, valid, "action_max", 2).tolist() == [4.0, 5.0]
    assert select_teacher_q(q, avail, valid, "context_argmax", 2).tolist() == [1.0, 5.0]
    only_a0 = jnp.array([1.0, 0.0])
    assert select_teacher_q(q, only_a0, valid, "context_argmax", 2).tolist() == [
        4.0,
        0.0,
    ]
    assert select_teacher_q(q, avail, valid, "topk_mean", 2).tolist() == [2.5, 2.5]
