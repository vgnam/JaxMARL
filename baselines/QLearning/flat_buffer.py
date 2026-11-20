"""Replay buffer that stores each trajectory item as one flat row.

The QLearning baselines keep ``NUM_STEPS``-long trajectories in a flashbax
trajectory buffer with leaves ``(NUM_ENVS, slots, NUM_STEPS, *feature)``. On GPU,
XLA gives that 4-D buffer a different layout inside the training loop than at
its boundaries (adds and samples prefer different axes), so every update
rewrites the whole buffer and the device needs two copies of it: ~16 GiB on
SMAX 10m_vs_11m, which does not fit a 16 GB T4.

Flattening each item to ``(NUM_STEPS * prod(feature),)`` makes both the add (one
row per env) and the sample (gather of whole rows) contiguous, so the buffer is
updated in place. The stored values, the sampling and the returned minibatch
are unchanged: ``sample`` reshapes the rows back to ``(B, 1, NUM_STEPS, *feature)``.
"""

import jax


class FlatTrajectoryBuffer:
    def __init__(self, buffer, item):
        """``buffer``: a flashbax trajectory buffer; ``item``: one trajectory
        (pytree of ``(NUM_STEPS, *feature)`` arrays) as passed to ``init``."""
        self._buffer = buffer
        leaves, self._treedef = jax.tree_util.tree_flatten(item)
        self._shapes = [x.shape for x in leaves]

    def _flatten(self, tree, lead):
        leaves = self._treedef.flatten_up_to(tree)
        return jax.tree_util.tree_unflatten(
            self._treedef, [x.reshape(x.shape[:lead] + (-1,)) for x in leaves]
        )

    def init(self, item):
        return self._buffer.init(self._flatten(item, 0))

    def add(self, state, batch):
        # batch leaves: (add_batch_size, 1, NUM_STEPS, *feature)
        return self._buffer.add(state, self._flatten(batch, 2))

    def can_sample(self, state):
        return self._buffer.can_sample(state)

    def sample(self, state, rng):
        sample = self._buffer.sample(state, rng)
        leaves = self._treedef.flatten_up_to(sample.experience)
        experience = jax.tree_util.tree_unflatten(
            self._treedef,
            [x.reshape(x.shape[:2] + shape) for x, shape in zip(leaves, self._shapes)],
        )
        return sample.replace(experience=experience)
