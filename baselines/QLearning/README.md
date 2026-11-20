# QLearning Baselines

Pure JAX implementations of:
* PQN-VDN (Prallelised Q-Network)
* IQL (Independent Q-Learners)
* VDN (Value Decomposition Network)
* QMIX
* TransfQMix (Transformers for Leveraging the Graph Structure of MARL Problems)
* SHAQ (Incorporating Shapley Value Theory into Multi-Agent Q-Learning)
* RAC (In-Context Fully Decentralized Cooperative MARL) and its HC-RAC variant
* I2Q (A Fully Decentralized Q-Learning Algorithm)

PQN implementation follows [purejaxql](https://github.com/mttga/purejaxql). IQL, VDN and QMix follow the original [Pymarl](https://github.com/oxwhirl/pymarl/blob/master/src/learners/q_learner.py) codebase while SHAQ follows the [paper code](https://github.com/hsvgbkhgbv/shapley-q-learning). 


Standard algorithms (iql, vdn, qmix) support:
- MPE
- SMAX
- Overcooked (qmix not supported)

PQN-VDN supports:
- MPE
- SMAX
- Hanabi
- Overcooked

**At the moment, PQN-VDN should be the most performant baseline for Q-Learning in terms of returns and training speed.**

❗ TransfQMix and Shaq still use an old implementation of the scripts and need refactoring to match the new format. 

### Fully decentralized: RAC, HC-RAC and I2Q

These learners only use each agent's own local history, reward and actions, with
no mixer and no global state. They share IQL's rollout, replay buffer, recurrent
Q-network and greedy evaluation, and support MPE, SMAX, JaxRobotarium and
Overcooked (`*_rnn_mpe` / `*_rnn_smax` / `*_rnn_jaxrobotarium` /
`*_rnn_overcooked` configs; `ql_rnn_overcooked` is the matching RNN IQL).

- **RAC** (`rac_rnn.py`, `+alg=rac_rnn_*`): besides the behaviour Q, each agent
  trains a twin Q conditioned on a one-hot bin of the episode return
  (`SLOT_NUMBER` bins over [`RETURN_MIN`, `RETURN_MAX`], in `REW_SCALE` units).
  The behaviour Q is pulled towards the twin maximised over contexts through
  `KL_WEIGHT * KL(softmax(Q) || softmax(Q_teacher))`. `TWIN_HEAD` picks RAC's
  Hyper-Net (`hyper`) or Normal-Net (`normal`, the paper's choice on SMAC).
- **HC-RAC** (`rac_rnn.py`, `+alg=hcrac_rnn_*`): the same learner with
  `CONTEXT_LABELER: corac`. The context is continuous, `[phi(R) ; u]`: cosine
  features of the return plus a trajectory embedding learned with trajectory
  contrastive learning. The twin reads it through a zero-delta hypernetwork, and
  the teacher maximises over a bank of recent contexts. `hcrac_rnn_smax` is the
  main HC-RAC setting (as PyMARL `corac_mate`): fixed return range and EMA
  context encoder, a constant `KL_WEIGHT` and RAC's per-action teacher
  (`action_max`); so are the `mpe`, `jaxrobotarium` and `overcooked` presets.
  Step C of the stability ladder (`TEACHER_CONTEXT_SELECT: context_argmax`)
  keeps one context per step instead.
- **I2Q** (`i2q_rnn.py`, `+alg=i2q_rnn_*`): each agent fits `Q_ss(h, h')` and
  an expectile value `V(h)` (`I2Q_TAU`) on its own target-network features,
  and blends `r + gamma * V(h')` into its TD target with weight `I2Q_LAMBDA`.

The RAC return range must match the environment's return scale. The defaults
are [-60, 0] per agent for MPE simple_spread, [0, 20] for SMAX (a return in
[0, 2] times `REW_SCALE` 10), [-240, 0] for Robotarium navigation (minus the
distance to the goal each step) and [0, 300] for Overcooked cramped_room (20 per
soup plus the annealed shaped reward). Set `RETURN_MIN`/`RETURN_MAX` (RAC) or
`CORAC_RETURN_RANGE` (HC-RAC) when you change the environment or `REW_SCALE`.
The SMAX presets use the default map `3s_vs_5z`, where the paper's `KL_WEIGHT`
is 0.001; use 0.01 on other maps.

```bash
python baselines/QLearning/rac_rnn.py +alg=rac_rnn_mpe
python baselines/QLearning/rac_rnn.py +alg=hcrac_rnn_smax alg.MAP_NAME=2s3z alg.KL_WEIGHT=0.01
python baselines/QLearning/i2q_rnn.py +alg=i2q_rnn_smax
python baselines/QLearning/rac_rnn.py +alg=hcrac_rnn_jaxrobotarium
python baselines/QLearning/i2q_rnn.py +alg=i2q_rnn_overcooked alg.ENV_KWARGS.layout=coord_ring
```

Overcooked: the RNN learners take the grid observation flattened by
`CTRolloutManager` and add `info["shaped_reward"]`, annealed from 1 to 0 over
`REW_SHAPING_HORIZON` env steps, to the training reward (as
`iql_cnn_overcooked.py`); the logged returns stay the sparse soup reward. One
update rolls out a whole 400-step episode per env (`NUM_STEPS: 401`), so the
buffer holds 1000 episodes.

JaxRobotarium is a git submodule of JaxMARL with its own simulator submodule:

```bash
git submodule update --init --recursive jaxmarl/environments/robotarium
pip install -e jaxmarl/environments/robotarium     -e jaxmarl/environments/robotarium/jaxrobotarium/robotarium_python_simulator jaxopt
```

The presets use the navigation scenario (as `ql_rnn_jaxrobotarium`); the other
scenarios need their own `ENV_KWARGS` and return range.

#### Checkpoint and resume

`iql_rnn.py`, `rac_rnn.py` and `i2q_rnn.py` can resume after a crash or reboot.
Set `CHECKPOINT_DIR` and training runs in chunks of `CHECKPOINT_INTERVAL`
(default 0.05) of the updates. After each chunk the full runner state (params,
optimizer, target network, replay buffer, test state, rng) is saved to
`CHECKPOINT_DIR/<alg>_<env>_seed<SEED>/`. Rerunning the same command loads it,
continues the same W&B run without logging finished updates twice, and gives
the same parameters as an uninterrupted chunked run. The config is stored in
`meta.json` and a resume with different hyperparameters stops with the list of
changed keys (output and logging settings such as `SAVE_PATH`, `WANDB_MODE` and
`CHECKPOINT_INTERVAL` may change). A finished run is marked in `meta.json`;
delete its folder to train it again.

```bash
python baselines/QLearning/rac_rnn.py +alg=rac_rnn_smax CHECKPOINT_DIR=checkpoints
```

`scripts/run_smax_resumable.sh <MAP> [iql rac hcrac i2q]` runs the methods one
after another in a single background queue under `flock`, retrying each from its
checkpoint on a crash and skipping finished ones. Run it again, or from
`crontab @reboot`, after a reboot.


## ⚙️ Implementation Details

All the algorithms take advantage of the `CTRolloutManager` environment wrapper (found in `jaxmarl.wrappers.baselines`), which is used to:

- Batchify the step and reset functions to run parallel environments.
- Add a global observation (`obs["__all__"]`) and a global reward (`rewards["__all__"]`) to the returns of `env.step` for centralized training.
- Preprocess and uniform the observation vectors (flatten, pad, add additional features like id one-hot encoding, etc.).

You might want to modify this wrapper for your needs.

## 🚀 Usage

If you have cloned JaxMARL and you are in the repository root, you can run the algorithms as scripts. You will need to specify which parameter configurations will be loaded by Hydra by choosing them (or adding yours) in the config folder. Below are some examples:

```bash
# vdn rnn in in mpe spread
python baselines/QLearning/vdn_rnn.py +alg=ql_rnn_mpe
# independent IQL rnn in competetive simple_tag (predator-prey)
python baselines/QLearning/iql_rnn.py +alg=ql_rnn_mpe alg.ENV_NAME=MPE_simple_tag_v3
# QMix with SMAX
python baselines/QLearning/qmix_rnn.py +alg=ql_rnn_smax
# VDN overcooked
python baselines/QLearning/vdn_cnn_overcooked.py +alg=ql_cnn_overcooked alg.ENV_KWARGS.LAYOUT=counter_circuit
# TransfQMix
python baselines/QLearning/transf_qmix.py +alg=transf_qmix_smax

# pqn feed-forward in MPE
python baselines/QLearning/pqn_vdn_ff.py +alg=pqn_vdn_ff_mpe
# pqn feed-forward in hanabi
python baselines/QLearning/pqn_vdn_ff.py +alg=pqn_vdn_ff_hanabi
# pqn CNN in overcooked
python baselines/QLearning/pqn_vdn_cnn_overcooked.py +alg=pqn_vdn_cnn_overcooked
# pqn with RNN in SMAX
python baselines/QLearning/pqn_vdn_rnn.py +alg=pqn_vdn_rnn_smax
```

Notice that with Hydra, you can modify parameters on the go in this way:

```bash
# change learning rate
python baselines/QLearning/iql_rnn.py +alg=ql_rnn_mpe alg.LR=0.001
# change overcooked layout
python baselines/QLearning/pqn_vdn_cnn_overcooked.py +alg=pqn_vdn_cnn_overcooked alg.ENV_KWARGS.LAYOUT=counter_circuit
# change smax map
python baselines/QLearning/pqn_vdn_rnn.py +alg=pqn_vdn_rnn_smax alg.MAP_NAME=5m_vs_6m
```

Take a look at [`config.yaml`](./config/config.yaml) for the default configuration when running these scripts. There you can choose how many seeds to vmap and you can setup WANDB. 

**❗Note on Transformers**: TransfQMix currently supports only MPE_Spread and SMAX. You will need to wrap the observation vectors into matrices to use transformers in other environments. See: ```jaxmarl.wrappers.transformers```

## 🎯 Hyperparameter tuning

All the scripts include a tune function to perform hyperparameter tuning. To use it, set `"HYP_TUNE": True` in the `config.yaml` and set the hyperparameters spaces in the tune function. For more information, check [wandb documentation](https://docs.wandb.ai/guides/sweeps).