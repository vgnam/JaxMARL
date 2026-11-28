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
no mixer and no global state. Every preset copies the hyper-parameters of the
IQL baseline JaxMARL ships for that environment and only adds the method's own
keys, so each method differs from IQL by its algorithm alone:

| Env | IQL baseline (unchanged) | RAC / HC-RAC / I2Q |
|---|---|---|
| MPE | `iql_rnn.py +alg=ql_rnn_mpe` | `rac_rnn.py` / `i2q_rnn.py`, `*_rnn_mpe` |
| SMAX | `iql_rnn.py +alg=ql_rnn_smax` | `rac_rnn.py` / `i2q_rnn.py`, `*_rnn_smax` |
| JaxRobotarium | `iql_rnn.py +alg=ql_rnn_jaxrobotarium` | `rac_rnn.py` / `i2q_rnn.py`, `*_rnn_jaxrobotarium` |
| Overcooked | `iql_cnn_overcooked.py +alg=ql_cnn_overcooked` | `rac_cnn_overcooked.py` / `i2q_cnn_overcooked.py`, `*_cnn_overcooked` |
| LBF | `iql_rnn.py +alg=ql_rnn_lbf` (no upstream preset, see below) | `rac_rnn.py` / `i2q_rnn.py`, `*_rnn_lbf` |
| Matrix games | `iql_rnn.py +alg=ql_rnn_matrix` (no upstream preset, see below) | `rac_rnn.py` / `i2q_rnn.py`, `*_rnn_matrix` |
| Predator-prey (StagHunt) | `iql_rnn.py +alg=ql_rnn_staghunt` (no upstream preset, see below) | `rac_rnn.py` / `i2q_rnn.py`, `*_rnn_staghunt` |

The RNN learners share `iql_rnn.py`'s rollout, replay buffer, recurrent
Q-network and greedy evaluation; the Overcooked ones share
`iql_cnn_overcooked.py`'s CNN, 1-step rollouts and flat transition buffer.

Two changes to the RNN learners (`iql_rnn.py`, `rac_rnn.py`, `i2q_rnn.py`)
make them match PyMARL / EPyMARL, and are on by default:

- `OBS_LAST_ACTION: True`: every agent also reads the one-hot of its previous
  action (zeros at the start of an episode), as PyMARL's `obs_last_action`.
  Without it the agent cannot tell what it just did when the observation does
  not change (a failed `LOAD`, waiting), and IQL, I2Q and RAC stay near 0 on
  LBF coop tasks (IQL: 0.005 without, 0.30-0.49 with, at 2e6 steps).
- `CORRECT_DONE_INDEX: True` (`iql_rnn.py` only): upstream stores
  `dones[t]` = "obs t starts a new episode" (commit 7b17099) but its TD target
  still reads `dones[t]` for transition t, so the last step of an episode
  bootstraps into the next episode and the first step of a new one is treated
  as terminal. `rac_rnn.py` / `i2q_rnn.py` always used `dones[t + 1]`.
  (`vdn_rnn.py` / `qmix_rnn.py` keep upstream's indexing.)

Set both to `False` to reproduce upstream JaxMARL. `iql_rnn.py` also has
`TIMELIMIT_MODE: mask` (+ `TIME_LIMIT`), which drops time-limit transitions from
the loss; it made no difference on LBF coop.

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
  (`action_max`); so are the `mpe`, `jaxrobotarium` and `cnn_overcooked` presets.
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
python baselines/QLearning/rac_cnn_overcooked.py +alg=hcrac_cnn_overcooked
python baselines/QLearning/i2q_cnn_overcooked.py +alg=i2q_cnn_overcooked alg.ENV_KWARGS.layout=coord_ring
```

Overcooked: RAC and HC-RAC need the return of the whole episode, so in
`rac_cnn_overcooked.py` transitions wait in a per-env staging area until their
400-step episode ends, are labelled, and only then enter the flat buffer;
learning starts after the first episodes (400 * `NUM_ENVS` env steps). I2Q needs
no labels and uses IQL-CNN's buffer as is. All Overcooked learners add the
annealed `info["shaped_reward"]` to the training reward as `iql_cnn_overcooked.py`
does; the logged returns stay the sparse soup reward.

`EPS_DECAY` is a fraction of the updates, as in JaxMARL. `EPS_DECAY_STEPS`
(env steps, `++alg.EPS_DECAY_STEPS=...`) fixes the anneal length instead in the
RNN learners.

JaxRobotarium is a git submodule of JaxMARL with its own simulator submodule:

```bash
git submodule update --init --recursive jaxmarl/environments/robotarium
pip install -e jaxmarl/environments/robotarium     -e jaxmarl/environments/robotarium/jaxrobotarium/robotarium_python_simulator jaxopt
```

The presets use the navigation scenario (as `ql_rnn_jaxrobotarium`); the other
scenarios need their own `ENV_KWARGS` and return range.

LBF (`ENV_NAME: LBF`, `jaxmarl/environments/lbf`) is a JAX port of lbforaging v3,
the version behind EPyMARL's `Foraging-*-v3` tasks; it matches lbforaging step by
step (observations, movement, loading, rewards, termination). JaxMARL ships no
LBF baseline, so `ql_rnn_lbf` is `ql_rnn_mpe` with the episode-dependent keys set
for 50-step episodes and PyMARL/EPyMARL's IQL settings (2e6 steps, gamma 0.99,
lr 5e-4, grad clip 10, epsilon over 50k steps, about one replay batch per
collected episode, target update every 200 batches). IQL then reaches ~0.9
greedy return on `Foraging-8x8-2p-2f-v3` (`force_coop: False`) in 2e6 steps. On
the `-coop` task IQL without `OBS_LAST_ACTION` stays near 0, while PyMARL's IQL
reaches ~0.89 (seed 0) and EPyMARL reports 1.00; with the last action it reaches
0.30-0.49 at 2e6 steps (seed 0), still below PyMARL (RMSprop, constant lr and
one environment per batch did not close the rest of the gap). The
default task is `Foraging-8x8-2p-2f-coop-v3`; `ENV_KWARGS` mirror the task name
(`field_size`, `num_agents`, `max_num_food`, `force_coop`, `sight: 2` for `-2s`).
With `shared_rewards` every agent gets the normalised team reward, so the
episode return and the RAC / HC-RAC return range are [0, 1] on every task.

```bash
python baselines/QLearning/iql_rnn.py +alg=ql_rnn_lbf
python baselines/QLearning/rac_rnn.py +alg=hcrac_rnn_lbf alg.ENV_KWARGS.sight=2
python baselines/QLearning/i2q_rnn.py +alg=i2q_rnn_lbf alg.ENV_KWARGS.field_size=10 alg.ENV_KWARGS.num_agents=3 alg.ENV_KWARGS.max_num_food=3
```

Matrix games (`ENV_NAME: MatrixGame`, `jaxmarl/environments/matrix_game`) port
PyMARL's cooperative two-player games: `climbing` (optimum (0, 0) = 11 next to
-30 penalties, the textbook case of relative overgeneralisation), `penalty`
(`penalty_k`, default -15), `partially_stochastic_climbing` and
`fully_stochastic_climbing`. Both agents receive the payoff; `episode_limit`
repeats the game. The `*_rnn_matrix` presets follow PyMARL (5e4 one-step
episodes, epsilon annealed over the whole run); RAC and HC-RAC use the payoff
range as return range, and HC-RAC uses the return features only (one-step
episodes have no TCL windows).

Predator-prey (`ENV_NAME: StagHunt`, `jaxmarl/environments/stag_hunt`) ports
the DCG / PyMARL `stag_hunt` grid world. The defaults and the `*_rnn_staghunt`
presets are PyMARL's `pred_prey_punish`: 10x10, 8 agents, 8 stags, 5x5 views,
a stag is caught when two adjacent agents play `catch` (+10, the catchers
freeze), and a lone catcher costs -2, so independent learners learn to avoid
catching. Actions are masked as in PyMARL (frozen agents only stay, no moves
into walls, `catch` only next to a prey); `CTRolloutManager` reads the env's
masks for this env. `pred_prey_original` is `capture_action=False`,
`capture_freezes=False`, `miscapture_punishment=0`, `reward_stag=50`,
`reward_time=-0.1`. The presets keep 1000 episodes in the replay buffer
(PyMARL: 5000, about 3 GB of GPU memory here).

```bash
python baselines/QLearning/iql_rnn.py +alg=ql_rnn_matrix alg.ENV_KWARGS.game=penalty
python baselines/QLearning/rac_rnn.py +alg=hcrac_rnn_matrix
python baselines/QLearning/rac_rnn.py +alg=rac_rnn_staghunt
python baselines/QLearning/i2q_rnn.py +alg=i2q_rnn_staghunt
```

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