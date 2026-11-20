#!/bin/bash
# Resumable SMAX runs (IQL / RAC / HC-RAC / I2Q, seed 0) for a shared GPU box.
#
#   baselines/QLearning/scripts/run_smax_resumable.sh <MAP> [method ...]
#
# Methods: iql rac hcrac i2q (default: all four, in that order). They run one
# at a time in a single background queue under flock, with
# CHECKPOINT_DIR=checkpoints, so:
#   - a crash restarts the current method from its last checkpoint (same W&B run);
#   - finished methods are skipped, so rerunning this script after a reboot (by
#     hand or from `crontab @reboot`) continues where the queue stopped;
#   - a second launch while the queue is running does nothing.
# KL_WEIGHT is 0.001 on 3s_vs_5z (RAC paper) and 0.01 elsewhere.
# Env vars: SEED (0), TOTAL_TIMESTEPS (preset default), TEST_INTERVAL (preset
# default 0.05 = fraction of training between evals; 0.01 gives 100 eval points),
# MAX_RETRIES (20), RETRY_SLEEP (60), WANDB_KEY_FILE, WANDB_RUN_GROUP.
set -u
MAP=${1:?usage: $0 MAP [iql rac hcrac i2q]}
shift

source ~/miniconda3/etc/profile.d/conda.sh
conda activate jaxmarl
cd ~/JaxMARL || exit 1

SEED=${SEED:-0}
L=logs/smax_$MAP
mkdir -p "$L" checkpoints

methods=("$@")
[ ${#methods[@]} -eq 0 ] && methods=(iql rac hcrac i2q)

if [ "${SMAX_WORKER:-0}" != 1 ]; then
    SMAX_WORKER=1 nohup flock -n "$L/queue.lock" "$0" "$MAP" "${methods[@]}" >> "$L/queue.log" 2>&1 &
    echo "queue ${methods[*]} pid $! (logs in $L/)"
    exit 0
fi

# ---- worker: methods one after another, each retried until it finishes ----
KL=0.01
[ "$MAP" = 3s_vs_5z ] && KL=0.001
STEPS=()
[ -n "${TOTAL_TIMESTEPS:-}" ] && STEPS=(alg.TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS")
[ -n "${TEST_INTERVAL:-}" ] && STEPS+=(alg.TEST_INTERVAL="$TEST_INTERVAL")

export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
export WANDB_API_KEY=$(cat "${WANDB_KEY_FILE:-$HOME/.rac_wandb_key}")
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-jaxmarl-smax-$MAP}

run_method() {
    local m=$1 alg args meta try
    case $m in
        iql)   alg=iql_rnn;   args=(baselines/QLearning/iql_rnn.py +alg=ql_rnn_smax) ;;
        rac)   alg=rac_rnn;   args=(baselines/QLearning/rac_rnn.py +alg=rac_rnn_smax alg.KL_WEIGHT=$KL) ;;
        hcrac) alg=hcrac_rnn; args=(baselines/QLearning/rac_rnn.py +alg=hcrac_rnn_smax alg.KL_WEIGHT=$KL) ;;
        i2q)   alg=i2q_rnn;   args=(baselines/QLearning/i2q_rnn.py +alg=i2q_rnn_smax) ;;
        *) echo "unknown method $m"; return 1 ;;
    esac
    meta=checkpoints/${alg}_HeuristicEnemySMAX_${MAP}_seed${SEED}/meta.json
    for try in $(seq 1 "${MAX_RETRIES:-20}"); do
        if grep -q '"finished": true' "$meta" 2>/dev/null; then
            echo "[$(date '+%F %T')] $m on $MAP finished"
            return 0
        fi
        echo "[$(date '+%F %T')] start $m on $MAP (attempt $try)"
        python "${args[@]}" alg.MAP_NAME="$MAP" "${STEPS[@]}" SEED="$SEED" PROJECT=dtde \
            ENTITY=null WANDB_MODE=online SAVE_PATH=models CHECKPOINT_DIR=checkpoints \
            >> "$L/$m.log" 2>&1
        echo "[$(date '+%F %T')] $m exited with code $?"
        grep -q '"finished": true' "$meta" 2>/dev/null || sleep "${RETRY_SLEEP:-60}"
    done
    echo "[$(date '+%F %T')] $m gave up after ${MAX_RETRIES:-20} attempts"
    return 1
}

for m in "${methods[@]}"; do
    run_method "$m"
done
echo "[$(date '+%F %T')] queue done"
