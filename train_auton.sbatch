#!/bin/bash
#SBATCH --job-name=marl_training
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --partition=general              # Use 'general' partition for production runs
#SBATCH --qos=qos_general                # QOS matching the partition
#SBATCH --nodelist=gpu17                 # Force job to run on gpu17 (for debugging)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32               # Adjust based on available CPUs
#SBATCH --mem=200G                       # Memory allocation
#SBATCH --time=47:59:00                  # Max time for general partition is 2d (48h)
#SBATCH --gres=gpu:a6000:4               # Request 4 A6000 GPUs (can also use v100, a5000, rtx_2080_ti)

# For debugging/testing, use these settings instead:
# #SBATCH --partition=debug
# #SBATCH --qos=qos_debug
# #SBATCH --time=02:00:00
# #SBATCH --gres=gpu:a6000:2

# For long runs with preemption tolerance, use these settings:
# #SBATCH --partition=preempt
# #SBATCH --qos=qos_preempt
# #SBATCH --time=7-00:00:00
# #SBATCH --gres=gpu:a6000:4

echo "=== Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Working directory: $SLURM_SUBMIT_DIR"
echo "======================="

# ============================================================================
# SETUP ENVIRONMENT (COMMON SCRIPT)
# ============================================================================

PROJECT_DIR="/zfsauton/scratch/cpulling/Verlog"
ENV_PATH="/zfsauton/scratch/cpulling/conda_envs/verlog"
DATA_DIR="/zfsauton/scratch/cpulling/data/gsm8k"

# Source common setup script
SETUP_SCRIPT="$PROJECT_DIR/setup_environment.sh"
if [ ! -f "$SETUP_SCRIPT" ]; then
    echo "❌ ERROR: Setup script not found at $SETUP_SCRIPT"
    exit 1
fi

source "$SETUP_SCRIPT" "$PROJECT_DIR" "$ENV_PATH" "$DATA_DIR"

# Python binary is now set by setup script
PYTHON_BIN="$ENV_PATH/bin/python3"

# Set up environment variables
export HF_HOME="/zfsauton/scratch/${USER}/models_cache"
mkdir -p "$HF_HOME"

export TMPDIR="/zfsauton/scratch/${USER}/tmp"
export RAY_TMPDIR="$TMPDIR"
mkdir -p "$TMPDIR"

# Increase file descriptor and process limits (crucial for Ray/vLLM)
ulimit -n 65535 2>/dev/null || true
ulimit -u 65535 2>/dev/null || true

# Clean up any stale Ray processes from previous runs
echo "=== Cleaning up stale Ray processes ==="
$PYTHON_BIN -m ray stop --force 2>/dev/null || true
pkill -9 -f "ray::" 2>/dev/null || true
rm -rf /tmp/ray 2>/dev/null || true
rm -rf ${TMPDIR}/ray 2>/dev/null || true

# Ray configuration
export RAY_NUM_CPUS=${SLURM_CPUS_PER_TASK}
export RAY_INCLUDE_DASHBOARD=0
export RAY_DISABLE_DASHBOARD=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Ray timeout and debugging settings
export RAY_raylet_startup_timeout_ms=120000
export RAY_gcs_server_request_timeout_seconds=60
export RAY_BACKEND_LOG_LEVEL=warning

# GPU configuration
NUM_GPUS_PER_NODE=4  # Match this to --gres setting
unset ROCR_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS_PER_NODE-1)))

# vLLM configuration
export VLLM_USE_V1=1
export VERL_AUTO_PADDING=TRUE

# WandB API Key - REPLACE WITH YOUR OWN
export WANDB_API_KEY="5cd6b6564c0f79ebf65ce694e5e3b54b725ff225"

# Project paths
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
LOG_PATH="${VERL_AGENT_IO_LOG_PATH:-$PROJECT_DIR/logs/agent_model_train_auton_${SLURM_JOB_ID}.log}"
mkdir -p "$(dirname "$LOG_PATH")"
: > "$LOG_PATH"
export VERL_AGENT_IO_LOG_PATH="$LOG_PATH"
export VERL_GAME_LOG_PATH="$PROJECT_DIR/logs/game_log_train_auton_${SLURM_JOB_ID}.log"

# Training hyperparameters
NUM_ENVS=32
BATCH_SIZE=256
MINI_BATCH_SIZE=$((BATCH_SIZE))
MICRO_BATCH_SIZE=4
FORWARD_BATCH_SIZE=$((4 * MICRO_BATCH_SIZE))
OFFLOAD=false
PPO_EPOCHS=2

echo "=== Training Configuration ==="
echo "GPUs: $NUM_GPUS_PER_NODE"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Num Envs: $NUM_ENVS"
echo "Batch Size: $BATCH_SIZE"
echo "Mini Batch Size: $MINI_BATCH_SIZE"
echo "Micro Batch Size: $MICRO_BATCH_SIZE"
echo "=============================="

source "$PROJECT_DIR/claude-tools/exp_logging/sbatch_logging.sh"
EXP_WHAT_CHANGED="${EXP_WHAT_CHANGED:-train_auton legacy submission}"
EXP_WANDB_PROJECT=wait_ratio_exps
EXP_NAME="train_auton_${SLURM_JOB_ID}"
EXP_PROJECT_DIR="$PROJECT_DIR"
EXP_MODEL=Qwen/Qwen2.5-3B-Instruct
EXP_ADV_ESTIMATOR=gae
EXP_BATCH_SIZE=$BATCH_SIZE
EXP_NUM_ENVS=$NUM_ENVS
EXP_GPUS=$NUM_GPUS_PER_NODE
EXP_ACTOR_LR=1e-6
EXP_CRITIC_LR=1e-5
EXP_PPO_EPOCHS=$PPO_EPOCHS
EXP_CRITIC_WARMUP_STEPS=10
EXP_CRITIC_WARMUP_REPEAT=40
EXP_ROLLOUT_N=1
EXP_TOKEN_BUDGET=1000
EXP_STUDENTS_PER_BATCH=5
EXP_FEATURE_DIM=5
exp_log_start
trap exp_log_finalize EXIT

# Run training
$PYTHON_BIN -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='gsm8k_multiturn_grpo' \
    algorithm.adv_estimator=gae \
    data.train_batch_size=${BATCH_SIZE} \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS} \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=${OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OFFLOAD} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${FORWARD_BATCH_SIZE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.agent.num_workers=${NUM_ENVS} \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${FORWARD_BATCH_SIZE} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=True \
    trainer.balance_batch=False \
    trainer.critic_warmup=10 \
    trainer.critic_warmup_batch_repeat_times=40 \
    trainer.critic_warmup_batch_divide_ratio=4 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=wait_ratio_exps \
    trainer.experiment_name="marl_auton_${SLURM_JOB_ID}" \
    trainer.n_gpus_per_node=${NUM_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=30 \
    trainer.total_epochs=60 \
    trainer.val_before_train=True \
    envs.num_envs=${NUM_ENVS} \
    envs.env_name=async_ticker_admissions \
    +envs.env_config.professor_ids='["prof_1","prof_2","prof_3"]' \
    +envs.env_config.students_per_batch=5 \
    +envs.env_config.token_budget=1000 \
    +envs.env_config.feature_dim=5 \
    +envs.env_config.vote_threshold=0.5 \
    +envs.env_config.max_steps=100 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=Qwen/Qwen2.5-3B-Instruct \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_epochs=${PPO_EPOCHS} \
    critic.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    critic.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
    critic.model.fsdp_config.param_offload=${OFFLOAD} \
    critic.model.fsdp_config.optimizer_offload=${OFFLOAD} \
    critic.ppo_max_token_len_per_gpu=16384 \
    critic.forward_max_token_len_per_gpu=16384 \
    critic.forward_micro_batch_size_per_gpu=${FORWARD_BATCH_SIZE} \
    data.train_files=/zfsauton/scratch/${USER}/data/gsm8k/test.parquet \
    data.val_files=/zfsauton/scratch/${USER}/data/gsm8k/test.parquet \
    $@

EXIT_CODE=$?

echo "=== Job Completion ==="
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "======================"

exit $EXIT_CODE
