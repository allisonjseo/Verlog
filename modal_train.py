"""
Modal equivalent of train_4_16384_shortest_rhea.sbatch.

Usage:
    modal run modal_train.py

First run pulls the verlai/verl Docker image and caches it.
Subsequent runs skip that step and go straight to training.

Model weights (Qwen/Qwen2.5-3B-Instruct) are cached in the
hf-model-cache volume and reused across runs.
Training logs are saved to the verl-training-logs volume.
"""
import os
import modal

app = modal.App("marl4llms")

_LOCAL_VERLOG = os.path.dirname(os.path.abspath(__file__))

# ── Secrets / volumes ──────────────────────────────────────────────────────────
wandb_secret = modal.Secret.from_dict({
    "WANDB_API_KEY": "5cd6b6564c0f79ebf65ce694e5e3b54b725ff225"
})
hf_cache   = modal.Volume.from_name("hf-model-cache",     create_if_missing=True)
train_logs = modal.Volume.from_name("verl-training-logs", create_if_missing=True)

# ── Container image ────────────────────────────────────────────────────────────
# Use the official verlai/verl image which ships with torch, flash-attn,
# vLLM, Ray, and all other heavy dependencies pre-built.
# Local source is synced at container startup (copy=False) so edits to
# verl/ or examples/ take effect without rebuilding the image.
image = (
    modal.Image.from_registry(
        "verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2"
    )
    .env({
        "HF_HOME":    "/hf-cache",
        "VLLM_USE_V1": "1",
        "PYTHONPATH": "/verl",
    })
    .pip_install("gym", "opencv-python-headless>=4.9.0")
    .add_local_dir(
        _LOCAL_VERLOG,
        remote_path="/verl",
        ignore=lambda p: (
            # skip heavy / irrelevant top-level dirs
            p.parts[0] in {
                "logs", "outputs", ".git", "wandb",
                "docs", "docker", "tests", "third_party",
            }
            # skip any nested __pycache__
            or "__pycache__" in p.parts
            # skip log / slurm / shell files
            or p.suffix in {".log", ".err", ".out", ".sbatch"}
        ),
    )
    # Inline the tiny GSM8K parquet (605 KB) used for train & val data
    .add_local_file(
        "/zfsauton2/home/ajseo/data/gsm8k/test.parquet",
        remote_path="/data/gsm8k/test.parquet",
    )
)

# ── Training function ──────────────────────────────────────────────────────────
@app.function(
    image=image,
    gpu="A100:4",          # 4× A100-40 GB; matches the 4× A6000-48 GB in sbatch
    secrets=[wandb_secret],
    volumes={
        "/hf-cache": hf_cache,
        "/logs":     train_logs,
    },
    timeout=86400,         # 24 h — Modal maximum (sbatch runs up to 36 h; increase save_freq if needed)
    cpu=16,
    memory=245760,         # 240 GB RAM — matches sbatch --mem=240G
)
def run_training():
    import os
    import resource
    import subprocess
    import sys

    # Mirror sbatch: ulimit -n 65535
    resource.setrlimit(resource.RLIMIT_NOFILE, (65535, 65535))

    os.makedirs("/logs/train_4_16384_shortest", exist_ok=True)
    os.environ.update({
        "CUDA_VISIBLE_DEVICES":   "0,1,2,3",
        "VERL_AGENT_IO_LOG_PATH": "/logs/train_4_16384_shortest/agent_model.log",
        "VERL_GAME_LOG_PATH":     "/logs/train_4_16384_shortest/game_log.log",
    })

    subprocess.run(
        [
            sys.executable, "-m", "verl.trainer.main_ppo",
            "--config-path=/verl/examples/sglang_multiturn/config",
            "--config-name=gsm8k_multiturn_grpo",
            # ── algorithm ──────────────────────────────────────────────────────
            "algorithm.adv_estimator=gae",
            "algorithm.use_kl_in_reward=True",
            # ── data ───────────────────────────────────────────────────────────
            "data.train_batch_size=256",
            "data.max_prompt_length=4096",
            "data.max_response_length=512",
            "data.filter_overlong_prompts=True",
            "data.truncation=error",
            "data.return_raw_chat=True",
            "data.train_files=/data/gsm8k/test.parquet",
            "data.val_files=/data/gsm8k/test.parquet",
            # ── actor / rollout / ref ───────────────────────────────────────────
            "actor_rollout_ref.rollout.mode=async",
            "actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct",
            "actor_rollout_ref.actor.optim.lr=1e-6",
            "actor_rollout_ref.model.use_remove_padding=True",
            "actor_rollout_ref.actor.ppo_mini_batch_size=256",
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4",
            "actor_rollout_ref.actor.use_kl_loss=False",
            "actor_rollout_ref.actor.ppo_epochs=2",
            "actor_rollout_ref.actor.entropy_coeff=0.001",
            "actor_rollout_ref.model.enable_gradient_checkpointing=True",
            "actor_rollout_ref.actor.fsdp_config.param_offload=False",
            "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16",
            "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
            "actor_rollout_ref.rollout.name=vllm",
            "actor_rollout_ref.rollout.gpu_memory_utilization=0.4",
            "actor_rollout_ref.rollout.agent.num_workers=32",
            "actor_rollout_ref.rollout.n=1",
            "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16",
            "actor_rollout_ref.ref.fsdp_config.param_offload=True",
            "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384",
            "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384",
            "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384",
            # ── critic ─────────────────────────────────────────────────────────
            "critic.optim.lr=1e-5",
            "critic.model.use_remove_padding=True",
            "critic.model.path=Qwen/Qwen2.5-3B-Instruct",
            "critic.model.enable_gradient_checkpointing=True",
            "critic.ppo_epochs=2",
            "critic.ppo_micro_batch_size_per_gpu=4",
            "critic.ppo_mini_batch_size=256",
            "critic.model.fsdp_config.param_offload=False",
            "critic.model.fsdp_config.optimizer_offload=False",
            "critic.ppo_max_token_len_per_gpu=16384",
            "critic.forward_max_token_len_per_gpu=16384",
            "critic.forward_micro_batch_size_per_gpu=16",
            # ── trainer ────────────────────────────────────────────────────────
            "trainer.balance_batch=False",
            "trainer.critic_warmup=10",
            "trainer.critic_warmup_batch_repeat_times=40",
            "trainer.critic_warmup_batch_divide_ratio=4",
            'trainer.logger=["console","wandb"]',
            "trainer.project_name=interactive",
            "trainer.experiment_name=marl_4_16384_shortest",
            "trainer.n_gpus_per_node=4",
            "trainer.nnodes=1",
            "trainer.save_freq=-1",
            "trainer.test_freq=30",
            "trainer.total_epochs=15",
            "trainer.val_before_train=True",
            # ── environment ────────────────────────────────────────────────────
            "envs.num_envs=32",
            "envs.env_name=async_ticker_admissions",
            '+envs.env_config.professor_ids=["prof_1","prof_2","prof_3"]',
            "+envs.env_config.students_per_batch=5",
            "+envs.env_config.token_budget=1000",
            "+envs.env_config.feature_dim=5",
            "+envs.env_config.vote_threshold=0.5",
            "+envs.env_config.max_steps=50",
        ],
        cwd="/verl",
        check=True,
    )


@app.local_entrypoint()
def main():
    run_training.remote()
