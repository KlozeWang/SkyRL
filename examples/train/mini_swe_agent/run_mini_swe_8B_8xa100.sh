set -x

# Colocated GRPO training+generation for Qwen/Qwen3-8B on the SWE-Bench task.
# Uses 1 node with 8 A100 GPUs.
# uv run --isolated examples/train/mini_swe_agent/preprocess_swegym.py --output_dir ~/data/swe_gym_subset
# bash /mnt/xujiawang/skyrl/examples/swe/run_mini_swe_8B_8xa100.sh
#
# Optional teacher usage:
# TEACHER_API_KEY=111 bash /mnt/xujiawang/skyrl/examples/swe/run_mini_swe_8B_8xa100.sh \
#   generator.teacher.enabled=true \
#   generator.teacher.base_url=http://4.155.72.80:10189 \
#   generator.teacher.model=gpt-5.2 \
#   generator.teacher.tool_env_mode=shared \
#   generator.teacher.max_turns=20 \
#   generator.teacher.output_token_penalty_coef=0.05

export MSWEA_COST_TRACKING="ignore_errors"
export LITELLM_MODEL_REGISTRY_PATH="${LITELLM_MODEL_REGISTRY_PATH:-examples/train/mini_swe_agent/litellm.json}"
# Nested Podman in Singularity needs --pid=host to avoid denied /proc mounts.
# Set MINISWE_PODMAN_USE_HOST_PID=0 only if your runtime supports private PID namespaces.
export MINISWE_PODMAN_USE_HOST_PID="${MINISWE_PODMAN_USE_HOST_PID:-1}"
DATA_DIR="/mnt/xujiawang/skyrl/examples/swe/swe_gym_subset"
OUTPUT_ROOT="/mnt/xujiawang/skyrl/examples/swe"
CKPT_PATH="$OUTPUT_ROOT/ckpts/llm_mini_swe_8B"

# Save trajectories here for debugging.
MINISWE_TRAJ_DIR="$OUTPUT_ROOT/trajs/mini_swe_8B"

NUM_GPUS=8
NNODES=1
NUM_INFERENCE_ENGINES=4
TP_SIZE=2
LOGGER=wandb
MINISWE_ROLLOUT_NUM_CPUS="${MINISWE_ROLLOUT_NUM_CPUS:-2}"

mkdir -p "$CKPT_PATH" "$MINISWE_TRAJ_DIR"

# We use a small batch size here for demonstration.
# NOTE (sumanthrh): The `generator.max_turns` here is actually unused, and we use the `step_limit`
# from the `swebench.yaml` file. This simply has to be a value > 1.
uv run --isolated --extra fsdp --extra miniswe --env-file examples/train/mini_swe_agent/.env.miniswe -m examples.train.mini_swe_agent.main_mini_swe \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen3-8B" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.policy_num_nodes=$NNODES \
  trainer.placement.ref_num_nodes=$NNODES \
  trainer.policy.sequence_parallel_size=2 \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$TP_SIZE \
  trainer.epochs=20 \
  trainer.eval_batch_size=25 \
  trainer.eval_before_train=false \
  trainer.eval_interval=10000 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=16 \
  trainer.policy_mini_batch_size=16 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.dump_data_batch=true \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=4096 \
  generator.sampling_params.max_generate_length=4096 \
  generator.max_input_length=30720 \
  generator.max_turns=20 \
  generator.rollout_num_cpus=$MINISWE_ROLLOUT_NUM_CPUS \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=True \
  generator.inference_engine.enable_http_endpoint=True \
  generator.inference_engine.http_endpoint_host='127.0.0.1' \
  generator.inference_engine.http_endpoint_port=8001 \
  generator.inference_engine.engine_init_kwargs.enable_auto_tool_choice=true \
  generator.inference_engine.engine_init_kwargs.tool_call_parser="hermes" \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.batched=true \
  generator.n_samples_per_prompt=4 \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="mini_swe" \
  trainer.run_name="mini_swe_8B_swe_gym_a100x8" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$CKPT_PATH" \
  generator.miniswe_config_path="examples/train/mini_swe_agent/swebench.yaml" \
  generator.miniswe_traj_dir="$MINISWE_TRAJ_DIR" \
  "$@"
