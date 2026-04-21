import asyncio
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import subprocess
import yaml
import traceback
import ray
from pathlib import Path

from minisweagent.models import get_model
from minisweagent.agents.default import DefaultAgent, ExecutionTimeoutError
from minisweagent.config import get_config_path
from .mini_swe_utils import close_environment, evaluate_trajectory, get_sb_environment, execute_env_command
from .teacher import (
    MiniSWETeacherConfig,
    TeacherClient,
    TeacherSessionResult,
    TeacherUsage,
    build_teacher_observation,
    calculate_teacher_penalty,
    parse_askteacher_command,
)

try:
    from minisweagent.run.utils.save import save_traj as _legacy_save_traj
except ModuleNotFoundError:
    _legacy_save_traj = None

from skyrl.train.config import GeneratorConfig, SkyRLGymConfig
from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator, GeneratorOutput, GeneratorInput
from skyrl.train.generators.base import TrajectoryID, TrainingPhase, BatchMetadata
from skyrl.backends.skyrl_train.inference_engines.base import ConversationType
from skyrl.backends.skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl.backends.skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl.train.generators.utils import (
    get_rollout_metrics,
    get_response_ids_and_loss_mask_from_messages,
)


@dataclass
class MiniSWEGeneratorConfig(GeneratorConfig):
    """Extended generator config with Mini-SWE-Agent-specific fields."""

    miniswe_config_path: str = ""
    miniswe_traj_dir: str = ""
    teacher: MiniSWETeacherConfig = field(default_factory=MiniSWETeacherConfig)


class DefaultAgentWithReminder(DefaultAgent):
    def __init__(
        self,
        *args,
        teacher_cfg: Optional[MiniSWETeacherConfig] = None,
        instance: Optional[dict] = None,
        sweagent_config: Optional[dict] = None,
        data_source: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.teacher_cfg = teacher_cfg or MiniSWETeacherConfig()
        self.instance = instance or {}
        self.sweagent_config = sweagent_config or {}
        self.data_source = data_source or ""
        self.teacher_usage = TeacherUsage()
        self.teacher_records: List[Dict[str, Any]] = []
        self._teacher_client: Optional[TeacherClient] = None
        self._teacher_env = None

    def _get_turns_remaining(self) -> int:
        n_calls = getattr(self, "n_calls", getattr(self.model, "n_calls", 0))
        return self.config.step_limit - n_calls

    def _append_reminder(self, content: str) -> str:
        remaining = self._get_turns_remaining()
        if remaining == 1:
            return f"{content}\nREMINDER: You only have 1 turn left. Please provide the final answer"
        if remaining > 1:
            return f"{content}\nREMINDER: You have {remaining} turns left to arrive at the solution."
        return content

    def get_observation(self, response: dict) -> dict:
        """Execute the action and return the output."""
        output = self.execute_action(self.parse_action(response))
        observation = self.render_template(self.config.action_observation_template, output=output)
        observation = self._append_reminder(observation)
        self.add_message("user", observation)
        return output

    def execute_action(self, action: dict) -> dict:
        command = action["action"]
        try:
            teacher_request = parse_askteacher_command(command)
        except ValueError as e:
            output = {"returncode": 2, "output": str(e)}
            return output | {"action": command}

        if teacher_request is not None:
            output = self._execute_teacher_request(teacher_request)
            return output | {"action": command}

        try:
            output = execute_env_command(self.env, command)
        except (TimeoutError, subprocess.TimeoutExpired) as e:
            output = getattr(e, "output", "") or ""
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            raise ExecutionTimeoutError(
                self.render_template(self.config.timeout_template, action=action, output=output)
            )
        self.has_finished(output)
        return output | {"action": command}

    def _execute_teacher_request(self, teacher_request) -> dict:
        if not self.teacher_cfg.enabled:
            return {
                "returncode": 127,
                "output": "askteacher is disabled. Set generator.teacher.enabled=true to enable it.",
            }

        self.teacher_usage.requests += 1
        try:
            result = self._get_teacher_client().run(
                teacher_request,
                instance=self.instance,
                learner_messages=self.messages,
                execute_command=self._execute_teacher_tool_command,
            )
        except Exception as e:
            result = TeacherSessionResult(
                answer="",
                usage=TeacherUsage(),
                error=f"{type(e).__name__}: {e}",
            )

        self.teacher_usage.calls += result.usage.calls
        self.teacher_usage.output_tokens += result.usage.output_tokens
        self.teacher_usage.tool_calls += result.usage.tool_calls
        record_usage = result.usage.to_dict()
        record_usage["requests"] = 1
        self.teacher_records.append(
            {
                "prompt": teacher_request.prompt,
                "think_level": teacher_request.think_level,
                "usage": record_usage,
                "error": result.error,
                "answer": result.answer,
                "transcript": result.transcript,
            }
        )
        return {
            "returncode": 1 if result.error else 0,
            "output": build_teacher_observation(
                result,
                return_transcript=self.teacher_cfg.return_teacher_transcript_to_learner,
            ),
        }

    def _get_teacher_client(self) -> TeacherClient:
        if self._teacher_client is None:
            self._teacher_client = TeacherClient(self.teacher_cfg)
        return self._teacher_client

    def _execute_teacher_tool_command(self, command: str, timeout: Optional[int]) -> dict:
        try:
            if self.teacher_cfg.tool_env_mode == "shared":
                return execute_env_command(self.env, command, timeout=timeout)
            if self.teacher_cfg.tool_env_mode not in ("shadow", "readonly"):
                return {
                    "returncode": 2,
                    "output": f"Unsupported teacher.tool_env_mode={self.teacher_cfg.tool_env_mode}",
                }
            if self._teacher_env is None:
                self._teacher_env = get_sb_environment(
                    copy.deepcopy(self.sweagent_config),
                    self.instance,
                    self.data_source,
                )
            return execute_env_command(self._teacher_env, command, timeout=timeout)
        except Exception as e:
            return {"returncode": 1, "output": f"{type(e).__name__}: {e}"}

    def close_teacher_environment(self) -> None:
        close_environment(self._teacher_env)
        self._teacher_env = None

    def execute_actions(self, message: dict) -> list[dict]:
        """Append the reminder to v2 observation messages after execution."""
        observations = super().execute_actions(message)
        if observations and observations[-1].get("role") == "user":
            observations[-1]["content"] = self._append_reminder(observations[-1]["content"])
        return observations


def save_traj_compat(
    agent: Optional[DefaultAgent],
    path: Path,
    *,
    exit_status: Optional[str] = None,
    result: Optional[dict | str] = None,
    extra_info: Optional[dict] = None,
    reward: float = 0,
    eval_error: Optional[str] = None,
) -> None:
    if _legacy_save_traj is not None:
        _legacy_save_traj(
            agent,
            path,
            exit_status=exit_status,
            result=result,
            extra_info=extra_info,
            reward=reward,
            eval_error=eval_error,
        )
        return

    extra_dict = {
        "info": extra_info or {},
        "reward": reward,
        "eval_error": eval_error,
    }
    if result is not None:
        extra_dict["result"] = result
    if exit_status is not None:
        extra_dict["info"]["exit_status"] = exit_status
    if isinstance(result, str):
        extra_dict["info"]["submission"] = result
    elif isinstance(result, dict):
        extra_dict["info"]["result"] = result

    if agent is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
        return

    agent.save(path, extra_dict)


@ray.remote(num_cpus=0.01)
def init_and_run(
    instance: dict,
    litellm_model_name: str,
    sweagent_config: dict,
    generator_cfg: GeneratorConfig,
    data_source: str,
    sampling_params: dict,
    trajectory_id: TrajectoryID,
    global_step: int,
    training_phase: TrainingPhase,
):
    from loguru import logger

    model_config = sweagent_config.get("model", {})
    # Use new sampling parameters
    # Can also have custom sampling parameters per trajectory (ex: custom max tokens)
    model_config.setdefault("model_kwargs", {}).update(sampling_params)
    model = get_model(litellm_model_name, model_config)

    agent = None
    env = None
    extra_info = None
    exit_status = "UnknownError"
    result = None
    reward = 0
    error = None
    trajectory_metrics: Dict[str, Any] = {}
    try:
        env = get_sb_environment(sweagent_config, instance, data_source)
        agent = DefaultAgentWithReminder(
            model,
            env,
            teacher_cfg=getattr(generator_cfg, "teacher", MiniSWETeacherConfig()),
            instance=instance,
            sweagent_config=sweagent_config,
            data_source=data_source,
            **sweagent_config.get("agent", {}),
        )
        run_result = agent.run(instance["problem_statement"])  # type: ignore[arg-type]
        # logger.info(f"Finished running agent for instance {instance['instance_id']} with result: {run_result}")
        if isinstance(run_result, tuple):
            exit_status, result = run_result
        else:
            exit_status = run_result.get("exit_status", "")
            result = run_result.get("submission", "")
    except Exception as e:
        logger.error(f"Error processing instance {instance['instance_id']}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        error = str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        # Create trajectory directory with proper structure: step_{global_step}/{train/eval}
        path = Path(generator_cfg.miniswe_traj_dir) / f"step_{global_step}" / training_phase
        path.mkdir(parents=True, exist_ok=True)
        # Use instance_id and repetition_id for meaningful filename: {instance_id}_{repetition_id}.json
        instance_id = instance["instance_id"]
        filename = f"{instance_id}_{trajectory_id.repetition_id}.json"
        path = path / filename
        if agent is not None:
            eval_error = None
            correctness_reward = 0.0
            try:
                result = evaluate_trajectory(instance, result, sweagent_config, data_source)
                correctness_reward = float(result["resolved"])
                eval_error = result["eval_error"]
                if eval_error:
                    error = eval_error
                    logger.debug(f"Error during evaluation {eval_error}")
            except Exception as e:
                logger.debug(f"Error during evaluation {e}")
                logger.debug(f"traceback: {traceback.format_exc()}")
                eval_error = str(e)
                error = str(e)

            teacher_cfg = getattr(generator_cfg, "teacher", MiniSWETeacherConfig())
            teacher_usage = agent.teacher_usage.to_dict()
            teacher_penalty = calculate_teacher_penalty(agent.teacher_usage.output_tokens, teacher_cfg)
            reward = correctness_reward - teacher_penalty
            teacher_usage["penalty"] = teacher_penalty
            trajectory_metrics = {
                "teacher_requests": agent.teacher_usage.requests,
                "teacher_api_calls": agent.teacher_usage.calls,
                "teacher_output_tokens": agent.teacher_usage.output_tokens,
                "teacher_tool_calls": agent.teacher_usage.tool_calls,
                "teacher_penalty": teacher_penalty,
                "correctness_reward": correctness_reward,
                "penalized_reward": reward,
            }
            save_extra_info = dict(extra_info or {})
            save_extra_info.update(
                {
                    "correctness_reward": correctness_reward,
                    "penalized_reward": reward,
                    "teacher": teacher_usage,
                    "teacher_records": agent.teacher_records,
                }
            )
            save_traj_compat(
                agent,
                path,
                exit_status=exit_status,
                result=result,
                extra_info=save_extra_info,
                reward=reward,
                eval_error=eval_error,
            )
            agent.close_teacher_environment()
        close_environment(env)

    return (agent.messages if agent is not None else [], reward, error, trajectory_metrics)


class MiniSweAgentGenerator(SkyRLGymGenerator):
    def __init__(
        self,
        generator_cfg: GeneratorConfig,
        skyrl_gym_cfg: SkyRLGymConfig,
        inference_engine_client: InferenceEngineClient,
        tokenizer,
        model_name: str,
    ):

        # Call parent constructor first
        super().__init__(generator_cfg, skyrl_gym_cfg, inference_engine_client, tokenizer)

        self.http_server_inference_engine_client_host = generator_cfg.inference_engine.http_endpoint_host

        self.http_server_inference_engine_client_port = generator_cfg.inference_engine.http_endpoint_port

        self.base_url = (
            f"http://{self.http_server_inference_engine_client_host}:{self.http_server_inference_engine_client_port}"
        )
        self.generator_cfg = generator_cfg
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.litellm_model_name = "openai/" + self.model_name

        if self.generator_cfg.chat_template.name_or_path is not None:
            raise NotImplementedError("MiniSWEAgentGenerator doesn't support custom chat template")

    async def minisweagent_agent_loop(
        self,
        prompt: ConversationType,
        env_extras: Dict[str, Any],
        max_tokens: int,
        max_input_length: int,
        sampling_params: Dict[str, Any],
        trajectory_id: TrajectoryID,
        batch_metadata: BatchMetadata,
    ) -> Tuple[List[int], float, str, List[int], List[int], Optional[List[int]], Dict[str, Any]]:

        sweagent_config = yaml.safe_load(get_config_path(self.generator_cfg.miniswe_config_path).read_text())
        # NOTE (sumanthrh): Input `prompt` is not used here because mini-swe-agent uses a similar entry from the `instance` obj
        messages, reward, error, trajectory_metrics = await init_and_run.remote(
            env_extras["instance"],
            self.litellm_model_name,
            sweagent_config,
            self.generator_cfg,
            env_extras["data_source"],
            sampling_params,
            trajectory_id,
            batch_metadata.global_step,
            batch_metadata.training_phase,
        )
        if not len(messages):
            return None, None, None, None, None, None, trajectory_metrics

        # TODO (sumanthrh): This is currently hardcoded for SWEBench with 2 initial messages (system and user).
        response_messages = [message for message in messages[2:] if message["role"] in ("user", "assistant")]

        # from loguru import logger
        # logger.info("================================")
        # for message in response_messages:
        #     logger.info(f"Response message: {message}")

        for message in messages[:2]:
            assert message["role"] in (
                "system",
                "user",
            ), "Expected the first two messages to be system and user messages"

        initial_input_ids = self.tokenizer.apply_chat_template(
            messages[:2], add_generation_prompt=False, return_dict=False, tokenize=True
        )
        initial_prompt_length = len(initial_input_ids)

        # We remove trailing `user` messages - this is added by Mini-SWE-Agent to capture the final git diff for the trajectory
        last_idx = len(response_messages) - 1
        while last_idx >= 0 and response_messages[last_idx]["role"] == "user":
            last_idx -= 1
        if last_idx < 0:
            raise ValueError(
                "Found no assistant messages. Please ensure that your environment is configured correctly and the `OPENAI_BASE_URL` points to the HTTP server from the inference engine client"
            )
        response_messages = response_messages[: last_idx + 1]

        response_ids, loss_mask, _ = get_response_ids_and_loss_mask_from_messages(
            response_messages,
            self.tokenizer,
            assistant_logprobs=None,
        )

        # Extract prompt ids
        prompt_ids = initial_input_ids

        # Calculate maximum response tokens allowed
        max_response_tokens = max_tokens + max_input_length - initial_prompt_length

        # Determine stop reason
        stop_reason = "complete"  # Default for trial completion
        if len(response_ids) > max_response_tokens:
            stop_reason = "length"

        # Truncate to maximum allowed length
        response_ids = response_ids[:max_response_tokens]
        loss_mask = loss_mask[:max_response_tokens]

        return (response_ids, reward, stop_reason, loss_mask, prompt_ids, None, trajectory_metrics)

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        """
        Generate trajectories for the input batch.

        Returns outputs in the same order as the input batch.
        Args:
            input_batch: GeneratorInput
        Returns:
            GeneratorOutput
        """
        prompts = input_batch["prompts"]
        env_extras = input_batch["env_extras"]
        trajectory_ids = input_batch["trajectory_ids"]
        batch_metadata = input_batch["batch_metadata"]
        max_tokens = self.generator_cfg.sampling_params.max_generate_length
        max_input_length = self.generator_cfg.max_input_length
        sampling_params = get_sampling_params_for_backend(
            self.generator_cfg.inference_engine.backend, self.generator_cfg.sampling_params
        )

        tasks = []

        for i in range(len(prompts)):
            tasks.append(
                self.minisweagent_agent_loop(
                    prompts[i],
                    env_extras[i],
                    max_tokens=max_tokens,
                    max_input_length=max_input_length,
                    sampling_params=sampling_params,
                    trajectory_id=trajectory_ids[i],
                    batch_metadata=batch_metadata,
                )
            )

        all_outputs = await asyncio.gather(*tasks)

        from loguru import logger

        responses = []
        rewards = []
        stop_reasons = []
        loss_masks = []
        prompt_token_ids = []
        trajectory_metrics = []

        for i, output in enumerate(all_outputs):
            response_ids, reward, stop_reason, loss_mask, prompt_ids, _, metrics = output
            if response_ids is None:
                instance_id = env_extras[i]["instance"].get("instance_id", "unknown")
                logger.warning(
                    f"Trajectory generation failed for instance {instance_id}; "
                    "using an empty placeholder response to preserve batch alignment."
                )
                response_ids = []
                reward = 0.0
                stop_reason = "error"
                loss_mask = []
                prompt_ids = self.tokenizer.apply_chat_template(
                    prompts[i], add_generation_prompt=False, return_dict=False, tokenize=True
                )
                metrics = metrics or {}

            responses.append(response_ids)
            rewards.append(reward)
            stop_reasons.append(stop_reason)
            loss_masks.append(loss_mask)
            prompt_token_ids.append(prompt_ids)
            trajectory_metrics.append(metrics)

        if not len(responses):
            raise ValueError(
                "Found no valid responses for this step. This means that generation failed for all trajectories, likely due to errors in environment setup."
            )
        rollout_metrics = get_rollout_metrics(responses, rewards, trajectory_metrics=trajectory_metrics)

        generator_output: GeneratorOutput = {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": responses,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "stop_reasons": stop_reasons,
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": None,
            "trajectory_metrics": trajectory_metrics,
        }

        return generator_output
