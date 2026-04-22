#!/usr/bin/env python3
"""Run one Mini-SWE rollout from the training dataset for debugging."""

import argparse
import copy
import json
import os
import socket
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

import yaml
from datasets import load_dataset
from minisweagent.config import get_config_path
from minisweagent.models import get_model

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))

from examples.train.mini_swe_agent.mini_swe_generator import (  # noqa: E402
    DefaultAgentWithReminder,
    save_traj_compat,
)
from examples.train.mini_swe_agent.mini_swe_utils import (  # noqa: E402
    close_environment,
    evaluate_trajectory,
    get_sb_environment,
)
from examples.train.mini_swe_agent.teacher import (  # noqa: E402
    AskTeacherRequest,
    MiniSWETeacherConfig,
    TeacherClient,
    calculate_teacher_penalty,
    parse_askteacher_command,
)


def parse_args() -> argparse.Namespace:
    teacher_defaults = MiniSWETeacherConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Run a full Mini-SWE-Agent rollout for one row in the training parquet. "
            "This does not start SkyRL training; it expects OPENAI_BASE_URL to point at "
            "the rollout model endpoint, just like the training run."
        )
    )
    parser.add_argument(
        "--train-data",
        default=None,
        help="Path to train.parquet. Defaults to $DATA_DIR/train.parquet.",
    )
    parser.add_argument(
        "--data-dir",
        default=os.getenv("DATA_DIR", "/mnt/xujiawang/skyrl/examples/swe/swe_gym_subset"),
        help="Dataset directory used when --train-data is omitted.",
    )
    parser.add_argument("--instance-index", type=int, default=0, help="Training dataset row index to run.")
    parser.add_argument("--instance-id", default=None, help="Optional SWE instance_id. Overrides --instance-index.")
    parser.add_argument("--model-name", default=os.getenv("MODEL_NAME", "Qwen/Qwen3-8B"))
    parser.add_argument("--miniswe-config-path", default="examples/train/mini_swe_agent/swebench.yaml")
    parser.add_argument(
        "--traj-dir",
        default=os.getenv("MINISWE_TRAJ_DIR", "mini_swe_agent_debug_trajs"),
        help="Directory where the debug trajectory JSON is saved.",
    )
    parser.add_argument("--repetition-id", type=int, default=0)
    parser.add_argument("--skip-eval", action="store_true", help="Run the rollout but skip SWE eval.")
    parser.add_argument(
        "--force-teacher-first",
        action="store_true",
        help="Debug-only: prepend an instruction asking the learner to call askteacher on its first turn.",
    )
    parser.add_argument(
        "--no-print-turns",
        action="store_true",
        help="Do not print assistant turns, parsed commands, askteacher requests, and observations.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors in debug rollout logs.",
    )
    parser.add_argument(
        "--teacher-smoke-test",
        action="store_true",
        help="Call the teacher once outside the rollout to verify teacher API connectivity.",
    )
    parser.add_argument(
        "--skip-rollout-preflight",
        action="store_true",
        help="Skip checking that OPENAI_BASE_URL is reachable before starting the SWE container.",
    )
    parser.add_argument(
        "--teacher-smoke-prompt",
        default="Reply with one concise sentence confirming that the teacher endpoint is reachable.",
    )

    parser.add_argument("--max-generate-length", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--logprobs", type=int, default=1)

    parser.add_argument("--teacher-enabled", action="store_true")
    parser.add_argument("--teacher-base-url", default=teacher_defaults.base_url)
    parser.add_argument("--teacher-api-key", default=teacher_defaults.api_key)
    parser.add_argument("--teacher-api-key-env", default=teacher_defaults.api_key_env)
    parser.add_argument("--teacher-model", default=teacher_defaults.model)
    parser.add_argument("--teacher-max-turns", type=int, default=teacher_defaults.max_turns)
    parser.add_argument("--teacher-timeout-s", type=int, default=teacher_defaults.timeout_s)
    parser.add_argument("--teacher-tool-env-mode", default=teacher_defaults.tool_env_mode)
    parser.add_argument("--teacher-tool-timeout-s", type=int, default=teacher_defaults.tool_timeout_s)
    parser.add_argument("--teacher-max-completion-tokens", type=int, default=teacher_defaults.max_completion_tokens)
    parser.add_argument(
        "--teacher-output-token-penalty-coef",
        type=float,
        default=teacher_defaults.output_token_penalty_coef,
    )
    parser.add_argument(
        "--teacher-output-token-penalty-unit",
        type=int,
        default=teacher_defaults.output_token_penalty_unit,
    )
    parser.add_argument("--teacher-max-penalty", type=float, default=teacher_defaults.max_penalty)
    parser.add_argument("--no-teacher-history", action="store_true")
    parser.add_argument("--return-teacher-transcript-to-learner", action="store_true")
    parser.add_argument("--no-pass-reasoning-effort", action="store_true")
    return parser.parse_args()


class Ansi:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    STUDENT = "\033[36m"
    TEACHER = "\033[35m"
    ACTION = "\033[33m"
    ERROR = "\033[31m"


def colorize(text: str, color: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{color}{text}{Ansi.RESET}"


def print_debug_header(title: str, *, color: str, enabled: bool) -> None:
    print(colorize(f"\n===== {title} =====", f"{Ansi.BOLD}{color}", enabled), flush=True)


def print_debug_body(content: Any, *, color: str, enabled: bool) -> None:
    print(colorize(str(content), color, enabled), flush=True)


def preflight_rollout_endpoint() -> None:
    base_url = os.getenv("OPENAI_BASE_URL")
    if not base_url:
        raise RuntimeError(
            "OPENAI_BASE_URL is not set. The rollout model endpoint must be running before this demo starts."
        )

    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise RuntimeError(f"OPENAI_BASE_URL is not a valid HTTP URL: {base_url!r}")

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((parsed.hostname, port), timeout=5):
            pass
    except OSError as exc:
        raise RuntimeError(
            "Cannot connect to the rollout/student endpoint from OPENAI_BASE_URL="
            f"{base_url!r}. Start the SkyRL inference HTTP endpoint first, or point "
            "OPENAI_BASE_URL at an already-running OpenAI-compatible rollout server. "
            "In the full training run this endpoint is created by "
            "generator.inference_engine.enable_http_endpoint=true."
        ) from exc

    print(f"Rollout endpoint TCP preflight passed: {parsed.hostname}:{port}")


def load_training_row(train_data: Path, instance_index: int, instance_id: Optional[str]) -> Tuple[Dict[str, Any], int]:
    dataset = load_dataset("parquet", data_files=str(train_data), keep_in_memory=True)["train"]
    if instance_id is None:
        if instance_index < 0 or instance_index >= len(dataset):
            raise IndexError(f"--instance-index {instance_index} is outside dataset length {len(dataset)}")
        return dataset[instance_index], instance_index

    for idx in range(len(dataset)):
        row = dataset[idx]
        instance = row.get("instance") or {}
        if str(instance.get("instance_id", "")) == instance_id:
            return row, idx
    raise ValueError(f"Could not find instance_id={instance_id!r} in {train_data}")


def make_sampling_params(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "min_tokens": 1,
        "skip_special_tokens": True,
        "include_stop_str_in_output": True,
        "max_tokens": args.max_generate_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "logprobs": args.logprobs,
        "stop": None,
    }


def make_teacher_config(args: argparse.Namespace) -> MiniSWETeacherConfig:
    return MiniSWETeacherConfig(
        enabled=args.teacher_enabled,
        base_url=args.teacher_base_url,
        api_key=args.teacher_api_key,
        api_key_env=args.teacher_api_key_env,
        model=args.teacher_model,
        max_turns=args.teacher_max_turns,
        timeout_s=args.teacher_timeout_s,
        tool_env_mode=args.teacher_tool_env_mode,
        include_learner_history=not args.no_teacher_history,
        return_teacher_transcript_to_learner=args.return_teacher_transcript_to_learner,
        output_token_penalty_coef=args.teacher_output_token_penalty_coef,
        output_token_penalty_unit=args.teacher_output_token_penalty_unit,
        max_penalty=args.teacher_max_penalty,
        pass_reasoning_effort=not args.no_pass_reasoning_effort,
        max_completion_tokens=args.teacher_max_completion_tokens,
        tool_timeout_s=args.teacher_tool_timeout_s,
    )


def run_teacher_smoke_test(
    teacher_cfg: MiniSWETeacherConfig,
    instance: Dict[str, Any],
    prompt: str,
) -> None:
    if not teacher_cfg.enabled:
        print("Skipping teacher smoke test because --teacher-enabled was not set.")
        return

    def disabled_tool(command: str, timeout: Optional[int]) -> Dict[str, Any]:
        return {
            "returncode": 126,
            "output": f"Teacher smoke test did not execute tools. Requested command: {command}",
        }

    print("Running teacher smoke test outside the rollout...")
    result = TeacherClient(teacher_cfg).run(
        AskTeacherRequest(prompt=prompt, think_level="low"),
        instance=instance,
        learner_messages=[],
        execute_command=disabled_tool,
    )
    print(
        json.dumps(
            {
                "teacher_smoke_error": result.error,
                "teacher_smoke_answer": result.answer,
                "teacher_smoke_usage": result.usage.to_dict(),
            },
            indent=2,
        )
    )


class DebugAgentWithTurnLogging(DefaultAgentWithReminder):
    def __init__(self, *args, print_turns: bool = True, color: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.print_turns = print_turns
        self.color = color and os.getenv("NO_COLOR") is None
        self._debug_turn_idx = 0

    def get_observation(self, response: dict) -> dict:
        if self.print_turns:
            self._print_assistant_turn(response)
        output = super().get_observation(response)
        if self.print_turns:
            self._print_observation(output)
        return output

    def add_message(self, role: str, content: str, **kwargs):
        if self.print_turns and role in ("system", "user") and self._debug_turn_idx == 0:
            self._print_initial_prompt(role, content)

        parent = super()
        if hasattr(parent, "add_message"):
            return parent.add_message(role, content, **kwargs)
        return parent.add_messages({"role": role, "content": content, **kwargs})

    def add_messages(self, *messages: dict) -> list[dict]:
        if self.print_turns:
            for message in messages:
                self._print_added_message(message)
        return super().add_messages(*messages)

    def _execute_teacher_request(self, teacher_request) -> dict:
        record_idx = len(self.teacher_records)
        if self.print_turns:
            self._print_teacher_start(teacher_request)
        output = super()._execute_teacher_request(teacher_request)
        if self.print_turns:
            if len(self.teacher_records) > record_idx:
                self._print_teacher_record(self.teacher_records[-1])
            else:
                self._print_teacher_no_record(output)
        return output

    def _print_initial_prompt(self, role: str, content: Any) -> None:
        print_debug_header(f"STUDENT INITIAL {role.upper()} PROMPT", color=Ansi.STUDENT, enabled=self.color)
        print_debug_body(content, color=Ansi.STUDENT, enabled=self.color)

    def _print_added_message(self, message: dict) -> None:
        role = str(message.get("role", ""))
        content = message.get("content", "")
        if role in ("system", "user") and self._debug_turn_idx == 0:
            self._print_initial_prompt(role, content)
            return

        if role == "assistant":
            self._debug_turn_idx += 1
            print_debug_header(
                f"STUDENT TURN {self._debug_turn_idx}: ASSISTANT",
                color=Ansi.STUDENT,
                enabled=self.color,
            )
            print_debug_body(content, color=Ansi.STUDENT, enabled=self.color)
            for action in message.get("extra", {}).get("actions", []):
                self._print_action(str(action))
            return

        if role == "user" and self._debug_turn_idx > 0:
            print_debug_header(
                f"STUDENT TURN {self._debug_turn_idx}: OBSERVATION",
                color=Ansi.STUDENT,
                enabled=self.color,
            )
            print_debug_body(content, color=Ansi.STUDENT, enabled=self.color)
            return

        if role == "exit":
            print_debug_header("STUDENT EXIT", color=Ansi.STUDENT, enabled=self.color)
            print_debug_body(content, color=Ansi.STUDENT, enabled=self.color)

    def _print_assistant_turn(self, response: dict) -> None:
        self._debug_turn_idx += 1
        content = str(response.get("content", ""))
        action = ""
        try:
            action = self.parse_action(response).get("action", "")
        except Exception as exc:
            action = f"<parse_error: {type(exc).__name__}: {exc}>"

        print_debug_header(
            f"STUDENT TURN {self._debug_turn_idx}: ASSISTANT",
            color=Ansi.STUDENT,
            enabled=self.color,
        )
        print_debug_body(content, color=Ansi.STUDENT, enabled=self.color)
        if action:
            self._print_action(action)

    def _print_action(self, action: str) -> None:
        print(
            colorize(f"\n----- [STUDENT] parsed action -----\n{action}", Ansi.ACTION, self.color),
            flush=True,
        )
        try:
            teacher_request = parse_askteacher_command(action)
        except ValueError as exc:
            print(
                colorize(f"\n----- [TEACHER] askteacher parse error -----\n{exc}", Ansi.ERROR, self.color),
                flush=True,
            )
        else:
            if teacher_request is not None:
                print(
                    colorize(
                        "\n----- [TEACHER] askteacher request from student -----\n"
                        f"think_level={teacher_request.think_level}\n"
                        f"prompt={teacher_request.prompt}",
                        Ansi.TEACHER,
                        self.color,
                    ),
                    flush=True,
                )

    def _print_observation(self, output: dict) -> None:
        print_debug_header(
            f"STUDENT TURN {self._debug_turn_idx}: OBSERVATION",
            color=Ansi.STUDENT,
            enabled=self.color,
        )
        print_debug_body(f"returncode={output.get('returncode')}", color=Ansi.STUDENT, enabled=self.color)
        print_debug_body(output.get("output", ""), color=Ansi.STUDENT, enabled=self.color)

    def _print_teacher_start(self, teacher_request: AskTeacherRequest) -> None:
        print_debug_header("TEACHER ROLLOUT START", color=Ansi.TEACHER, enabled=self.color)
        print_debug_body(
            "student requested teacher\n"
            f"think_level={teacher_request.think_level}\n"
            f"prompt={teacher_request.prompt}\n"
            f"tool_env_mode={self.teacher_cfg.tool_env_mode}\n"
            f"teacher_turns_limit={self.teacher_cfg.max_turns}",
            color=Ansi.TEACHER,
            enabled=self.color,
        )

    def _print_teacher_record(self, record: Dict[str, Any]) -> None:
        usage = record.get("usage", {})
        print_debug_header("TEACHER ROLLOUT TRANSCRIPT", color=Ansi.TEACHER, enabled=self.color)
        print_debug_body(
            json.dumps(
                {
                    "error": record.get("error"),
                    "usage": usage,
                },
                indent=2,
                ensure_ascii=False,
            ),
            color=Ansi.TEACHER,
            enabled=self.color,
        )

        teacher_turn = 0
        for message in record.get("transcript", []):
            role = str(message.get("role", ""))
            if role == "assistant":
                teacher_turn += 1
                self._print_teacher_assistant_message(teacher_turn, message)
            elif role == "tool":
                self._print_teacher_tool_message(teacher_turn, message)
            else:
                print_debug_header(
                    f"TEACHER MESSAGE: {role.upper()}",
                    color=Ansi.TEACHER,
                    enabled=self.color,
                )
                print_debug_body(message, color=Ansi.TEACHER, enabled=self.color)

        print_debug_header("TEACHER HANDOFF TO STUDENT", color=Ansi.TEACHER, enabled=self.color)
        print_debug_body(record.get("answer", ""), color=Ansi.TEACHER, enabled=self.color)

    def _print_teacher_assistant_message(self, teacher_turn: int, message: Dict[str, Any]) -> None:
        print_debug_header(
            f"TEACHER TURN {teacher_turn}: ASSISTANT",
            color=Ansi.TEACHER,
            enabled=self.color,
        )
        content = message.get("content") or ""
        if content:
            print_debug_body(content, color=Ansi.TEACHER, enabled=self.color)

        for tool_call in message.get("tool_calls") or []:
            name, arguments = self._format_tool_call(tool_call)
            print_debug_body(
                f"tool_call={name}\narguments={arguments}",
                color=Ansi.TEACHER,
                enabled=self.color,
            )

    def _print_teacher_tool_message(self, teacher_turn: int, message: Dict[str, Any]) -> None:
        print_debug_header(
            f"TEACHER TURN {teacher_turn}: TOOL RESULT",
            color=Ansi.TEACHER,
            enabled=self.color,
        )
        content = str(message.get("content", ""))
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            print_debug_body(content, color=Ansi.TEACHER, enabled=self.color)
            return

        if isinstance(parsed, dict):
            returncode = parsed.get("returncode", parsed.get("error", ""))
            output = parsed.get("output", parsed)
            print_debug_body(f"returncode={returncode}", color=Ansi.TEACHER, enabled=self.color)
            print_debug_body(output, color=Ansi.TEACHER, enabled=self.color)
            return

        print_debug_body(parsed, color=Ansi.TEACHER, enabled=self.color)

    def _print_teacher_no_record(self, output: Dict[str, Any]) -> None:
        print_debug_header("TEACHER ROLLOUT FAILED BEFORE RECORD", color=Ansi.TEACHER, enabled=self.color)
        print_debug_body(output, color=Ansi.TEACHER, enabled=self.color)

    def _format_tool_call(self, tool_call: Any) -> Tuple[str, str]:
        if not isinstance(tool_call, dict):
            return type(tool_call).__name__, str(tool_call)

        function = tool_call.get("function", {})
        if isinstance(function, dict):
            return str(function.get("name", "")), str(function.get("arguments", ""))
        return str(function), str(tool_call)


def get_problem_statement(instance: Dict[str, Any], force_teacher_first: bool) -> str:
    problem_statement = str(instance["problem_statement"])
    if not force_teacher_first:
        return problem_statement

    return (
        "DEBUG INSTRUCTION: On your first turn, ask the teacher for a concise plan using exactly "
        "`askteacher --prompt \"What should I inspect first for this issue?\" --think_level low`. "
        "After receiving the teacher response, continue solving the task normally.\n\n"
        + problem_statement
    )


def main() -> None:
    args = parse_args()

    train_data = Path(args.train_data or Path(args.data_dir) / "train.parquet").expanduser()
    if not train_data.exists():
        raise FileNotFoundError(f"Training data not found: {train_data}")

    row, dataset_index = load_training_row(train_data, args.instance_index, args.instance_id)
    instance = row["instance"]
    data_source = row.get("data_source", "")
    instance_id = instance.get("instance_id", f"row_{dataset_index}")

    sweagent_config = yaml.safe_load(get_config_path(args.miniswe_config_path).read_text())
    sampling_params = make_sampling_params(args)
    model_config = sweagent_config.setdefault("model", {})
    model_config.setdefault("model_kwargs", {}).update(sampling_params)

    teacher_cfg = make_teacher_config(args)
    if args.teacher_smoke_test:
        run_teacher_smoke_test(teacher_cfg, instance, args.teacher_smoke_prompt)

    if not args.skip_rollout_preflight:
        preflight_rollout_endpoint()

    litellm_model_name = "openai/" + args.model_name
    model = get_model(litellm_model_name, model_config)

    env = None
    agent = None
    exit_status = "UnknownError"
    submission: Any = None
    error: Optional[str] = None
    eval_result: Optional[Dict[str, Any]] = None
    eval_error: Optional[str] = None

    try:
        print(
            json.dumps(
                {
                    "dataset_index": dataset_index,
                    "instance_id": instance_id,
                    "data_source": data_source,
                    "litellm_model_name": litellm_model_name,
                    "openai_base_url": os.getenv("OPENAI_BASE_URL"),
                    "teacher_enabled": teacher_cfg.enabled,
                    "teacher_base_url": teacher_cfg.base_url,
                    "teacher_model": teacher_cfg.model,
                    "teacher_tool_env_mode": teacher_cfg.tool_env_mode,
                    "force_teacher_first": args.force_teacher_first,
                },
                indent=2,
            )
        )

        env = get_sb_environment(copy.deepcopy(sweagent_config), instance, data_source)
        agent = DebugAgentWithTurnLogging(
            model,
            env,
            print_turns=not args.no_print_turns,
            color=not args.no_color,
            teacher_cfg=teacher_cfg,
            instance=instance,
            sweagent_config=sweagent_config,
            data_source=data_source,
            **sweagent_config.get("agent", {}),
        )
        run_result = agent.run(get_problem_statement(instance, args.force_teacher_first))
        if isinstance(run_result, tuple):
            exit_status, submission = run_result
        else:
            exit_status = run_result.get("exit_status", "")
            submission = run_result.get("submission", "")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        exit_status = type(exc).__name__
        submission = str(exc)
        print(traceback.format_exc(), file=sys.stderr)
    finally:
        close_environment(env)

    correctness_reward: Optional[float] = None
    if agent is not None and not args.skip_eval:
        eval_result = evaluate_trajectory(instance, str(submission or ""), copy.deepcopy(sweagent_config), data_source)
        correctness_reward = float(eval_result["resolved"])
        eval_error = eval_result.get("eval_error")
        if eval_error:
            error = eval_error

    teacher_usage = agent.teacher_usage.to_dict() if agent is not None else {}
    teacher_penalty = (
        calculate_teacher_penalty(agent.teacher_usage.output_tokens, teacher_cfg) if agent is not None else 0.0
    )
    reward = correctness_reward - teacher_penalty if correctness_reward is not None else None

    traj_dir = Path(args.traj_dir).expanduser()
    traj_dir.mkdir(parents=True, exist_ok=True)
    traj_path = traj_dir / f"{instance_id}_{args.repetition_id}.json"
    if agent is not None:
        save_traj_compat(
            agent,
            traj_path,
            exit_status=exit_status,
            result=eval_result if eval_result is not None else submission,
            extra_info={
                "debug_single_rollout": True,
                "dataset_index": dataset_index,
                "data_source": data_source,
                "submission": submission,
                "correctness_reward": correctness_reward,
                "penalized_reward": reward,
                "teacher": teacher_usage | {"penalty": teacher_penalty},
                "teacher_records": agent.teacher_records,
                "error": error,
            },
            reward=reward if reward is not None else 0.0,
            eval_error=eval_error,
        )
        agent.close_teacher_environment()

    summary = {
        "trajectory_path": str(traj_path),
        "exit_status": exit_status,
        "error": error,
        "correctness_reward": correctness_reward,
        "teacher_penalty": teacher_penalty,
        "penalized_reward": reward,
        "teacher_usage": teacher_usage,
        "teacher_records": agent.teacher_records if agent is not None else [],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if agent is not None and agent.teacher_usage.requests == 0:
        print(
            "No askteacher calls occurred in this rollout. "
            "Use --force-teacher-first to verify the teacher command path, or inspect the saved trajectory."
        )

    if error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
