import argparse
import copy
import json
import os
import shlex
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional


VALID_THINK_LEVELS = {"low", "medium", "high"}
MAX_TOOL_OUTPUT_CHARS = 10000


@dataclass
class MiniSWETeacherConfig:
    enabled: bool = False
    base_url: str = ""
    api_key: Optional[str] = None
    api_key_env: str = "TEACHER_API_KEY"
    model: str = "gpt-5.2"
    max_turns: int = 8
    timeout_s: int = 600
    tool_env_mode: str = "shadow"
    include_learner_history: bool = True
    return_teacher_transcript_to_learner: bool = False
    output_token_penalty_coef: float = 0.0
    output_token_penalty_unit: int = 1000
    max_penalty: Optional[float] = None
    pass_reasoning_effort: bool = True
    max_completion_tokens: Optional[int] = None
    tool_timeout_s: Optional[int] = None


@dataclass
class AskTeacherRequest:
    prompt: str
    think_level: str


@dataclass
class TeacherUsage:
    requests: int = 0
    calls: int = 0
    output_tokens: int = 0
    tool_calls: int = 0

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


@dataclass
class TeacherSessionResult:
    answer: str
    usage: TeacherUsage = field(default_factory=TeacherUsage)
    transcript: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def parse_askteacher_command(command: str) -> Optional[AskTeacherRequest]:
    """Parse an askteacher command, returning None for non-teacher commands."""
    try:
        tokens = shlex.split(command)
    except ValueError as e:
        stripped = command.strip()
        if stripped.startswith("askteacher"):
            raise ValueError(f"Invalid askteacher command: {e}") from e
        return None

    if not tokens or tokens[0] != "askteacher":
        return None

    parser = _ArgumentParser(prog="askteacher", add_help=False)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--think_level", required=True, choices=sorted(VALID_THINK_LEVELS))
    args = parser.parse_args(tokens[1:])
    return AskTeacherRequest(prompt=args.prompt, think_level=args.think_level)


def calculate_teacher_penalty(output_tokens: int, cfg: MiniSWETeacherConfig) -> float:
    if output_tokens <= 0 or cfg.output_token_penalty_coef <= 0:
        return 0.0
    unit = max(cfg.output_token_penalty_unit, 1)
    penalty = cfg.output_token_penalty_coef * output_tokens / unit
    if cfg.max_penalty is not None:
        penalty = min(penalty, cfg.max_penalty)
    return float(penalty)


def build_teacher_observation(result: TeacherSessionResult, *, return_transcript: bool = False) -> str:
    if result.error:
        return (
            "<teacher_error>\n"
            f"{result.error}\n"
            "</teacher_error>\n"
            f"<teacher_output_tokens>{result.usage.output_tokens}</teacher_output_tokens>"
        )

    parts = [
        f'<teacher_answer output_tokens="{result.usage.output_tokens}" tool_calls="{result.usage.tool_calls}">',
        result.answer.strip(),
        "</teacher_answer>",
    ]
    if return_transcript:
        parts.extend(
            [
                "<teacher_transcript>",
                json.dumps(result.transcript, ensure_ascii=False),
                "</teacher_transcript>",
            ]
        )
    return "\n".join(parts)


class TeacherClient:
    def __init__(self, cfg: MiniSWETeacherConfig):
        if not cfg.base_url:
            raise ValueError("Teacher is enabled but generator.teacher.base_url is empty")
        api_key = cfg.api_key or os.getenv(cfg.api_key_env)
        if not api_key:
            raise ValueError(
                f"Teacher is enabled but neither generator.teacher.api_key nor ${cfg.api_key_env} is set"
            )

        try:
            from openai import OpenAI
        except ModuleNotFoundError as e:
            raise RuntimeError("Teacher API requires the `openai` Python package.") from e

        self.cfg = cfg
        self.client = OpenAI(api_key=api_key, base_url=cfg.base_url)

    def run(
        self,
        request: AskTeacherRequest,
        *,
        instance: Dict[str, Any],
        learner_messages: List[Dict[str, Any]],
        execute_command: Callable[[str, Optional[int]], Dict[str, Any]],
    ) -> TeacherSessionResult:
        messages = self._initial_messages(request, instance, learner_messages)
        transcript: List[Dict[str, Any]] = []
        usage = TeacherUsage()

        for _ in range(self.cfg.max_turns):
            response = self._create_completion(messages, request.think_level)
            usage.calls += 1
            usage.output_tokens += _completion_tokens(response)

            message = response.choices[0].message
            assistant_message = _message_to_dict(message)
            messages.append(assistant_message)
            transcript.append(copy.deepcopy(assistant_message))

            tool_calls = getattr(message, "tool_calls", None) or []
            if not tool_calls:
                return TeacherSessionResult(
                    answer=(getattr(message, "content", None) or "").strip(),
                    usage=usage,
                    transcript=transcript,
                )

            for tool_call in tool_calls:
                name = tool_call.function.name
                args = _json_args(tool_call.function.arguments)
                if name == "handoff_to_learner":
                    return TeacherSessionResult(
                        answer=str(args.get("answer", "")).strip(),
                        usage=usage,
                        transcript=transcript,
                    )
                if name != "run_bash":
                    tool_result = {"error": f"Unknown teacher tool: {name}"}
                else:
                    usage.tool_calls += 1
                    command = str(args.get("command", ""))
                    if not command:
                        tool_result = {"returncode": 2, "output": "Missing command"}
                    else:
                        tool_result = execute_command(command, self.cfg.tool_timeout_s)

                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(_truncate_tool_result(tool_result), ensure_ascii=False),
                }
                messages.append(tool_message)
                transcript.append(copy.deepcopy(tool_message))

        return TeacherSessionResult(
            answer="Teacher did not hand control back before max_turns.",
            usage=usage,
            transcript=transcript,
            error="teacher_max_turns_exceeded",
        )

    def _create_completion(self, messages: List[Dict[str, Any]], think_level: str):
        kwargs: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "tools": TEACHER_TOOLS,
            "tool_choice": "auto",
            "timeout": self.cfg.timeout_s,
        }
        if self.cfg.pass_reasoning_effort:
            kwargs["reasoning_effort"] = think_level
        if self.cfg.max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = self.cfg.max_completion_tokens
        return self.client.chat.completions.create(**kwargs)

    def _initial_messages(
        self,
        request: AskTeacherRequest,
        instance: Dict[str, Any],
        learner_messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        system = (
            "You are a teacher model helping a learner solve a programming task. "
            "You may inspect the repository with the run_bash tool. "
            "When you are ready to return control to the learner, call handoff_to_learner with concise advice. "
            "Do not claim that you changed the learner environment; your shell access may be isolated."
        )
        parts = [
            f"<instance_id>{instance.get('instance_id', '')}</instance_id>",
            "<problem_statement>",
            str(instance.get("problem_statement", "")),
            "</problem_statement>",
            f"<learner_question think_level=\"{request.think_level}\">",
            request.prompt,
            "</learner_question>",
        ]
        if self.cfg.include_learner_history:
            parts.extend(
                [
                    "<learner_history>",
                    json.dumps(_compact_learner_messages(learner_messages), ensure_ascii=False),
                    "</learner_history>",
                ]
            )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n".join(parts)},
        ]


TEACHER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": "Run one non-interactive bash command in the teacher shell environment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to run.",
                    }
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "handoff_to_learner",
            "description": "Return control to the learner with the teacher's answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Concise guidance to show to the learner.",
                    }
                },
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
    },
]


def _message_to_dict(message: Any) -> Dict[str, Any]:
    if hasattr(message, "model_dump"):
        dumped = message.model_dump(exclude_none=True)
    else:
        dumped = {
            "role": getattr(message, "role", "assistant"),
            "content": getattr(message, "content", None),
        }
        if getattr(message, "tool_calls", None):
            dumped["tool_calls"] = [
                tool_call.model_dump(exclude_none=True) if hasattr(tool_call, "model_dump") else tool_call
                for tool_call in message.tool_calls
            ]
    dumped.setdefault("role", "assistant")
    dumped.setdefault("content", getattr(message, "content", None) or "")
    return dumped


def _completion_tokens(response: Any) -> int:
    usage = getattr(response, "usage", None)
    if usage is not None:
        tokens = getattr(usage, "completion_tokens", None)
        if tokens is not None:
            return int(tokens)

    message = response.choices[0].message
    text = getattr(message, "content", None) or ""
    token_estimate = len(text) // 4
    for tool_call in getattr(message, "tool_calls", None) or []:
        token_estimate += len(getattr(tool_call.function, "arguments", "") or "") // 4
    return max(token_estimate, 0)


def _json_args(raw_args: str) -> Dict[str, Any]:
    if not raw_args:
        return {}
    try:
        parsed = json.loads(raw_args)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _truncate_tool_result(result: Dict[str, Any]) -> Dict[str, Any]:
    output = str(result.get("output", ""))
    if len(output) <= MAX_TOOL_OUTPUT_CHARS:
        return result

    truncated = dict(result)
    truncated["output"] = (
        output[:5000]
        + f"\n...[{len(output) - MAX_TOOL_OUTPUT_CHARS} characters truncated]...\n"
        + output[-5000:]
    )
    truncated["truncated"] = True
    return truncated


def _compact_learner_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    compact = []
    for message in messages:
        role = str(message.get("role", ""))
        content = str(message.get("content", ""))
        if len(content) > 6000:
            content = content[:3000] + "\n...[truncated]...\n" + content[-3000:]
        compact.append({"role": role, "content": content})
    return compact
