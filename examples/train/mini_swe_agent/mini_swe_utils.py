import os
from typing import TypedDict, Optional
import traceback
import uuid
import inspect

from typing import Dict, Any
from loguru import logger

from jinja2 import Template

from minisweagent.environments import Environment, get_environment


class MiniSWEEvaluationResult(TypedDict):
    instance_id: str
    resolved: bool
    eval_error: Optional[str]


_NESTED_PODMAN_RUN_ARGS = [
    "--privileged",
    "--security-opt",
    "label=disable",
    "--userns=host",
    "--pid=host",
    "--ipc=host",
    "--net=host",
    "--cgroupns=host",
]


def _extend_unique_args(base_args: list[str], extra_args: list[str]) -> list[str]:
    merged = list(base_args)
    for arg in extra_args:
        if arg not in merged:
            merged.append(arg)
    return merged


def _apply_podman_compat(env_config: dict) -> None:
    executable = os.path.basename(env_config.get("executable", "docker"))
    if executable != "podman":
        return

    compat_mode = env_config.pop("podman_compat_mode", "")
    extra_run_args = env_config.pop("podman_run_args", [])

    run_args = list(env_config.get("run_args", ["--rm"]))
    if compat_mode == "nested":
        run_args = _extend_unique_args(run_args, _NESTED_PODMAN_RUN_ARGS)
    if extra_run_args:
        run_args = _extend_unique_args(run_args, list(extra_run_args))
    env_config["run_args"] = run_args


def get_sb_environment(config: dict, instance: dict, data_source: str) -> Environment:
    env_config = config.setdefault("environment", {})
    env_config["environment_class"] = env_config.get("environment_class", "docker")
    _apply_podman_compat(env_config)
    image_name = get_docker_image_name(instance, data_source)
    if env_config["environment_class"] == "docker":
        env_config["image"] = image_name
    elif env_config["environment_class"] == "singularity":
        env_config["image"] = f"docker://{image_name}"
    env = get_environment(env_config)
    if startup_command := config.get("run", {}).get("env_startup_command"):
        startup_command = Template(startup_command).render(**instance)
        out = execute_env_command(env, startup_command)
        if out["returncode"] != 0:
            raise RuntimeError(f"Error executing startup command: {out}")
    return env


def execute_env_command(env: Environment, command: str, *, timeout: int | None = None) -> dict[str, Any]:
    """Handle minisweagent environment.execute API differences across versions."""
    execute_kwargs = {"timeout": timeout} if timeout is not None else {}
    command_param = inspect.signature(env.execute).parameters.get("command")
    if command_param is not None and command_param.annotation is dict:
        return env.execute({"command": command}, **execute_kwargs)

    try:
        return env.execute(command, **execute_kwargs)
    except AttributeError as e:
        if "'str' object has no attribute 'get'" not in str(e):
            raise
    except TypeError as e:
        if "get" not in str(e) and "mapping" not in str(e).lower():
            raise
    return env.execute({"command": command}, **execute_kwargs)


def close_environment(env: Environment | None) -> None:
    """Best-effort cleanup across Mini-SWE-Agent environment versions."""
    if env is None:
        return
    for method_name in ("close", "cleanup"):
        method = getattr(env, method_name, None)
        if method is None:
            continue
        try:
            method()
        except Exception as e:
            logger.debug(f"Error closing Mini-SWE environment with {method_name}: {e}")
        return


def get_docker_image_name(instance: dict, data_source: str) -> str:
    """Get the image name for a SWEBench/SWE-Gym instance."""
    image_name = instance.get("image_name", None)
    if image_name is None:
        iid = instance["instance_id"]
        if "swe-gym" in data_source.lower():
            id_docker_compatible = iid.replace("__", "_s_")  # to comply with docker image naming convention
            image_name = f"docker.io/xingyaoww/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
        elif "swe-bench" in data_source.lower():
            # Docker doesn't allow double underscore, so we replace them with a magic token
            id_docker_compatible = iid.replace("__", "_1776_")
            image_name = f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
        else:
            raise NotImplementedError(f"Data source: {data_source} is not supported")
    return image_name


def evaluate_trajectory(
    instance: Dict[str, Any], model_patch: str, sweagent_config: dict, data_source: str
) -> MiniSWEEvaluationResult:

    ret = MiniSWEEvaluationResult(instance_id=instance["instance_id"], resolved=False, eval_error=None)

    env = None
    try:
        env = get_sb_environment(sweagent_config, instance, data_source)
    except Exception as e:
        ret["eval_error"] = f"Env creation failed with {e}"
        logger.info(f"Starting environment failed with exception: {e}\n, {traceback.format_exc()}")
        return ret

    try:
        # apply git patch
        # NOTE (sumanthrh): This applies patch in-line, and the maximum patch size is limited by the OS limits for `ARG_MAX`.
        # In modern systems, this is typically ~ 1 MB, which is pretty generous.
        # For simplicity, we assume that large patches greater than `ARG_MAX` are meant to fail
        delimiter = f"PATCH_{uuid.uuid4().hex}"  # unlikely to collide with symbols in the patch
        command = f"git apply <<'{delimiter}'\n{model_patch}\n{delimiter}"

        obs = execute_env_command(env, command)

        if obs["returncode"] != 0:
            ret["eval_error"] = obs["output"]
        else:
            # run eval script in-line
            eval_script = instance["eval_script"]
            eval_cmd = f"bash <<'EOF'\n{eval_script}\nEOF"
            # add longer timeout for evaluation
            obs = execute_env_command(env, eval_cmd, timeout=3600)
            # use the return value
            ret["resolved"] = obs["returncode"] == 0
            # truncate to last 1000 characters for brevity
            if not ret["resolved"]:
                ret["eval_error"] = f"(truncated to last 1000 characters)\n{obs['output'][-1000:]}"
    finally:
        close_environment(env)
    return ret
