import os
import shlex
import signal
import subprocess
import time
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
        # Azure/Singularity nested Podman cannot mount a private /proc reliably,
        # so host PID namespace is the default. Cleanup kills inspected payload
        # PIDs directly when Podman/crun cannot reap them.
        if _env_flag("MINISWE_PODMAN_USE_HOST_PID", default=True):
            run_args = _extend_unique_args(run_args, ["--pid=host"])
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
    try:
        env = get_environment(env_config)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(_format_subprocess_error(e)) from e
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
    if _remove_container_synchronously(env):
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


def _remove_container_synchronously(env: Environment) -> bool:
    container_id = getattr(env, "container_id", None)
    config = getattr(env, "config", None)
    executable = getattr(config, "executable", None)
    if not container_id or not executable:
        return False

    cmd = [str(executable), "rm", "-f", str(container_id)]
    if _run_container_cleanup_command(cmd, timeout=90):
        _clear_container_id(env)
        return True

    logger.warning(f"Falling back to direct payload kill for Mini-SWE container {container_id}")
    _kill_container_payload(str(executable), str(container_id))
    if _run_container_cleanup_command(cmd, timeout=30):
        _clear_container_id(env)
        return True

    logger.warning(
        f"Unable to remove Mini-SWE container {container_id} after direct payload kill. "
        "Skipping Mini-SWE's async cleanup fallback to avoid accumulating stuck podman rm processes."
    )
    return True


def _run_container_cleanup_command(cmd: list[str], *, timeout: int) -> bool:
    try:
        result = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        output = e.output.decode("utf-8", errors="replace") if isinstance(e.output, bytes) else (e.output or "")
        logger.warning(f"Timed out running {shlex.join(cmd)}. Output:\n{output}")
        return False

    if result.returncode != 0:
        logger.warning(
            f"Failed to run {shlex.join(cmd)}. returncode={result.returncode} output:\n{result.stdout}"
        )
        return False

    return True


def _kill_container_payload(executable: str, container_id: str) -> None:
    _run_quiet([executable, "kill", "--signal", "KILL", container_id], timeout=15)

    pid = _container_pid(executable, container_id)
    if pid <= 0:
        return

    logger.warning(f"Killing Mini-SWE container payload pid={pid} for container {container_id}")
    _kill_process_group(pid)
    _kill_process(pid)
    time.sleep(1)


def _container_pid(executable: str, container_id: str) -> int:
    result = _run_quiet(
        [executable, "inspect", "--format", "{{.State.Pid}}", container_id],
        timeout=10,
    )
    if result is None or result.returncode != 0:
        return 0
    try:
        return int((result.stdout or "").strip())
    except ValueError:
        return 0


def _kill_process_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError as e:
        logger.warning(f"Permission denied killing process group {pid}: {e}")
    except Exception as e:
        logger.warning(f"Error killing process group {pid}: {type(e).__name__}: {e}")


def _kill_process(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError as e:
        logger.warning(f"Permission denied killing process {pid}: {e}")
    except Exception as e:
        logger.warning(f"Error killing process {pid}: {type(e).__name__}: {e}")


def _run_quiet(cmd: list[str], *, timeout: int) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        output = e.output.decode("utf-8", errors="replace") if isinstance(e.output, bytes) else (e.output or "")
        logger.warning(f"Timed out running {shlex.join(cmd)}. Output:\n{output}")
        return None


def _clear_container_id(env: Environment) -> None:
    try:
        setattr(env, "container_id", None)
    except Exception:
        pass


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _format_subprocess_error(e: subprocess.CalledProcessError) -> str:
    stdout = e.stdout.decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
    stderr = e.stderr.decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
    command = shlex.join(map(str, e.cmd)) if isinstance(e.cmd, (list, tuple)) else str(e.cmd)
    return (
        f"Command failed with returncode={e.returncode}: {command}\n"
        f"<stdout>\n{stdout}\n</stdout>\n"
        f"<stderr>\n{stderr}\n</stderr>"
    )


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
