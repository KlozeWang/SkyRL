import pytest

from examples.train.mini_swe_agent.teacher import (
    MiniSWETeacherConfig,
    calculate_teacher_penalty,
    parse_askteacher_command,
)
from skyrl.train.generators.utils import get_rollout_metrics


def test_parse_askteacher_command_accepts_valid_command():
    request = parse_askteacher_command('askteacher --prompt "where should I inspect?" --think_level high')

    assert request.prompt == "where should I inspect?"
    assert request.think_level == "high"


def test_parse_askteacher_command_rejects_extra_shell_command():
    with pytest.raises(ValueError):
        parse_askteacher_command('askteacher --prompt "help" --think_level low && cat secret.txt')


def test_parse_askteacher_command_ignores_normal_commands():
    assert parse_askteacher_command("grep -R foo .") is None


def test_teacher_penalty_scales_with_output_tokens():
    cfg = MiniSWETeacherConfig(output_token_penalty_coef=0.05, output_token_penalty_unit=1000)

    assert calculate_teacher_penalty(2000, cfg) == pytest.approx(0.1)


def test_rollout_metrics_aggregate_trajectory_metrics():
    metrics = get_rollout_metrics(
        responses=[[1, 2], [3]],
        rewards=[1.0, 0.5],
        trajectory_metrics=[
            {"teacher_output_tokens": 100, "teacher_penalty": 0.1},
            {"teacher_output_tokens": 300, "teacher_penalty": 0.3},
        ],
    )

    assert metrics["trajectory/teacher_output_tokens/mean"] == pytest.approx(200)
    assert metrics["trajectory/teacher_output_tokens/sum"] == pytest.approx(400)
    assert metrics["trajectory/teacher_penalty/max"] == pytest.approx(0.3)
