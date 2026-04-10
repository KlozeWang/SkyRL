import skyrl.backends.skyrl_train.inference_engines.vllm.utils as vllm_utils
from skyrl.backends.skyrl_train.inference_engines.vllm.utils import pop_openai_kwargs


def test_pop_openai_kwargs(monkeypatch):
    """
    Test pop_openai_kwargs with both primary and alias.
    Ensure OpenAI kwargs are popped, non-OpenAI kwargs are kept.
    """
    engine_kwargs = {
        "enable_auto_tools": 1,
        "tool_parser": "json",
        "reasoning_parser": "my_parser",
        "other": "keep",
    }
    monkeypatch.setattr(vllm_utils, "_get_vllm_openai_kwarg_names", lambda: ("enable_auto_tools", "tool_parser"))
    openai_kwargs = pop_openai_kwargs(engine_kwargs)

    assert openai_kwargs == {"enable_auto_tools": True, "tool_parser": "json", "reasoning_parser": "my_parser"}
    assert engine_kwargs == {"other": "keep"}

    engine_kwargs = {"enable_auto_tool_choice": 0, "tool_call_parser": "proto"}
    openai_kwargs = pop_openai_kwargs(engine_kwargs)

    assert openai_kwargs == {"enable_auto_tools": False, "tool_parser": "proto"}
    assert engine_kwargs == {}


def test_pop_openai_kwargs_legacy_vllm_names(monkeypatch):
    monkeypatch.setattr(
        vllm_utils, "_get_vllm_openai_kwarg_names", lambda: ("enable_auto_tool_choice", "tool_call_parser")
    )

    engine_kwargs = {"enable_auto_tools": 1, "tool_parser": "hermes"}
    openai_kwargs = pop_openai_kwargs(engine_kwargs)

    assert openai_kwargs == {"enable_auto_tool_choice": True, "tool_call_parser": "hermes"}
    assert engine_kwargs == {}
