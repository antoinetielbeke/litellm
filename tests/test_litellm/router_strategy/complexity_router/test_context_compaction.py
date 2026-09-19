import asyncio
import json
from collections.abc import Mapping
from copy import deepcopy
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.litellm_core_utils.initialize_dynamic_callback_params import initialize_standard_callback_dynamic_params
from litellm.router_strategy.complexity_router.config import ContextCompactionConfig
from litellm.router_strategy.complexity_router.context_compaction import (
    CompactionState,
    _history,
    compact_to_fit,
    compaction_executor,
)
from litellm.router_utils.auto_router_model_naming import strategy_router_dependencies


@pytest.fixture(autouse=True)
def isolated_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    monkeypatch.setenv("LITELLM_LICENSE", "")


def make_router(*, enabled: bool = True, compactor_window: int = 8192, target_window: int = 512, pin_output: bool = True) -> litellm.Router:
    config: Final = {
        "tiers": {"SIMPLE": "small", "MEDIUM": "large", "COMPLEX": "large", "REASONING": "large"},
        "keyword_tier_rules": [{"keywords": ["answer"], "tier": "SIMPLE"}],
        **({"context_compaction": {"model": "large", "max_tokens": 512}} if enabled else {}),
        "enable_context_window_escalation": enabled,
        "max_tokens_from_tier_model": pin_output,
    }
    return litellm.Router(
        model_list=[
            {"model_name": "auto", "litellm_params": {"model": "auto_router/complexity_router", "complexity_router_config": config}},
            {"model_name": "small", "litellm_params": {"model": "anthropic/claude-haiku-4-5-20251001", "api_key": "test"}, "model_info": {"id": "pinned-small", "max_input_tokens": target_window, "max_output_tokens": 128}},
            {"model_name": "large", "litellm_params": {"model": "anthropic/claude-sonnet-5", "api_key": "test"}, "model_info": {"id": "native-large", "max_input_tokens": compactor_window, "max_output_tokens": 1024}},
        ],
        enable_pre_call_checks=True,
        num_retries=0,
        disable_cooldowns=True,
    )


def history() -> list[dict[str, object]]:
    return [
        {"role": "user", "content": "Project code MAPLE-47. Background detail. " * 150},
        {"role": "assistant", "content": "Recorded"},
        {"role": "user", "content": "Answer with the project code"},
    ]


def provider_reply(payload: Mapping[str, object], summary: str = "Project code MAPLE-47") -> httpx.Response:
    compact: Final = "compaction" in payload
    return httpx.Response(200, json={
        "id": "msg_test", "type": "message", "role": "assistant", "model": payload["model"],
        "content": [{"type": "compaction", "content": summary, "signature": "native-signature"}] if compact else [{"type": "text", "text": "MAPLE-47"}],
        "stop_reason": "compaction" if compact else "end_turn",
        "usage": {"input_tokens": 0, "output_tokens": 0, "iterations": [{"type": "compaction", "input_tokens": 1200, "output_tokens": 20}]} if compact else {"input_tokens": 60, "output_tokens": 8},
    })


@pytest.mark.asyncio
@pytest.mark.parametrize("messages_api", [False, True])
@pytest.mark.parametrize("target_window", [512, 1640])
async def test_router_compacts_on_native_model_and_answers_on_selected_deployment(messages_api: bool, target_window: int) -> None:
    router: Final = make_router(target_window=target_window)
    messages: Final = history()
    original: Final = deepcopy(messages)
    expected: Final = messages if messages_api else [{**message, "content": [{"type": "text", "text": message["content"]}]} for message in messages]

    def respond(request: httpx.Request) -> httpx.Response:
        payload: Final = json.loads(request.content)
        assert "_context_compaction_state" not in payload
        if "compaction" in payload:
            assert payload["model"] == "claude-sonnet-5"
            assert payload["compaction"] == {"type": "summarize"}
            assert payload["messages"] == expected[:-1]
            assert payload["max_tokens"] == 512
        else:
            assert payload["model"] == "claude-haiku-4-5-20251001"
            assert payload["messages"][-1] == expected[-1]
            assert "MAPLE-47" in str(payload["messages"][0])
            assert "signature" not in str(payload["messages"])
        return provider_reply(payload)

    with respx.mock(assert_all_called=False) as transport:
        route: Final = transport.post("https://api.anthropic.com/v1/messages").mock(side_effect=respond)
        call: Final = router.aanthropic_messages if messages_api else router.acompletion
        response: Final = await call(model="auto", messages=messages, max_tokens=64)
        assert route.call_count == 2
        assert response is not None
    assert messages == original


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["disabled", "no_history", "compactor_too_small", "summary_too_large", "retained_too_large", "not_native", "empty_summary"])
async def test_overflow_fails_without_sending_an_oversized_answer(failure: str) -> None:
    router: Final = make_router(enabled=failure != "disabled", compactor_window=800 if failure == "compactor_too_small" else 8192)
    messages: Final = history()[:1] if failure == "no_history" else [*history()[:-1], {"role": "user", "content": "Answer " * 1500}] if failure == "retained_too_large" else history()

    def respond(request: httpx.Request) -> httpx.Response:
        payload: Final = json.loads(request.content)
        assert "compaction" in payload
        return provider_reply(
            {"model": payload["model"]} if failure == "not_native" else payload,
            summary="" if failure == "empty_summary" else "oversized " * 1500,
        )

    with respx.mock(assert_all_called=False) as transport:
        route: Final = transport.post("https://api.anthropic.com/v1/messages").mock(side_effect=respond)
        with pytest.raises((litellm.BadRequestError, litellm.ContextWindowExceededError)):
            await router.acompletion(model="small" if failure == "disabled" else "auto", messages=messages, max_tokens=64)
        assert route.call_count == (1 if failure in ("summary_too_large", "not_native", "empty_summary") else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize(("messages_api", "output"), (
    (False, {"max_tokens": 64}),
    (False, {"max_tokens": 64, "max_completion_tokens": None}),
    (False, {"max_tokens": None}),
    (True, {"max_tokens": 64}),
    (True, {"max_tokens": 64, "max_completion_tokens": None}),
))
async def test_fitting_requests_do_not_call_compactor(messages_api: bool, output: Mapping[str, int | None]) -> None:
    router: Final = make_router(pin_output=False)
    with respx.mock(assert_all_called=False) as transport:
        route: Final = transport.post("https://api.anthropic.com/v1/messages").mock(return_value=provider_reply({"model": "claude-haiku-4-5-20251001"}))
        call: Final = router.aanthropic_messages if messages_api else router.acompletion
        await call(model="auto", messages=[{"role": "user", "content": "Answer hello"}], **output)
        assert route.call_count == 1
        assert "compaction" not in json.loads(route.calls[0].request.content)


@pytest.mark.asyncio
@pytest.mark.parametrize(("privacy", "settings", "global_privacy"), (
    (True, {"turn_off_message_logging": True}, False),
    (True, {"metadata": {"turn_off_message_logging": True}}, False),
    (True, {}, True),
    (True, {"metadata": {"headers": {"x-litellm-enable-message-redaction": "true"}}}, False),
    (False, {"turn_off_message_logging": False}, False),
    (False, {"metadata": {"headers": {"litellm-disable-message-redaction": "true"}}}, True),
))
async def test_summary_reused_on_retry_and_includes_system_and_tools_in_budget(
    monkeypatch: pytest.MonkeyPatch, privacy: bool, settings: Mapping[str, object], global_privacy: bool,
) -> None:
    router: Final = make_router()
    deployment: Final = router.get_deployment(model_id="pinned-small").model_dump()
    state: Final = CompactionState(config=ContextCompactionConfig(model="large", max_tokens=512))
    monkeypatch.setattr(litellm, "turn_off_message_logging", global_privacy)
    payload: Final = {
        "model": "small", "messages": history(), "max_tokens": 64, "_context_compaction_state": state, **settings,
    }
    calls: Final = asyncio.Queue[Mapping[str, object]]()

    async def execute(protocol: str, request: Mapping[str, object]) -> Mapping[str, object]:
        assert "turn_off_message_logging" not in request and "turn_off_message_logging" not in request["metadata"]
        assert initialize_standard_callback_dynamic_params().get("turn_off_message_logging", False) is privacy
        calls.put_nowait(request)
        return {"choices": [{"message": {"provider_specific_fields": {"compaction_blocks": [{"type": "compaction", "content": "MAPLE-47", "signature": "signed"}]}}}]}

    token: Final = compaction_executor.set(execute)
    try:
        first: Final = await compact_to_fit(router, deployment, payload, "chat")
        second: Final = await compact_to_fit(router, deployment, payload, "chat")
        assert first == second and calls.qsize() == 1
        with pytest.raises(litellm.BadRequestError, match="leave no room"):
            await compact_to_fit(router, deployment, {**payload, "max_tokens": 450}, "chat")
        with pytest.raises(litellm.BadRequestError, match="leave no room"):
            await compact_to_fit(router, deployment, {**payload, "system": "instructions " * 600}, "chat")
    finally:
        compaction_executor.reset(token)
    assert initialize_standard_callback_dynamic_params().get("turn_off_message_logging") is None


@pytest.mark.asyncio
async def test_compaction_timeout_is_replayed_without_cancelling_retry_or_spending_again() -> None:
    router: Final = make_router()
    deployment: Final = router.get_deployment(model_id="pinned-small").model_dump()
    state: Final = CompactionState(config=ContextCompactionConfig(model="large", timeout_seconds=0.01))
    payload: Final = {"model": "small", "messages": history(), "max_tokens": 64, "_context_compaction_state": state}
    calls: Final = asyncio.Queue[None]()
    stopped: Final = asyncio.Event()

    async def execute(protocol: str, request: Mapping[str, object]) -> Mapping[str, object]:
        calls.put_nowait(None)
        try:
            return await asyncio.Future[Mapping[str, object]]()
        finally:
            stopped.set()

    token: Final = compaction_executor.set(execute)
    try:
        for _attempt in range(2):
            with pytest.raises(asyncio.TimeoutError):
                await compact_to_fit(router, deployment, payload, "chat")
            assert calls.qsize() == 1 and stopped.is_set()
    finally:
        compaction_executor.reset(token)


def test_native_tool_result_tail_is_retained_and_compactor_is_an_authorized_dependency() -> None:
    messages: Final = [*history(), {"role": "assistant", "content": [{"type": "tool_use", "id": "call", "name": "lookup", "input": {}}]}, {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call", "content": "value"}]}]
    _, prefix, tail = _history(messages, "small")
    assert list(prefix) == messages[:2]
    assert list(tail) == messages[2:]
    dependencies: Final = strategy_router_dependencies({"model": "auto_router/complexity_router", "complexity_router_config": {"context_compaction": {"model": "large"}}})
    assert [(dependency.model_name, dependency.role) for dependency in dependencies] == [("large", "compactor")]
