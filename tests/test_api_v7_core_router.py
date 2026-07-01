from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from opensquilla.engine.steps import squilla_router as squilla_router_step
from opensquilla.gateway.config import SquillaRouterConfig
from opensquilla.squilla_router.api_v7_core import V7CoreApiStrategy


@pytest.fixture(autouse=True)
def reset_squilla_router_state() -> None:
    squilla_router_step._history_store.clear()
    squilla_router_step._strategy = None
    squilla_router_step._strategy_key = None
    yield
    squilla_router_step._history_store.clear()
    squilla_router_step._strategy = None
    squilla_router_step._strategy_key = None


def test_api_v7_core_strategy_posts_training_shape_and_maps_level() -> None:
    captured: dict[str, object] = {}
    label = {
        "schema": "model_router.v7-core",
        "intent": {"type": "data_analyze", "domain": "data_science", "language": "en"},
        "response": {"shape": "table", "interaction": "direct"},
        "target": {
            "family": "data",
            "level": 2,
            "signals": {"context": 0, "precision": 3, "structure": 1, "risk": 0},
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        payload = json.loads(request.content.decode("utf-8"))
        captured["payload"] = payload
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(label)}}]},
        )

    strategy = V7CoreApiStrategy(
        api_url="http://router.local/api/predict/service",
        model="Qwen3.5-9B",
        api_key="secret",
        transport=httpx.MockTransport(handler),
    )

    tier, confidence, source, extra = asyncio.run(
        strategy.classify(
            "Analyze this CSV.",
            ["c0", "c1", "c2", "c3"],
            history_user_texts=["Analyze this CSV."],
            attachments=[{"type": "application/pdf", "_material_estimated_tokens": 9000}],
            estimated_input_tokens=9000,
        )
    )

    assert captured["url"] == "http://router.local/api/predict/service/v1/chat/completions"
    assert captured["headers"]["authorization"] == "Bearer secret"
    payload = captured["payload"]
    assert payload["model"] == "Qwen3.5-9B"
    assert payload["stream"] is False
    assert payload["temperature"] == 0.0
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["messages"][0]["role"] == "system"
    assert "model_router.v7-core" in payload["messages"][0]["content"]
    assert "当前用户输入：\nAnalyze this CSV." in payload["messages"][1]["content"]

    assert tier == "c2"
    assert confidence == 1.0
    assert source == "api_v7_core"
    assert extra["route_class"] == "R2"
    assert extra["target_family"] == "data"
    assert extra["target_level"] == 2
    assert extra["target_effective_level"] == 3
    assert extra["level_adjustments"] == ["target.family=data:level>=3"]
    assert extra["v7_core"]["target"]["signals"]["context"] == 2
    assert extra["v7_core"]["target"]["signals"]["structure"] == 4


def test_api_v7_core_strategy_accepts_reasoning_content_fallback() -> None:
    label = {
        "schema": "model_router.v7-core",
        "intent": {"type": "design", "domain": "llm_infra", "language": "zh"},
        "response": {"shape": "structured_text", "interaction": "direct"},
        "target": {
            "family": "reasoning",
            "level": 4,
            "signals": {"context": 1, "precision": 4, "structure": 4, "risk": 0},
        },
    }

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "reasoning_content": json.dumps(label),
                        }
                    }
                ]
            },
        )

    strategy = V7CoreApiStrategy(
        api_url="http://router.local/v1/chat/completions",
        api_key="secret",
        transport=httpx.MockTransport(handler),
    )

    tier, _confidence, source, extra = asyncio.run(
        strategy.classify("设计一套路由 schema", ["c0", "c1", "c2", "c3"])
    )

    assert tier == "c3"
    assert source == "api_v7_core"
    assert extra["route_class"] == "R3"
    assert extra["v7_core"]["target"]["family"] == "reasoning"


def test_api_v7_core_strategy_fails_open_without_key() -> None:
    strategy = V7CoreApiStrategy(
        api_url="http://router.local/v1/chat/completions",
        api_key="",
        api_key_env="OPENSQUILLA_TEST_MISSING_ROUTER_KEY",
    )

    tier, confidence, source, extra = asyncio.run(
        strategy.classify("hello", ["c0", "c1", "c2", "c3"])
    )

    assert tier == "c1"
    assert confidence == 0.0
    assert source == "api_v7_core_unavailable"
    assert extra["route_class"] == "R1"
    assert "api key is not set" in extra["error"]


def test_squilla_router_strategy_factory_builds_api_v7_core_strategy() -> None:
    config = SquillaRouterConfig(
        strategy="api_v7_core",
        api_v7_core_url="http://router.local/v1/chat/completions",
        api_v7_core_model="Qwen3.5-9B",
        api_v7_core_api_key="secret",
    )

    strategy = squilla_router_step.preload_strategy(config)

    assert isinstance(strategy, V7CoreApiStrategy)
    assert strategy.source == "api_v7_core"
    assert strategy.api_url == "http://router.local/v1/chat/completions"
