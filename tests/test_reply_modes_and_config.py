from __future__ import annotations

import pytest

from opensquilla.gateway.config import GatewayConfig
from opensquilla.reply_modes import normalize_reply_mode


def test_reply_mode_aliases_normalize() -> None:
    assert normalize_reply_mode("fusion-reply") == "fusion"
    assert normalize_reply_mode("default", default="direct") == "direct"

    with pytest.raises(ValueError):
        normalize_reply_mode("unknown")


def test_gateway_config_accepts_fusion_reply_models() -> None:
    cfg = GatewayConfig.model_validate(
        {
            "reply": {"default_mode": "fusion"},
            "fusion_reply": {
                "enabled": True,
                "models": [
                    {"id": "a", "provider": "openrouter", "model": "model-a", "weight": 2.0},
                    {"id": "b", "provider": "openrouter", "model": "model-b"},
                ],
                "trace": {"log_dir": "D:/tmp/fusion-traces", "include_prompts": False},
            },
        }
    )

    assert cfg.reply.default_mode == "fusion"
    assert cfg.fusion_reply.enabled is True
    assert cfg.fusion_reply.max_rounds == 4
    assert cfg.fusion_reply.min_rounds == 2
    assert cfg.fusion_reply.adaptive_segments is True
    assert [m.id for m in cfg.fusion_reply.models] == ["a", "b"]
    assert cfg.fusion_reply.models[0].weight == 2.0
    assert cfg.fusion_reply.trace.enabled is True
    assert cfg.fusion_reply.trace.level == "full"
    assert cfg.fusion_reply.trace.log_dir == "D:/tmp/fusion-traces"
    assert cfg.fusion_reply.trace.include_prompts is False
