"""Build Fusion Reply providers from GatewayConfig."""

from __future__ import annotations

import os
from typing import Any

from opensquilla.fusion_reply.provider import FusionMember, FusionReplyProvider
from opensquilla.fusion_reply.trace import FusionTraceSettings
from opensquilla.gateway.llm_runtime import (
    provider_base_url_env_name,
    resolve_llm_runtime_config,
)
from opensquilla.provider.model_catalog import ModelCatalog
from opensquilla.provider.registry import get_provider_spec
from opensquilla.provider.selector import ModelSelector, ProviderConfig, SelectorConfig


class FusionReplyConfigError(ValueError):
    """Raised when Fusion Reply cannot be built from configuration."""


def build_fusion_reply_provider(config: Any) -> FusionReplyProvider:
    """Construct a FusionReplyProvider using configured base model entries."""

    fusion_cfg = getattr(config, "fusion_reply", None)
    if fusion_cfg is None or not bool(getattr(fusion_cfg, "enabled", False)):
        raise FusionReplyConfigError("Fusion Reply is disabled. Set fusion_reply.enabled=true.")

    entries = list(getattr(fusion_cfg, "models", []) or [])
    if len(entries) < 2:
        raise FusionReplyConfigError(
            "Fusion Reply requires at least two entries in fusion_reply.models."
        )

    llm_runtime = resolve_llm_runtime_config(config)
    action_provider_id = _entry_text(fusion_cfg, "action_provider") or llm_runtime.provider
    action_model = _entry_text(fusion_cfg, "action_model") or llm_runtime.model
    action_entry = {
        "api_key": _entry_text(fusion_cfg, "action_api_key"),
        "api_key_env": _entry_text(fusion_cfg, "action_api_key_env"),
        "base_url": _entry_text(fusion_cfg, "action_base_url"),
        "proxy": _entry_text(fusion_cfg, "action_proxy"),
        "provider_routing": _entry_mapping(fusion_cfg, "action_provider_routing"),
    }
    action_provider_cfg = _provider_config_for_entry(
        action_entry,
        provider_id=action_provider_id,
        model=action_model,
        llm_runtime=llm_runtime,
    )
    action_provider = ModelSelector(SelectorConfig(primary=action_provider_cfg)).resolve()
    model_catalog = ModelCatalog()
    members: list[FusionMember] = []
    action_member_id = ""
    for index, entry in enumerate(entries):
        provider_id = _entry_text(entry, "provider") or llm_runtime.provider
        model = _entry_text(entry, "model")
        if not model:
            raise FusionReplyConfigError(
                f"fusion_reply.models[{index}].model is required."
            )
        member_id = _entry_text(entry, "id") or f"{provider_id}:{model}"
        provider_cfg = _provider_config_for_entry(
            entry,
            provider_id=provider_id,
            model=model,
            llm_runtime=llm_runtime,
        )
        provider = ModelSelector(SelectorConfig(primary=provider_cfg)).resolve()
        model_capabilities = model_catalog.get_capabilities(
            model,
            provider_name=provider_id,
            base_url=provider_cfg.base_url,
        )
        members.append(
            FusionMember(
                id=member_id,
                provider_id=provider_id,
                model=model,
                provider=provider,
                weight=_entry_float(entry, "weight", 1.0),
                model_capabilities=model_capabilities,
            )
        )
        if not action_member_id and provider_id == action_provider_id and model == action_model:
            action_member_id = member_id

    return FusionReplyProvider(
        members,
        action_provider=action_provider,
        action_member_id=action_member_id,
        action_model=action_model,
        architecture=_entry_text(fusion_cfg, "architecture") or "agent_loop_assist",
        assist_max_rounds=int(getattr(fusion_cfg, "assist_max_rounds", 1) or 1),
        max_rounds=int(getattr(fusion_cfg, "max_rounds", 4) or 4),
        min_rounds=int(getattr(fusion_cfg, "min_rounds", 2) or 2),
        step_max_tokens=int(getattr(fusion_cfg, "step_max_tokens", 1024) or 1024),
        judge_max_tokens=int(getattr(fusion_cfg, "judge_max_tokens", 512) or 512),
        temperature=_optional_float(getattr(fusion_cfg, "temperature", 0.7)),
        judge_temperature=_optional_float(getattr(fusion_cfg, "judge_temperature", 0.0)),
        adaptive_segments=bool(getattr(fusion_cfg, "adaptive_segments", True)),
        trace_settings=FusionTraceSettings.from_config(getattr(fusion_cfg, "trace", None)),
    )


def _provider_config_for_entry(
    entry: Any,
    *,
    provider_id: str,
    model: str,
    llm_runtime: Any,
) -> ProviderConfig:
    spec = get_provider_spec(provider_id)
    same_provider = provider_id == llm_runtime.provider

    explicit_api_key = _entry_text(entry, "api_key")
    api_key_env = _entry_text(entry, "api_key_env") or ("" if explicit_api_key else spec.env_key)
    env_api_key = os.environ.get(api_key_env, "") if api_key_env else ""
    api_key = explicit_api_key or env_api_key or (llm_runtime.api_key if same_provider else "")

    base_url_env = provider_base_url_env_name(provider_id)
    env_base_url = os.environ.get(base_url_env, "")
    base_url = (
        _entry_text(entry, "base_url")
        or env_base_url
        or (llm_runtime.base_url if same_provider else "")
        or spec.default_base_url
    )
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]

    proxy = _entry_text(entry, "proxy") or (llm_runtime.proxy if same_provider else "")
    provider_routing = _entry_mapping(entry, "provider_routing")
    if not provider_routing and same_provider:
        provider_routing = dict(llm_runtime.provider_routing)

    return ProviderConfig(
        provider=provider_id,
        model=model,
        api_key=api_key,
        base_url=base_url,
        proxy=proxy,
        provider_routing=provider_routing,
    )


def _entry_text(entry: Any, key: str) -> str:
    value = _entry_value(entry, key)
    if value is None:
        return ""
    return str(value).strip()


def _entry_float(entry: Any, key: str, default: float) -> float:
    value = _entry_value(entry, key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _entry_mapping(entry: Any, key: str) -> dict[str, str]:
    value = _entry_value(entry, key)
    if not isinstance(value, dict):
        return {}
    return {str(k): str(v) for k, v in value.items()}


def _entry_value(entry: Any, key: str) -> Any:
    if isinstance(entry, dict):
        return entry.get(key)
    return getattr(entry, key, None)
