"""OpenSquilla Fusion Reply provider integration."""

from __future__ import annotations

from typing import Any

__all__ = ["FusionMember", "FusionReplyProvider", "build_fusion_reply_provider"]


def __getattr__(name: str) -> Any:
    if name == "build_fusion_reply_provider":
        from opensquilla.fusion_reply.factory import build_fusion_reply_provider

        return build_fusion_reply_provider
    if name in {"FusionMember", "FusionReplyProvider"}:
        from opensquilla.fusion_reply.provider import FusionMember, FusionReplyProvider

        exports = {
            "FusionMember": FusionMember,
            "FusionReplyProvider": FusionReplyProvider,
        }
        return exports[name]
    raise AttributeError(name)
