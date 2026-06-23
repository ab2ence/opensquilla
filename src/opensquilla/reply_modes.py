"""Reply-mode normalization shared by CLI, WebUI RPC, and TurnRunner."""

from __future__ import annotations

VALID_REPLY_MODES = frozenset({"router", "direct", "fusion"})

_ALIASES = {
    "": "",
    "auto": "",
    "default": "",
    "normal": "router",
    "router": "router",
    "squilla_router": "router",
    "direct": "direct",
    "base": "direct",
    "llm": "direct",
    "fusion": "fusion",
    "fusion_reply": "fusion",
    "model_fusion": "fusion",
}


def normalize_reply_mode(value: object | None, *, default: str = "router") -> str:
    """Return a stable reply-mode id.

    Empty / ``default`` / ``auto`` resolve to ``default`` so per-turn callers can
    choose between explicit mode selection and the configured default.
    """

    normalized_default = _ALIASES.get(str(default or "router").strip().lower(), default)
    if not normalized_default:
        normalized_default = "router"
    if normalized_default not in VALID_REPLY_MODES:
        allowed = ", ".join(sorted(VALID_REPLY_MODES))
        raise ValueError(f"reply mode must be one of: {allowed}")

    raw = "" if value is None else str(value).strip().lower().replace("-", "_")
    mode = _ALIASES.get(raw)
    if mode is None:
        allowed = ", ".join(sorted(VALID_REPLY_MODES))
        raise ValueError(f"reply mode must be one of: {allowed}")
    return normalized_default if not mode else mode


__all__ = ["VALID_REPLY_MODES", "normalize_reply_mode"]
