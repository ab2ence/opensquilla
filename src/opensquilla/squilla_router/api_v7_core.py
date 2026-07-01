"""OpenAI-compatible API strategy for the model_router.v7-core SFT router."""
# ruff: noqa: E501

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx
import structlog

from opensquilla.env import trust_env as _trust_env
from opensquilla.router_tiers import (
    DEFAULT_TEXT_TIER,
    ROUTE_CLASS_TO_TIER,
    TEXT_TIERS,
    TIER_TO_ROUTE_CLASS,
    normalize_text_tier,
)

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class RouterModelOutput:
    """Parsed router-model output preserving fields outside the route label."""

    label: dict[str, Any]
    raw: dict[str, Any]
    envelope: dict[str, Any]

# The deployed SFT router was trained from router_model's §5 ChatML samples.
# Keep the wire prompt compatible with that training shape, then normalize the
# parsed label to the neutral ``target.level`` name before exposing metadata.
V7_CORE_SYSTEM_PROMPT = """你是 Model Router。你只判断当前请求需要什么模型能力，不回答用户问题，不规划工具调用，不选择具体模型名。目标是在满足任务需求的前提下选择最低足够模型能力。只输出合法 JSON。

输出 schema 固定为 model_router.v7-core：

{
  "schema": "model_router.v7-core",
  "intent": {
    "type": "chat|qa|rewrite|summarize|extract|classify|translate|brainstorm|creative_write|plan|compare|design|code_generate|debug|code_review|agent_execute|data_analyze|math_solve|document_understand|image_understand|audio_understand|safety_sensitive",
    "domain": "general|llm_infra|software_engineering|data_science|business|finance|legal|medical|education|marketing|science|math|creative|personal",
    "language": "zh|en|ja|ko|fr|de|es|mixed|other"
  },
  "response": {
    "shape": "plain_text|structured_text|json|table|code|mixed",
    "interaction": "direct|clarify_required|safe_completion"
  },
  "target": {
    "family": "general|reasoning|code|agentic|math|data|visual|document|audio|creative",
    "level": 1-4,
    "signals": {
      "context": 0-5,
      "precision": 0-5,
      "structure": 0-5,
      "risk": 0-5
    }
  }
}

target.family 选择这轮任务最容易导致普通模型失败的主瓶颈能力。
level 只允许 1-4，禁止输出 5（极端链路由后处理规则触发）：1=极简单，2=普通，3=中等偏难，4=专家级。
signals: context=上下文需求，precision=准确性需求，structure=结构化需求，risk=安全风险。

强制规则：
risk>=4 时 family 保持真实能力，安全链路由 Policy Engine 触发。
PDF/长文档/多页文件优先 document。
图片/截图/图表/OCR 优先 visual。
音频优先 audio。
代码生成/debug/review 优先 code。
数学计算/证明/算法推导优先 math。
数据分析/表格/统计/实验优先 data。
创意写作/营销/风格控制优先 creative。
多步执行/环境操作/工具编排/长程任务优先 agentic。
复杂设计/分析/规划/比较/架构优先 reasoning。
简单聊天/普通问答/简单改写优先 general。
信息不足且无法直接完成时 response.interaction=clarify_required。
需要安全完成时 response.interaction=safe_completion。

不要输出 tools、freshness、fallback、secondary、constraints、model_name、provider、cost、latency、evidence、explanation、answer。"""

_INTENT_TYPES = {
    "chat",
    "qa",
    "rewrite",
    "summarize",
    "extract",
    "classify",
    "translate",
    "brainstorm",
    "creative_write",
    "plan",
    "compare",
    "design",
    "code_generate",
    "debug",
    "code_review",
    "agent_execute",
    "data_analyze",
    "math_solve",
    "document_understand",
    "image_understand",
    "audio_understand",
    "safety_sensitive",
}
_DOMAINS = {
    "general",
    "llm_infra",
    "software_engineering",
    "data_science",
    "business",
    "finance",
    "legal",
    "medical",
    "education",
    "marketing",
    "science",
    "math",
    "creative",
    "personal",
}
_LANGUAGES = {"zh", "en", "ja", "ko", "fr", "de", "es", "mixed", "other"}
_SHAPES = {"plain_text", "structured_text", "json", "table", "code", "mixed"}
_INTERACTIONS = {"direct", "clarify_required", "safe_completion"}
_FAMILIES = {
    "general",
    "reasoning",
    "code",
    "agentic",
    "math",
    "data",
    "visual",
    "document",
    "audio",
    "creative",
}
_V7_LEVEL_TO_TEXT_TIER = {1: "c0", 2: "c1", 3: "c2", 4: "c3", 5: "c3"}
_MAX_HISTORY_USER_TURNS = 2
_MAX_MESSAGE_CHARS = 500
_MAX_SUMMARY_CHARS = 3000
_MAX_ANCHOR_CHARS = 300


def _api_url(url: str) -> str:
    stripped = url.strip().rstrip("/")
    if not stripped:
        return ""
    if stripped.endswith("/chat/completions"):
        return stripped
    if stripped.endswith("/v1"):
        return f"{stripped}/chat/completions"
    return f"{stripped}/v1/chat/completions"


def _coerce_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _coerce_enum(value: object, allowed: set[str], default: str) -> str:
    text = str(value or "").strip()
    return text if text in allowed else default


def _context_bucket(estimated_tokens: int | None) -> int | None:
    if estimated_tokens is None:
        return None
    if estimated_tokens >= 128_000:
        return 4
    if estimated_tokens >= 32_000:
        return 3
    if estimated_tokens >= 8_000:
        return 2
    if estimated_tokens >= 512:
        return 1
    return 0


def _structure_floor(shape: str) -> int:
    if shape == "json":
        return 4
    if shape in {"table", "code"}:
        return 4
    if shape in {"mixed", "structured_text"}:
        return 3
    return 0


def _extract_json_object(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
        decoder = json.JSONDecoder()
        start = text.find("{")
        while start >= 0:
            try:
                parsed, _end = decoder.raw_decode(text[start:])
                break
            except json.JSONDecodeError:
                start = text.find("{", start + 1)
        if parsed is None:
            raise
    if not isinstance(parsed, dict):
        raise ValueError("router response JSON must be an object")
    return parsed


def _looks_like_v7_core_label_fields(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    if value.get("schema") == "model_router.v7-core":
        return True
    return (
        isinstance(value.get("intent"), Mapping)
        and isinstance(value.get("response"), Mapping)
        and isinstance(value.get("target"), Mapping)
    )


def _looks_like_v7_core_output(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    if isinstance(value.get("label"), Mapping):
        return _looks_like_v7_core_label_fields(value["label"])
    return _looks_like_v7_core_label_fields(value)


def _looks_like_cap_v1_output(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    if value.get("schema") == "router.cap.v1":
        return True
    return (
        isinstance(value.get("step"), Mapping)
        and isinstance(value.get("requirement"), Mapping)
        and isinstance(value.get("response"), Mapping)
    )


def _cap_v1_to_v7_core_label(raw: Mapping[str, Any]) -> dict[str, Any]:
    step = raw.get("step") if isinstance(raw.get("step"), Mapping) else {}
    requirement = (
        raw.get("requirement") if isinstance(raw.get("requirement"), Mapping) else {}
    )
    response = raw.get("response") if isinstance(raw.get("response"), Mapping) else {}
    signals = (
        requirement.get("signals") if isinstance(requirement.get("signals"), Mapping) else {}
    )
    return {
        "schema": "model_router.v7-core",
        "intent": {
            "type": step.get("intent"),
            "domain": step.get("domain"),
            "language": step.get("language"),
        },
        "response": {
            "shape": response.get("shape"),
            "interaction": response.get("interaction"),
        },
        "target": {
            "family": requirement.get("capability", requirement.get("family")),
            "level": requirement.get("level", requirement.get("tier")),
            "signals": dict(signals),
        },
    }


def _router_output_from_object(raw: Mapping[str, Any]) -> RouterModelOutput | None:
    raw_dict = dict(raw)
    if isinstance(raw.get("label"), Mapping) and _looks_like_v7_core_label_fields(raw["label"]):
        return RouterModelOutput(
            label=dict(raw["label"]),
            raw=raw_dict,
            envelope={key: value for key, value in raw.items() if key != "label"},
        )
    if _looks_like_v7_core_label_fields(raw):
        label_keys = {"schema", "intent", "response", "target"}
        return RouterModelOutput(
            label=raw_dict,
            raw=raw_dict,
            envelope={key: value for key, value in raw.items() if key not in label_keys},
        )
    if _looks_like_cap_v1_output(raw):
        label_keys = {"schema", "step", "requirement", "response"}
        return RouterModelOutput(
            label=_cap_v1_to_v7_core_label(raw),
            raw=raw_dict,
            envelope={key: value for key, value in raw.items() if key not in label_keys},
        )
    return None


def _coerce_output_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("content", "text", "output_text", "reasoning_content"):
            nested = value.get(key)
            if isinstance(nested, str) and nested:
                return nested
        return ""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return _content_text(value)
    return ""


def _extract_router_model_output_from_response_body(
    body: Mapping[str, Any],
) -> RouterModelOutput:
    direct = _router_output_from_object(body)
    if direct is not None:
        return direct

    candidates: list[object] = []
    choices = body.get("choices")
    if isinstance(choices, Sequence) and not isinstance(choices, (str, bytes, bytearray)):
        for choice in choices:
            if not isinstance(choice, Mapping):
                continue
            message = choice.get("message")
            if isinstance(message, Mapping):
                candidates.extend(
                    [
                        message.get("content"),
                        message.get("reasoning_content"),
                        message,
                    ]
                )
            candidates.extend([choice.get("text"), choice.get("content")])

    for key in ("output", "result", "response", "data", "text", "content"):
        if key in body:
            candidates.append(body.get(key))

    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, Mapping):
            parsed_candidate = _router_output_from_object(candidate)
            if parsed_candidate is not None:
                return parsed_candidate
        text = _coerce_output_text(candidate)
        if not text:
            continue
        parsed = _extract_json_object(text)
        parsed_output = _router_output_from_object(parsed)
        if parsed_output is not None:
            return parsed_output

    raise ValueError("router response did not contain a supported router label")


def _attachment_input_type(attachment: Mapping[str, Any]) -> str | None:
    raw = str(
        attachment.get("type")
        or attachment.get("mime")
        or attachment.get("mime_type")
        or attachment.get("media_type")
        or ""
    ).lower()
    if raw.startswith("image/") or "image" in raw:
        return "image"
    if raw.startswith("audio/") or "audio" in raw:
        return "audio"
    return None


def _build_input_metadata(
    *,
    attachments: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    input_types = ["text"]
    attachment_count = 0
    for attachment in attachments or []:
        kind = _attachment_input_type(attachment)
        if kind is None:
            continue
        attachment_count += 1
        if kind not in input_types:
            input_types.append(kind)
    metadata: dict[str, Any] = {
        "has_attachment": attachment_count > 0,
        "input_types": input_types,
        "attachment_count": attachment_count,
    }
    return metadata


def _truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return f"{text[: limit - 24].rstrip()}\n[truncated for router]"


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = sum(
        1
        for ch in text
        if "一" <= ch <= "鿿" or "぀" <= ch <= "ヿ" or "가" <= ch <= "힯"
    )
    other = len(text) - cjk
    return cjk + math.ceil(other / 4)


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, Mapping):
                continue
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
                continue
            block_type = block.get("type")
            if block_type == "image_url":
                parts.append("[图片]")
            elif block_type == "input_audio":
                parts.append("[音频]")
            elif block_type == "tool_result":
                parts.append(_format_tool_result_segment(block))
        return "\n".join(part for part in parts if part)
    return ""


def _tool_segment_name(segment: Mapping[str, Any]) -> str | None:
    name = segment.get("name")
    if isinstance(name, str) and name:
        return name
    function = segment.get("function")
    if isinstance(function, Mapping):
        function_name = function.get("name")
        if isinstance(function_name, str) and function_name:
            return function_name
    return None


def _format_tool_result_segment(segment: Mapping[str, Any]) -> str:
    preview = segment.get("preview")
    if not isinstance(preview, str):
        result = segment.get("result", "")
        preview = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    if not preview and segment.get("is_error"):
        preview = "is_error=True"
    return _truncate(preview, _MAX_MESSAGE_CHARS)


def _tool_result_texts(message: Mapping[str, Any]) -> list[str]:
    texts: list[str] = []
    if message.get("role") == "tool":
        text = _content_text(message.get("content"))
        if text:
            texts.append(text)
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, Sequence) or isinstance(tool_calls, (str, bytes, bytearray)):
        return texts
    for segment in tool_calls:
        if isinstance(segment, Mapping) and segment.get("type") == "tool_result":
            text = _format_tool_result_segment(segment)
            if text:
                texts.append(text)
    return texts


def _message_tool_names(message: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, Sequence) or isinstance(tool_calls, (str, bytes, bytearray)):
        return names
    for segment in tool_calls:
        if not isinstance(segment, Mapping):
            continue
        if segment.get("type") != "tool_use" and not isinstance(segment.get("function"), Mapping):
            continue
        name = _tool_segment_name(segment)
        if name:
            names.append(name)
    return names


def _format_tool_names(names: Sequence[str]) -> str:
    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    return ", ".join(name if count == 1 else f"{name} ×{count}" for name, count in counts.items())


def _message_line(message: Mapping[str, Any]) -> str:
    role = str(message.get("role") or "unknown")
    text = _content_text(message.get("content"))
    names = _message_tool_names(message)
    if names:
        tools = f"[工具调用: {_format_tool_names(names)}]"
        text = f"{tools} {text}".strip()
    if not text and not names:
        return ""
    return f"{role}: {_truncate(text, _MAX_MESSAGE_CHARS)}"


def _build_history_summary(history_messages: Sequence[Mapping[str, Any]]) -> str:
    user_positions = [
        idx for idx, message in enumerate(history_messages) if message.get("role") == "user"
    ]
    kept = user_positions[-_MAX_HISTORY_USER_TURNS:]
    start = kept[0] if len(kept) == _MAX_HISTORY_USER_TURNS else 0
    lines: list[str] = []
    tool_run = 0
    last_tool_text = ""

    def flush_tool_run() -> None:
        nonlocal tool_run, last_tool_text
        if tool_run <= 0:
            return
        prefix = f"tool ×{tool_run}: " if tool_run > 1 else "tool: "
        lines.append(prefix + _truncate(last_tool_text, _MAX_MESSAGE_CHARS))
        tool_run = 0
        last_tool_text = ""

    for message in history_messages[start:]:
        if message.get("role") == "tool":
            tool_texts = _tool_result_texts(message)
            if tool_texts:
                tool_run += len(tool_texts)
                last_tool_text = tool_texts[-1]
            continue
        flush_tool_run()
        line = _message_line(message)
        if line:
            lines.append(line)
        for tool_text in _tool_result_texts(message):
            tool_run += 1
            last_tool_text = tool_text
    flush_tool_run()
    summary = "\n".join(lines)
    while lines and len(summary) > _MAX_SUMMARY_CHARS:
        lines.pop(0)
        summary = "\n".join(lines)
    return summary[:_MAX_SUMMARY_CHARS]


def _task_anchor(history_messages: Sequence[Mapping[str, Any]]) -> str:
    for message in history_messages:
        if message.get("role") == "user":
            return _content_text(message.get("content"))[:_MAX_ANCHOR_CHARS]
    return ""


def _collect_tool_activity(history_messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = 0
    last_turn: list[str] = []
    for message in history_messages:
        names = _message_tool_names(message)
        total += len(names)
        if names and message.get("role") == "assistant":
            last_turn = names
    return {
        "last_turn_tools": last_turn,
        "history_tool_call_count": total,
        "has_tool_history": total > 0,
    }


def _history_text(history_messages: Sequence[Mapping[str, Any]]) -> str:
    parts: list[str] = []
    for message in history_messages:
        if message.get("role") == "tool":
            parts.extend(_tool_result_texts(message))
            continue
        text = _content_text(message.get("content"))
        if text:
            parts.append(text)
        parts.extend(_tool_result_texts(message))
    return "\n".join(parts)


def _coerce_history_messages(
    history_messages: Sequence[Mapping[str, Any]] | None,
) -> list[Mapping[str, Any]]:
    if not history_messages:
        return []
    return [message for message in history_messages if isinstance(message, Mapping)]


def _conversation_summary(
    *,
    history_user_texts: Sequence[str] | None,
    prev_assistant_text: str | None,
    limit_chars: int,
) -> str:
    parts: list[str] = []
    history = [str(item).strip() for item in history_user_texts or [] if str(item).strip()]
    if history:
        parts.append("最近用户消息：")
        for idx, item in enumerate(history[-5:], start=1):
            parts.append(f"{idx}. {_truncate(item, 600)}")
    if prev_assistant_text:
        parts.append("上一轮助手回复摘要：")
        parts.append(_truncate(prev_assistant_text.strip(), 1200))
    if not parts:
        return ""
    return _truncate("\n".join(parts), limit_chars)


def _build_router_view(
    *,
    message: str,
    history_user_texts: Sequence[str] | None,
    history_messages: Sequence[Mapping[str, Any]] | None,
    history_estimated_tokens: int | None,
    prev_assistant_text: str | None,
    attachments: Sequence[Mapping[str, Any]] | None,
    estimated_input_tokens: int | None,
    context_max_chars: int,
) -> dict[str, Any]:
    history = _coerce_history_messages(history_messages)
    legacy_history = [str(item).strip() for item in history_user_texts or [] if str(item).strip()]
    if history:
        task_anchor = _task_anchor(history)
        summary = _build_history_summary(history)
        tool_activity = _collect_tool_activity(history)
        local_token_estimate = _estimate_tokens(
            "\n".join(part for part in (_history_text(history), message) if part)
        )
    else:
        task_anchor = legacy_history[0] if legacy_history and legacy_history[0] != message.strip() else ""
        summary = _conversation_summary(
            history_user_texts=legacy_history,
            prev_assistant_text=prev_assistant_text,
            limit_chars=context_max_chars,
        )
        tool_activity = {"last_turn_tools": [], "history_tool_call_count": 0, "has_tool_history": False}
        local_token_estimate = _estimate_tokens("\n".join([*legacy_history, prev_assistant_text or "", message]))
    input_metadata = _build_input_metadata(
        attachments=attachments,
    )
    estimated_total = max(
        estimated_input_tokens or 0,
        (history_estimated_tokens or 0) + (estimated_input_tokens or 0),
        local_token_estimate,
    )
    return {
        "current_user_message": message,
        "task_anchor": task_anchor,
        "conversation_summary": _truncate(summary, context_max_chars),
        "input_metadata": input_metadata,
        "estimated_input_tokens": estimated_total,
        "tool_activity": tool_activity,
    }


def _build_user_prompt(view: Mapping[str, Any]) -> str:
    return (
        "当前用户输入：\n"
        f"{view.get('current_user_message', '')}\n\n"
        "会话初始任务（首条用户消息，空则为单轮会话）：\n"
        f"{view.get('task_anchor', '')}\n\n"
        "对话上下文摘要：\n"
        f"{view.get('conversation_summary', '')}\n\n"
        "输入元信息：\n"
        f"{json.dumps(view.get('input_metadata', {}), ensure_ascii=False)}\n\n"
        "估算输入 token 数：\n"
        f"{view.get('estimated_input_tokens', 0)}\n\n"
        "只输出 JSON。"
    )


def _normalize_label(
    raw: Mapping[str, Any],
    *,
    estimated_input_tokens: int | None,
) -> dict[str, Any]:
    if isinstance(raw.get("label"), Mapping):
        raw = raw["label"]

    intent_raw = raw.get("intent") if isinstance(raw.get("intent"), Mapping) else {}
    response_raw = raw.get("response") if isinstance(raw.get("response"), Mapping) else {}
    target_raw = raw.get("target") if isinstance(raw.get("target"), Mapping) else {}
    signals_raw = (
        target_raw.get("signals") if isinstance(target_raw.get("signals"), Mapping) else {}
    )

    shape = _coerce_enum(response_raw.get("shape"), _SHAPES, "plain_text")
    context = _coerce_int(signals_raw.get("context"), default=0, minimum=0, maximum=5)
    bucket = _context_bucket(estimated_input_tokens)
    if bucket is not None and context != 5:
        context = bucket
    structure = _coerce_int(signals_raw.get("structure"), default=1, minimum=0, maximum=5)
    structure = max(structure, _structure_floor(shape))

    level = _coerce_int(
        target_raw.get("level", target_raw.get("tier")),
        default=2,
        minimum=1,
        maximum=5,
    )

    return {
        "schema": "model_router.v7-core",
        "intent": {
            "type": _coerce_enum(intent_raw.get("type"), _INTENT_TYPES, "qa"),
            "domain": _coerce_enum(intent_raw.get("domain"), _DOMAINS, "general"),
            "language": _coerce_enum(intent_raw.get("language"), _LANGUAGES, "other"),
        },
        "response": {
            "shape": shape,
            "interaction": _coerce_enum(
                response_raw.get("interaction"),
                _INTERACTIONS,
                "direct",
            ),
        },
        "target": {
            "family": _coerce_enum(target_raw.get("family"), _FAMILIES, "general"),
            "level": level,
            "signals": {
                "context": context,
                "precision": _coerce_int(
                    signals_raw.get("precision"),
                    default=2,
                    minimum=0,
                    maximum=5,
                ),
                "structure": structure,
                "risk": _coerce_int(
                    signals_raw.get("risk"),
                    default=0,
                    minimum=0,
                    maximum=5,
                ),
            },
        },
    }


def _find_valid_tier(start_tier: str, valid_tiers: Sequence[str]) -> str:
    normalized_valid = [normalize_text_tier(tier) or str(tier) for tier in valid_tiers]
    if not normalized_valid:
        return DEFAULT_TEXT_TIER
    start = normalize_text_tier(start_tier) or DEFAULT_TEXT_TIER
    start_idx = TEXT_TIERS.index(start) if start in TEXT_TIERS else 1
    for idx in range(start_idx, len(TEXT_TIERS)):
        if TEXT_TIERS[idx] in normalized_valid:
            return TEXT_TIERS[idx]
    for tier in TEXT_TIERS:
        if tier in normalized_valid:
            return tier
    return normalized_valid[0]


def _label_section(label: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = label.get(key)
    return value if isinstance(value, Mapping) else {}


def _label_signals(label: Mapping[str, Any]) -> Mapping[str, Any]:
    target = _label_section(label, "target")
    signals = target.get("signals")
    return signals if isinstance(signals, Mapping) else {}


def _signal_int(signals: Mapping[str, Any], key: str) -> int:
    return _coerce_int(signals.get(key), default=0, minimum=0, maximum=5)


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    if isinstance(value, int):
        return value != 0
    return False


def _router_output_section(output: RouterModelOutput | None, key: str) -> Mapping[str, Any]:
    if output is None:
        return {}
    value = output.envelope.get(key, output.raw.get(key))
    return value if isinstance(value, Mapping) else {}


def _router_output_controls(output: RouterModelOutput | None) -> Mapping[str, Any]:
    return _router_output_section(output, "controls")


def _router_output_quality(output: RouterModelOutput | None) -> Mapping[str, Any]:
    return _router_output_section(output, "quality")


def _router_output_evidence(output: RouterModelOutput | None) -> Mapping[str, Any]:
    return _router_output_section(output, "evidence")


def _router_output_schema(output: RouterModelOutput | None) -> str:
    if output is None:
        return ""
    return str(output.raw.get("schema") or output.label.get("schema") or "")


def _has_tool_history(router_view: Mapping[str, Any]) -> bool:
    tool_activity = router_view.get("tool_activity")
    if not isinstance(tool_activity, Mapping):
        return False
    return bool(tool_activity.get("has_tool_history")) or bool(
        _coerce_int(
            tool_activity.get("history_tool_call_count"),
            default=0,
            minimum=0,
            maximum=1_000_000,
        )
    )


def _effective_level_for_label(
    label: Mapping[str, Any],
    *,
    router_view: Mapping[str, Any],
    router_output: RouterModelOutput | None = None,
) -> tuple[int, list[str]]:
    """Derive the level used for routing from the full v7-core label.

    The model's ``target.level`` remains the base decision. The other structured
    fields only apply conservative floors when they reveal a capability need
    that would make a low route brittle.
    """

    intent = _label_section(label, "intent")
    response = _label_section(label, "response")
    target = _label_section(label, "target")
    signals = _label_signals(label)

    base_level = _coerce_int(target.get("level"), default=2, minimum=1, maximum=5)
    effective_level = base_level
    adjustments: list[str] = []

    def floor(min_level: int, reason: str) -> None:
        nonlocal effective_level
        min_level = max(1, min(5, min_level))
        if effective_level < min_level:
            effective_level = min_level
            adjustments.append(f"{reason}:level>={min_level}")

    intent_type = str(intent.get("type") or "")
    domain = str(intent.get("domain") or "")
    family = str(target.get("family") or "")
    shape = str(response.get("shape") or "")
    interaction = str(response.get("interaction") or "")
    context = _signal_int(signals, "context")
    precision = _signal_int(signals, "precision")
    structure = _signal_int(signals, "structure")
    risk = _signal_int(signals, "risk")
    has_tool_history = _has_tool_history(router_view)
    substantial_context = context >= 2
    high_precision = precision >= 4
    elevated_risk = risk >= 2
    high_structure = structure >= 4
    simple_low_risk = (
        context <= 1
        and precision <= 3
        and risk <= 1
        and not has_tool_history
    )
    quality = _router_output_quality(router_output)
    evidence = _router_output_evidence(router_output)
    negative_evidence = evidence.get("negative")
    has_negative_evidence = isinstance(negative_evidence, Sequence) and not isinstance(
        negative_evidence,
        (str, bytes, bytearray),
    ) and bool(negative_evidence)

    if interaction == "safe_completion":
        floor(2, "response.interaction=safe_completion")
    if interaction == "clarify_required" and structure >= 3:
        floor(2, "response.interaction=clarify_required")
    if intent_type == "safety_sensitive" or risk >= 4:
        floor(3, "safety_or_risk")

    if family == "agentic" or intent_type == "agent_execute":
        floor(3, "target.family=agentic")
    if family in {"document", "visual", "audio"}:
        floor(3, f"target.family={family}")
    if intent_type in {"document_understand", "image_understand", "audio_understand"}:
        floor(3, f"intent.type={intent_type}")

    if intent_type == "code_review":
        floor(3, f"intent.type={intent_type}")
    elif intent_type in {"debug", "code_generate"}:
        if high_precision or substantial_context or elevated_risk or has_tool_history:
            floor(3, f"intent.type={intent_type}")
    elif family == "code" and (high_precision or substantial_context or has_tool_history):
        floor(3, "target.family=code")

    if (intent_type == "math_solve" or family == "math") and (
        high_precision or substantial_context or elevated_risk or base_level >= 3
    ):
        floor(3, "target.family=math")
    if (intent_type == "data_analyze" or family == "data") and (
        high_precision
        or substantial_context
        or elevated_risk
        or has_tool_history
        or (intent_type == "data_analyze" and high_structure and not simple_low_risk)
    ):
        floor(3, "target.family=data")

    if intent_type in {"plan", "compare", "design"} and (
        family in {"reasoning", "agentic"} or precision >= 4 or structure >= 4
    ):
        floor(3, f"intent.type={intent_type}")
    elif family == "reasoning" and (precision >= 4 or structure >= 4):
        floor(3, "target.family=reasoning")

    if shape in {"json", "table", "code"} and structure >= 4:
        if simple_low_risk and base_level <= 2:
            floor(2, f"response.shape={shape}")
        else:
            floor(3, f"response.shape={shape}")
    if structure >= 5 and precision >= 4:
        floor(4, "signals.structure>=5+precision>=4")
    if context >= 3:
        floor(3, "signals.context>=3")
    if context >= 4:
        floor(4, "signals.context>=4")
    if precision >= 5 and (
        context >= 3 or risk >= 3 or family in {"math", "reasoning", "agentic"}
    ):
        floor(4, "signals.precision>=5")
    if domain in {"legal", "medical", "finance"} and (precision >= 4 or risk >= 3):
        floor(3, f"intent.domain={domain}")
    if has_tool_history and family in {"agentic", "code"}:
        floor(3, "router_view.tool_history")
    if _coerce_bool(quality.get("ambiguous")):
        floor(2, "quality.ambiguous")
    if has_negative_evidence:
        floor(2, "evidence.negative")
    if _coerce_bool(quality.get("needs_human_review")):
        floor(4, "quality.needs_human_review")

    return effective_level, adjustments


def _thinking_mode_for_tier(tier: str) -> str:
    return {"c0": "T0", "c1": "T1", "c2": "T2", "c3": "T3"}.get(tier, "T1")


def _thinking_mode_for_label(
    label: Mapping[str, Any],
    tier: str,
    *,
    router_output: RouterModelOutput | None = None,
) -> str:
    controls = _router_output_controls(router_output)
    budget = str(controls.get("reasoning_budget") or "").strip().lower()
    if budget in {"off", "none", "minimal"}:
        return "T0"
    if budget == "low":
        return "T1"
    if budget in {"medium", "normal"}:
        return "T2"
    if budget in {"high", "xhigh", "extra_high", "extra-high"}:
        return "T3"
    signals = _label_signals(label)
    if _signal_int(signals, "precision") >= 5 and _signal_int(signals, "context") >= 3:
        return "T3"
    return _thinking_mode_for_tier(tier)


def _prompt_policy_for_label(
    label: Mapping[str, Any],
    tier: str,
    *,
    router_output: RouterModelOutput | None = None,
) -> str:
    response = label.get("response") if isinstance(label.get("response"), Mapping) else {}
    interaction = response.get("interaction")
    if interaction in {"clarify_required", "safe_completion"}:
        return "P1"
    quality = _router_output_quality(router_output)
    evidence = _router_output_evidence(router_output)
    controls = _router_output_controls(router_output)
    if _coerce_bool(quality.get("ambiguous")) or _coerce_bool(
        quality.get("needs_human_review")
    ):
        return "P1"
    negative = evidence.get("negative")
    if isinstance(negative, Sequence) and not isinstance(negative, (str, bytes, bytearray)) and negative:
        return "P1"
    output_budget = str(controls.get("output_budget") or "").strip().lower()
    if output_budget in {"long", "verbose", "detailed"}:
        return "P1"
    signals = _label_signals(label)
    shape = str(response.get("shape") or "")
    if (
        _signal_int(signals, "risk") >= 4
        or _signal_int(signals, "structure") >= 4
        or (shape in {"json", "table", "code"} and _signal_int(signals, "precision") >= 3)
    ):
        return "P1"
    if tier in {"c2", "c3"}:
        return "P1"
    return "P0"


class V7CoreApiStrategy:
    """Call a deployed v7-core router over an OpenAI-compatible API."""

    requires_history = True
    source = "api_v7_core"

    def __init__(
        self,
        *,
        api_url: str | None,
        model: str = "Qwen3.5-9B",
        api_key: str | None = None,
        api_key_env: str = "OPENSQUILLA_SQUILLA_ROUTER_API_V7_CORE_API_KEY",
        timeout_seconds: float = 5.0,
        max_tokens: int = 512,
        temperature: float = 0.0,
        disable_thinking: bool = True,
        confidence: float = 1.0,
        context_max_chars: int = 4000,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_url = _api_url(api_url or "")
        self.model = model
        self.api_key = api_key or ""
        self.api_key_env = api_key_env
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.disable_thinking = disable_thinking
        self.confidence = confidence
        self.context_max_chars = context_max_chars
        self._transport = transport

    def _resolved_api_key(self) -> str:
        return self.api_key or os.environ.get(self.api_key_env, "")

    def _unavailable_classify(
        self,
        valid_tiers: Sequence[str],
        *,
        error: str,
    ) -> tuple[str, float, str, dict[str, Any]]:
        tier = _find_valid_tier(DEFAULT_TEXT_TIER, valid_tiers)
        route_class = TIER_TO_ROUTE_CLASS.get(tier, "R1")
        return (
            tier,
            0.0,
            "api_v7_core_unavailable",
            {
                "route_class": route_class,
                "top1_label": route_class,
                "thinking_mode": _thinking_mode_for_tier(tier),
                "prompt_policy": "P1",
                "model_version": self.model,
                "error": error,
            },
        )

    async def classify(
        self,
        message: str,
        valid_tiers: list[str],
        routing_history: list[dict] | None = None,
        prev_assistant_text: str | None = None,
        prev_assistant_usage: dict | None = None,
        history_user_texts: list[str] | None = None,
        history_messages: list[dict] | None = None,
        history_estimated_tokens: int | None = None,
        flags_text_override: str | None = None,
        attachments: list[dict] | None = None,
        estimated_input_tokens: int | None = None,
        surface_kind: str | None = None,
    ) -> tuple[str, float, str, dict]:
        del routing_history, prev_assistant_usage, flags_text_override, surface_kind

        api_key = self._resolved_api_key()
        if not self.api_url:
            return self._unavailable_classify(valid_tiers, error="api_v7_core_url is not set")
        if not api_key:
            return self._unavailable_classify(valid_tiers, error="api_v7_core api key is not set")

        router_view = _build_router_view(
            message=message,
            history_user_texts=history_user_texts,
            history_messages=history_messages,
            history_estimated_tokens=history_estimated_tokens,
            prev_assistant_text=prev_assistant_text,
            attachments=attachments,
            estimated_input_tokens=estimated_input_tokens,
            context_max_chars=self.context_max_chars,
        )
        user_prompt = _build_user_prompt(router_view)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": V7_CORE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
                trust_env=_trust_env(),
                transport=self._transport,
            ) as client:
                response = await client.post(
                    self.api_url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
            response.raise_for_status()
            body = response.json()
            router_output = _extract_router_model_output_from_response_body(body)
            label = _normalize_label(
                router_output.label,
                estimated_input_tokens=estimated_input_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - router must fail open
            log.warning("api_v7_core.classify_failed", error=str(exc))
            return self._unavailable_classify(valid_tiers, error=str(exc))

        raw_level = int(label["target"]["level"])
        effective_level, level_adjustments = _effective_level_for_label(
            label,
            router_view=router_view,
            router_output=router_output,
        )
        tier = _find_valid_tier(
            _V7_LEVEL_TO_TEXT_TIER.get(effective_level, DEFAULT_TEXT_TIER),
            valid_tiers,
        )
        route_class = TIER_TO_ROUTE_CLASS.get(tier) or next(
            (key for key, value in ROUTE_CLASS_TO_TIER.items() if value == tier),
            "R1",
        )
        extra: dict[str, Any] = {
            "route_class": route_class,
            "top1_label": route_class,
            "thinking_mode": _thinking_mode_for_label(
                label,
                tier,
                router_output=router_output,
            ),
            "prompt_policy": _prompt_policy_for_label(
                label,
                tier,
                router_output=router_output,
            ),
            "model_version": self.model,
            "v7_core": label,
            "router_model_output_schema": _router_output_schema(router_output),
            "router_raw_output": router_output.raw,
            "router_output_envelope": router_output.envelope,
            "router_quality": dict(_router_output_quality(router_output)),
            "router_evidence": dict(_router_output_evidence(router_output)),
            "router_controls": dict(_router_output_controls(router_output)),
            "router_view": router_view,
            "target_family": label["target"]["family"],
            "target_level": raw_level,
            "target_effective_level": effective_level,
            "level_adjustments": level_adjustments,
            "target_signals": dict(label["target"]["signals"]),
            "intent_type": label["intent"]["type"],
            "intent_domain": label["intent"]["domain"],
            "intent_language": label["intent"]["language"],
            "response_shape": label["response"]["shape"],
            "response_interaction": label["response"]["interaction"],
        }
        return tier, self.confidence, self.source, extra
