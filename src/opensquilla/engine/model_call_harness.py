"""Shared harness for tool-disabled model text calls.

This module hosts the lightweight provider-call protections that are useful
outside the full Agent action loop: visible-text collection, invalid-response
classification, retry policy, fallback handoff, and usage aggregation.
"""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from typing import Any

from opensquilla.engine.agent import (
    _ProviderAttemptKind,
    _ProviderRetryPolicy,
    _chat_config_with_thinking_disabled,
    _classify_provider_attempt,
    _is_large_context_invalid_response,
)
from opensquilla.provider.protocol import LLMProvider
from opensquilla.provider.types import (
    ChatConfig,
    DoneEvent,
    ErrorEvent,
    ProviderHeartbeatEvent,
    ReasoningDeltaEvent,
    TextDeltaEvent,
    ToolUseDeltaEvent,
    ToolUseEndEvent,
    ToolUseStartEvent,
)


def _strict_thinking_required() -> bool:
    return os.environ.get("OPENSQUILLA_STRICT_THINKING", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass(frozen=True)
class HarnessUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    billed_cost: float = 0.0
    cost_source: str = "none"

    @classmethod
    def from_done(cls, event: DoneEvent | None) -> "HarnessUsage":
        if event is None:
            return cls()
        return cls(
            input_tokens=event.input_tokens,
            output_tokens=event.output_tokens,
            reasoning_tokens=event.reasoning_tokens,
            cached_tokens=event.cached_tokens,
            cache_write_tokens=event.cache_write_tokens,
            billed_cost=event.billed_cost,
            cost_source=getattr(event, "cost_source", "none") or "none",
        )

    def add(self, other: "HarnessUsage") -> "HarnessUsage":
        return HarnessUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            billed_cost=self.billed_cost + other.billed_cost,
            cost_source=(
                other.cost_source
                if self.cost_source in {"", "none"} and other.cost_source
                else self.cost_source
            )
            or "none",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cached_tokens": self.cached_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "billed_cost": self.billed_cost,
            "cost_source": self.cost_source,
        }


@dataclass(frozen=True)
class HarnessAttempt:
    index: int
    classification: str
    text_chars: int
    got_done_event: bool
    stop_reason: str | None
    reasoning_tokens: int
    reasoning_chars: int
    input_tokens: int
    output_tokens: int
    model: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "classification": self.classification,
            "text_chars": self.text_chars,
            "got_done_event": self.got_done_event,
            "stop_reason": self.stop_reason,
            "reasoning_tokens": self.reasoning_tokens,
            "reasoning_chars": self.reasoning_chars,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "model": self.model,
        }


@dataclass(frozen=True)
class HarnessWarning:
    code: str
    message: str
    attempt: int
    classification: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "attempt": self.attempt,
            "classification": self.classification,
        }


@dataclass(frozen=True)
class HarnessTextResult:
    text: str
    usage: HarnessUsage
    attempts: tuple[HarnessAttempt, ...]
    warnings: tuple[HarnessWarning, ...]
    classification: str
    model: str

    def trace_payload(self) -> dict[str, Any]:
        return {
            "classification": self.classification,
            "attempt_count": len(self.attempts),
            "attempts": [attempt.as_dict() for attempt in self.attempts],
            "warnings": [warning.as_dict() for warning in self.warnings],
            "model": self.model,
        }


@dataclass(frozen=True)
class _RawCallResult:
    text: str
    done: DoneEvent | None
    got_done_event: bool
    reasoning_content: str | None
    reasoning_tokens: int
    tool_events_seen: bool


async def collect_visible_text_with_harness(
    provider: LLMProvider,
    messages: list[Any],
    *,
    config: ChatConfig,
    max_provider_retries: int = 1,
    retry_reasoning_only_without_thinking: bool = True,
    retry_empty: bool = True,
) -> HarnessTextResult:
    """Collect visible text through OpenSquilla's invalid-response policy.

    The full Agent loop owns tools and history mutation. This helper is for
    isolated text calls such as Fusion draft/verify, where tools are disabled
    but provider behavior must still be classified instead of silently accepting
    empty or reasoning-only responses.
    """

    retry_policy = _ProviderRetryPolicy.from_provider_budget(max_provider_retries)
    attempts_used = retry_policy.used_attempts()
    attempts: list[HarnessAttempt] = []
    warnings: list[HarnessWarning] = []
    aggregate_usage = HarnessUsage()
    current_config = config
    attempt_index = 0
    last_text = ""
    last_model = ""
    last_classification = _ProviderAttemptKind.MALFORMED_EMPTY.value

    while True:
        raw = await _raw_collect(provider, messages, config=current_config)
        usage = HarnessUsage.from_done(raw.done)
        aggregate_usage = aggregate_usage.add(usage)
        stop_reason = raw.done.stop_reason if raw.done else None
        model = raw.done.model if raw.done else ""
        classification = _classify_provider_attempt(
            text=raw.text,
            tool_calls=[],
            pending_tools={"tool_events": object()} if raw.tool_events_seen else {},
            got_done_event=raw.got_done_event,
            stop_reason=stop_reason,
            reasoning_content=raw.reasoning_content,
            reasoning_tokens=raw.reasoning_tokens,
            user_visible_emitted=bool(raw.text.strip()),
        )
        classification_value = classification.kind.value
        last_text = raw.text
        last_model = model
        last_classification = classification_value
        attempts.append(
            HarnessAttempt(
                index=attempt_index,
                classification=classification_value,
                text_chars=len(raw.text),
                got_done_event=raw.got_done_event,
                stop_reason=stop_reason,
                reasoning_tokens=raw.reasoning_tokens,
                reasoning_chars=len(raw.reasoning_content or ""),
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                model=model,
            )
        )
        if classification.kind == _ProviderAttemptKind.OK:
            return HarnessTextResult(
                text=raw.text,
                usage=aggregate_usage,
                attempts=tuple(attempts),
                warnings=tuple(warnings),
                classification=classification_value,
                model=model,
            )

        if (
            classification.kind
            in {_ProviderAttemptKind.REASONING_ONLY, _ProviderAttemptKind.MALFORMED_EMPTY}
            and _is_large_context_invalid_response(
                classification.kind,
                input_tokens=usage.input_tokens,
            )
            and _try_provider_fallback(provider, classification_value)
        ):
            warnings.append(
                HarnessWarning(
                    code="provider_large_context_fallback",
                    message=(
                        "The provider returned no visible response for a large input; "
                        "trying a fallback provider once."
                    ),
                    attempt=attempt_index,
                    classification=classification_value,
                )
            )
            attempt_index += 1
            continue

        if (
            classification.kind == _ProviderAttemptKind.REASONING_ONLY
            and _is_large_context_invalid_response(
                classification.kind,
                input_tokens=usage.input_tokens,
            )
            and bool(current_config.thinking)
            and not _strict_thinking_required()
            and retry_policy.can_retry_attempt(
                _ProviderAttemptKind.REASONING_ONLY,
                attempts_used,
            )
        ):
            attempts_used[_ProviderAttemptKind.REASONING_ONLY] += 1
            current_config = _chat_config_with_thinking_disabled(current_config)
            warnings.append(
                HarnessWarning(
                    code="provider_large_context_visible_retry",
                    message=(
                        "The provider returned reasoning without visible content "
                        "for a large input; retrying once with thinking disabled."
                    ),
                    attempt=attempt_index,
                    classification=classification_value,
                )
            )
            attempt_index += 1
            continue

        if (
            classification.kind == _ProviderAttemptKind.REASONING_ONLY
            and retry_reasoning_only_without_thinking
            and not (bool(current_config.thinking) and _strict_thinking_required())
            and retry_policy.can_retry_attempt(
                _ProviderAttemptKind.REASONING_ONLY,
                attempts_used,
            )
        ):
            attempts_used[_ProviderAttemptKind.REASONING_ONLY] += 1
            warnings.append(
                HarnessWarning(
                    code="provider_reasoning_only_retry",
                    message=(
                        "The provider returned reasoning without visible content; "
                        "retrying once to request visible content."
                    ),
                    attempt=attempt_index,
                    classification=classification_value,
                )
            )
            attempt_index += 1
            continue

        if (
            classification.kind == _ProviderAttemptKind.MALFORMED_EMPTY
            and retry_empty
            and retry_policy.can_retry_attempt(
                _ProviderAttemptKind.MALFORMED_EMPTY,
                attempts_used,
            )
        ):
            attempts_used[_ProviderAttemptKind.MALFORMED_EMPTY] += 1
            warnings.append(
                HarnessWarning(
                    code="provider_empty_retry",
                    message="The provider returned an empty response; retrying once.",
                    attempt=attempt_index,
                    classification=classification_value,
                )
            )
            attempt_index += 1
            continue

        return HarnessTextResult(
            text=last_text,
            usage=aggregate_usage,
            attempts=tuple(attempts),
            warnings=tuple(warnings),
            classification=last_classification,
            model=last_model,
        )


async def _raw_collect(
    provider: LLMProvider,
    messages: list[Any],
    *,
    config: ChatConfig,
) -> _RawCallResult:
    parts: list[str] = []
    reasoning_parts: list[str] = []
    done: DoneEvent | None = None
    got_done_event = False
    tool_events_seen = False
    stream = provider.chat(messages, tools=None, config=config)
    if inspect.isawaitable(stream):
        stream = await stream
    async for event in stream:
        if isinstance(event, TextDeltaEvent):
            parts.append(event.text)
        elif isinstance(event, ReasoningDeltaEvent):
            if event.text:
                reasoning_parts.append(event.text)
        elif isinstance(event, DoneEvent):
            done = event
            got_done_event = True
        elif isinstance(
            event,
            ToolUseStartEvent | ToolUseDeltaEvent | ToolUseEndEvent,
        ):
            tool_events_seen = True
        elif isinstance(event, ProviderHeartbeatEvent):
            continue
        elif isinstance(event, ErrorEvent):
            raise RuntimeError(event.message or event.code or "provider error")
    reasoning_content = None
    if done and done.reasoning_content:
        reasoning_content = done.reasoning_content
    elif reasoning_parts:
        reasoning_content = "".join(reasoning_parts)
    return _RawCallResult(
        text="".join(parts),
        done=done,
        got_done_event=got_done_event,
        reasoning_content=reasoning_content,
        reasoning_tokens=done.reasoning_tokens if done else 0,
        tool_events_seen=tool_events_seen,
    )


def _try_provider_fallback(provider: LLMProvider, reason: str) -> bool:
    fallback = getattr(provider, "fallback_after_invalid_response", None)
    if not callable(fallback):
        return False
    try:
        return bool(fallback(reason))
    except Exception:
        return False
