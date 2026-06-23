"""Provider wrapper implementing OpenSquilla's Fusion Reply mode.

The implementation follows the SpecEM shape at API level:

1. OpenSquilla's normal action model decides and executes tool calls.
2. Once the action model reaches a final text answer, Fusion Reply performs
   iterative segment drafting, verification scoring, and online feedback over
   the configured fusion members.

The paper scores candidates with model logits and verify-in-line attention
masks. Hosted chat APIs generally do not expose comparable logits or attention
control, so the verification stage asks each model for normalized candidate
scores in strict JSON and applies the same weighted aggregation/update logic.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import random
import re
import string
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from opensquilla.engine.model_call_harness import (
    HarnessUsage,
    collect_visible_text_with_harness,
)
from opensquilla.fusion_reply.trace import FusionTraceRecorder, FusionTraceSettings
from opensquilla.provider.protocol import LLMProvider
from opensquilla.provider.types import (
    ChatConfig,
    DoneEvent,
    ErrorEvent,
    Message,
    ModelCapabilities,
    ModelInfo,
    ProviderHeartbeatEvent,
    ReasoningDeltaEvent,
    StreamEvent,
    TextDeltaEvent,
    ToolDefinition,
    ToolUseDeltaEvent,
    ToolUseEndEvent,
    ToolUseStartEvent,
)

_DONE_SENTINEL = "<fusion_done/>"

_DRAFT_SYSTEM = """OpenSquilla Fusion Reply candidate generation.
Generate a candidate assistant response segment for the current turn.
- Use only the conversation and available evidence already provided.
- Do not mention Fusion Reply, judging, candidate labels, or other models.
- Do not call tools or emit tool-call syntax.
- Be directly useful to the user.
"""

_STEP_DRAFT_SYSTEM = f"""{_DRAFT_SYSTEM}
You are writing one incremental segment in a multi-segment fused answer.
- Do not write the full answer.
- Target one compact section, 80-180 words, or a small table/list.
- Continue naturally from any already selected response text.
- Avoid repeating selected text.
- End at a boundary where another segment can continue.
- Append {_DONE_SENTINEL} only when the complete answer is finished and the
  current round instructions allow finishing.
"""

_VERIFY_SYSTEM = """You are an anonymous SpecEM verifier for OpenSquilla Fusion Reply.
Score each candidate segment by correctness, evidence use, instruction following,
helpfulness, and clarity. Return strict JSON only:
{"scores":{"A":0.82,"B":0.55,"C":0.31},"ranking":["A","B","C"]}.
Scores may be any non-negative numbers; they will be normalized after parsing.
"""


@dataclass(frozen=True)
class FusionMember:
    """One base model participating in fusion."""

    id: str
    provider_id: str
    model: str
    provider: LLMProvider
    weight: float = 1.0
    model_capabilities: ModelCapabilities | None = None


@dataclass(frozen=True)
class _CallUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    billed_cost: float = 0.0
    cost_source: str = "none"

    @classmethod
    def from_done(cls, event: DoneEvent) -> "_CallUsage":
        return cls(
            input_tokens=event.input_tokens,
            output_tokens=event.output_tokens,
            reasoning_tokens=event.reasoning_tokens,
            cached_tokens=event.cached_tokens,
            cache_write_tokens=event.cache_write_tokens,
            billed_cost=event.billed_cost,
            cost_source=getattr(event, "cost_source", "none") or "none",
        )

    @classmethod
    def from_harness(cls, usage: HarnessUsage) -> "_CallUsage":
        return cls(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            cached_tokens=usage.cached_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            billed_cost=usage.billed_cost,
            cost_source=usage.cost_source,
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
class _Candidate:
    index: int
    member: FusionMember
    text: str
    usage: _CallUsage
    latency_ms: float = 0.0
    seeded_from_action: bool = False


@dataclass(frozen=True)
class _SeedCandidate:
    member_id: str
    text: str
    usage: _CallUsage


@dataclass(frozen=True)
class _SegmentGoal:
    index: int
    title: str
    instruction: str
    target_words: str = "80-180 words"

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "title": self.title,
            "instruction": self.instruction,
            "target_words": self.target_words,
        }


@dataclass(frozen=True)
class _Verification:
    verifier_id: str
    scores: dict[int, float]
    usage: _CallUsage
    label_to_candidate: dict[str, int]
    raw_response: str
    label_scores: dict[str, float]
    ranking: list[str]
    parse_mode: str
    parse_fallback_used: bool
    normalized_scores: dict[int, float]
    latency_ms: float = 0.0


@dataclass(frozen=True)
class _ScoreParseResult:
    scores: dict[int, float]
    label_scores: dict[str, float]
    ranking: list[str]
    mode: str
    fallback_used: bool


class FusionReplyProvider:
    """LLMProvider that keeps tools single-model and fuses final text with SpecEM."""

    provider_name = "fusion_reply"
    finalize_artifact_delivery_with_provider = True

    def __init__(
        self,
        members: list[FusionMember],
        *,
        action_provider: LLMProvider | None = None,
        action_member_id: str = "",
        action_model: str = "",
        max_rounds: int = 4,
        min_rounds: int = 2,
        step_max_tokens: int = 1024,
        judge_max_tokens: int = 512,
        temperature: float | None = 0.7,
        judge_temperature: float | None = 0.0,
        feedback_alpha: float = 1.0,
        adaptive_segments: bool = True,
        trace_settings: FusionTraceSettings | None = None,
    ) -> None:
        self._members = list(members)
        self._action_provider = action_provider
        self._action_member_id = action_member_id
        self._action_model = action_model
        self._max_rounds = max(1, int(max_rounds or 4))
        self._min_rounds = min(
            self._max_rounds,
            max(1, int(min_rounds or 1)),
        )
        self._step_max_tokens = max(1, int(step_max_tokens or 1024))
        self._judge_max_tokens = max(1, int(judge_max_tokens or 512))
        self._temperature = temperature
        self._judge_temperature = judge_temperature
        self._feedback_alpha = max(float(feedback_alpha or 1.0), 0.0)
        self._adaptive_segments = bool(adaptive_segments)
        self._trace_settings = trace_settings or FusionTraceSettings()
        self.model = self._fusion_model_id()

    def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        config: ChatConfig | None = None,
    ) -> AsyncIterator[StreamEvent]:
        return self._chat(messages, tools=tools, config=config)

    async def _chat(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDefinition] | None,
        config: ChatConfig | None,
    ) -> AsyncIterator[StreamEvent]:
        base_config = config or ChatConfig()
        if self._action_provider is not None and tools:
            async for event in self._chat_action_then_fuse(
                messages,
                tools=tools,
                config=base_config,
            ):
                yield event
            return

        async for event in self._chat_specem(
            messages,
            config=base_config,
            initial_usage=_CallUsage(),
            seed_candidate=None,
        ):
            yield event

    async def _chat_action_then_fuse(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDefinition],
        config: ChatConfig,
    ) -> AsyncIterator[StreamEvent]:
        """Run one normal action-model call, fusing only final end-turn text."""

        action_text_parts: list[str] = []
        saw_tool = False
        try:
            stream = self._action_provider.chat(  # type: ignore[union-attr]
                messages,
                tools=tools,
                config=config,
            )
            if inspect.isawaitable(stream):
                stream = await stream
            async for event in stream:
                if isinstance(event, TextDeltaEvent):
                    action_text_parts.append(event.text)
                    continue
                if isinstance(event, ReasoningDeltaEvent):
                    yield event
                    continue
                if isinstance(event, (ToolUseStartEvent, ToolUseDeltaEvent, ToolUseEndEvent)):
                    if action_text_parts:
                        yield TextDeltaEvent(text="".join(action_text_parts))
                        action_text_parts = []
                    saw_tool = True
                    yield event
                    continue
                if isinstance(event, ProviderHeartbeatEvent):
                    yield event
                    continue
                if isinstance(event, ErrorEvent):
                    yield event
                    return
                if isinstance(event, DoneEvent):
                    action_usage = _CallUsage.from_done(event)
                    stop_reason = (event.stop_reason or "end_turn").lower()
                    if saw_tool or stop_reason == "tool_use":
                        if action_text_parts:
                            yield TextDeltaEvent(text="".join(action_text_parts))
                        yield event
                        return
                    if stop_reason not in {"end_turn", "stop"}:
                        if action_text_parts:
                            yield TextDeltaEvent(text="".join(action_text_parts))
                        yield event
                        return

                    seed_text = "".join(action_text_parts).strip()
                    seed = (
                        _SeedCandidate(
                            member_id=self._action_member_id,
                            text=seed_text,
                            usage=action_usage,
                        )
                        if (
                            seed_text
                            and self._action_member_id
                            and self._max_rounds <= 1
                        )
                        else None
                    )
                    initial_usage = _CallUsage() if seed is not None else action_usage
                    async for fused_event in self._chat_specem(
                        messages,
                        config=config,
                        initial_usage=initial_usage,
                        seed_candidate=seed,
                    ):
                        yield fused_event
                    return
        except Exception as exc:  # noqa: BLE001 - provider errors surface as stream events.
            yield ErrorEvent(message=str(exc), code="fusion_reply_action_error")
            return

        if action_text_parts:
            yield TextDeltaEvent(text="".join(action_text_parts))

    async def _chat_specem(
        self,
        messages: list[Message],
        *,
        config: ChatConfig,
        initial_usage: _CallUsage,
        seed_candidate: _SeedCandidate | None,
    ) -> AsyncIterator[StreamEvent]:
        if not self._members:
            yield ErrorEvent(
                message="Fusion Reply requires at least one configured model.",
                code="fusion_reply_no_models",
            )
            return

        trace = self._new_trace(config)
        weights = self._initial_weights()
        aggregate_usage = initial_usage
        selected_parts: list[str] = []
        rounds_completed = 0
        segment_plan = (
            self._segment_plan(messages)
            if self._adaptive_segments and self._max_rounds > 1
            else []
        )
        effective_min_rounds = self._effective_min_rounds(segment_plan)
        if trace is not None:
            trace.emit(
                "fusion.segment_plan",
                mode="adaptive_heuristic" if segment_plan else "fixed",
                effective_min_rounds=effective_min_rounds,
                segments=[goal.as_dict() for goal in segment_plan],
            )

        try:
            for round_index in range(self._max_rounds):
                segment_goal = self._segment_goal_for_round(segment_plan, round_index)
                if trace is not None:
                    trace.emit(
                        "fusion.round.start",
                        round_index=round_index,
                        weights=weights,
                        selected_chars=sum(len(part) for part in selected_parts),
                        segment_goal=(
                            segment_goal.as_dict() if segment_goal is not None else None
                        ),
                    )
                yield ProviderHeartbeatEvent(
                    phase="fusion_draft",
                    message=f"SpecEM drafting round {round_index + 1}",
                )
                candidates = await asyncio.gather(
                    *[
                        self._draft_candidate(
                            member,
                            index,
                            round_index,
                            messages,
                            selected_parts,
                            segment_plan,
                            config,
                            seed_candidate=seed_candidate if round_index == 0 else None,
                            trace=trace,
                        )
                        for index, member in enumerate(self._members)
                    ]
                )
                aggregate_usage = _add_usage(
                    aggregate_usage,
                    *[candidate.usage for candidate in candidates],
                )

                if len(candidates) == 1:
                    selected = candidates[0]
                    scores = {selected.index: 1.0}
                    verifications: list[_Verification] = []
                else:
                    yield ProviderHeartbeatEvent(
                        phase="fusion_verify",
                        message=f"SpecEM verification round {round_index + 1}",
                    )
                    verifications = await asyncio.gather(
                        *[
                            self._verify_candidates(
                                member,
                                round_index,
                            messages,
                            selected_parts,
                            segment_goal,
                            candidates,
                            config,
                            trace=trace,
                        )
                        for member in self._members
                    ]
                )
                    aggregate_usage = _add_usage(
                        aggregate_usage,
                        *[verification.usage for verification in verifications],
                    )
                    scores = self._weighted_scores(candidates, verifications, weights)
                    if trace is not None:
                        trace.emit(
                            "fusion.aggregate",
                            round_index=round_index,
                            verifier_results=[
                                {
                                    "verifier_id": verification.verifier_id,
                                    "weight": weights.get(verification.verifier_id, 1.0),
                                    "normalized_scores": verification.normalized_scores,
                                }
                                for verification in verifications
                            ],
                            weighted_scores=scores,
                        )
                    selected = self._select_candidate(candidates, scores, weights)

                previous_weights = weights
                weights = self._updated_weights(
                    round_index,
                    candidates,
                    verifications,
                    weights,
                )
                selected_text, candidate_is_done = _strip_done_sentinel(selected.text)
                selected_text = _prepare_selected_segment(selected_parts, selected_text)
                done_ignored_reason = ""
                if candidate_is_done and (round_index + 1) < effective_min_rounds:
                    is_done = False
                    done_ignored_reason = "min_rounds_not_reached"
                else:
                    is_done = candidate_is_done
                if trace is not None:
                    trace.emit(
                        "fusion.select",
                        round_index=round_index,
                        selected_candidate_id=selected.index,
                        selected_member_id=selected.member.id,
                        selected_text=(
                            selected_text
                            if trace.settings.include_candidate_text
                            else None
                        ),
                        candidate_is_done=candidate_is_done,
                        is_done=is_done,
                        done_ignored_reason=done_ignored_reason,
                        segment_goal=(
                            segment_goal.as_dict() if segment_goal is not None else None
                        ),
                        selection_reason=(
                            "highest weighted score, then member weight, then candidate order"
                        ),
                        weighted_scores=scores,
                    )
                    trace.emit(
                        "fusion.weights.update",
                        round_index=round_index,
                        weights_before=previous_weights,
                        weights_after=weights,
                    )
                    trace.record_selected(selected.member.id, selected_text)
                rounds_completed = round_index + 1
                if selected_text:
                    selected_parts.append(selected_text)
                    yield TextDeltaEvent(text=selected_text)
                if is_done:
                    break
        except Exception as exc:  # noqa: BLE001 - provider errors surface as stream events.
            if trace is not None:
                trace.finish(
                    status="error",
                    rounds_completed=rounds_completed,
                    error=str(exc),
                )
            yield ErrorEvent(message=str(exc), code="fusion_reply_error")
            return

        summary = (
            trace.finish(status="ok", rounds_completed=rounds_completed)
            if trace is not None
            else {}
        )
        yield DoneEvent(
            stop_reason="end_turn",
            input_tokens=aggregate_usage.input_tokens,
            output_tokens=aggregate_usage.output_tokens,
            reasoning_tokens=aggregate_usage.reasoning_tokens,
            cached_tokens=aggregate_usage.cached_tokens,
            billed_cost=aggregate_usage.billed_cost,
            model=self._fusion_model_id(),
            cache_write_tokens=aggregate_usage.cache_write_tokens,
            cost_source=aggregate_usage.cost_source,
            metadata=(
                {"fusion_summary": summary}
                if summary and self._trace_settings.expose_summary
                else {}
            ),
        )

    async def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                provider=self.provider_name,
                model_id=self._fusion_model_id(),
                display_name="Fusion Reply",
                supports_tools=self._action_provider is not None,
                supports_streaming=True,
            )
        ]

    async def _draft_candidate(
        self,
        member: FusionMember,
        index: int,
        round_index: int,
        messages: list[Message],
        selected_parts: list[str],
        segment_plan: list[_SegmentGoal],
        base_config: ChatConfig,
        *,
        seed_candidate: _SeedCandidate | None,
        trace: FusionTraceRecorder | None,
    ) -> _Candidate:
        if seed_candidate is not None and seed_candidate.member_id == member.id:
            candidate = _Candidate(
                index=index,
                member=member,
                text=seed_candidate.text,
                usage=seed_candidate.usage,
                latency_ms=0.0,
                seeded_from_action=True,
            )
            if trace is not None:
                trace.emit(
                    "fusion.draft.request",
                    round_index=round_index,
                    member_id=member.id,
                    provider=member.provider_id,
                    model=member.model,
                    seeded_from_action=True,
                )
                trace.emit(
                    "fusion.draft.result",
                    round_index=round_index,
                    candidate_id=index,
                    member_id=member.id,
                    text=(
                        candidate.text
                        if trace.settings.include_candidate_text
                        else None
                    ),
                    usage=candidate.usage.as_dict(),
                    latency_ms=candidate.latency_ms,
                    seeded_from_action=True,
                )
                trace.record_usage(
                    member.id,
                    stage="draft",
                    usage=candidate.usage.as_dict(),
                )
            return candidate

        draft_messages = self._round_messages(messages, selected_parts, segment_plan, round_index)
        system = self._draft_system(round_index, segment_plan)
        call_config = self._call_config(
            base_config,
            system_suffix=system,
            max_tokens=self._step_max_tokens,
            temperature=self._temperature,
            model_capabilities=member.model_capabilities,
        )
        if trace is not None:
            trace.emit(
                "fusion.draft.request",
                round_index=round_index,
                member_id=member.id,
                provider=member.provider_id,
                model=member.model,
                request=(
                    _trace_request(call_config, draft_messages)
                    if trace.settings.include_prompts
                    else None
                ),
        )
        started = time.perf_counter()
        harness_result = await collect_visible_text_with_harness(
            member.provider,
            draft_messages,
            config=call_config,
        )
        text = harness_result.text
        usage = _CallUsage.from_harness(harness_result.usage)
        latency_ms = (time.perf_counter() - started) * 1000.0
        candidate = _Candidate(
            index=index,
            member=member,
            text=text.strip(),
            usage=usage,
            latency_ms=latency_ms,
        )
        if trace is not None:
            trace.emit(
                "fusion.draft.result",
                round_index=round_index,
                candidate_id=index,
                member_id=member.id,
                text=(
                    candidate.text
                    if trace.settings.include_candidate_text
                    else None
                ),
                usage=candidate.usage.as_dict(),
                latency_ms=candidate.latency_ms,
                harness=harness_result.trace_payload(),
            )
            trace.record_usage(member.id, stage="draft", usage=candidate.usage.as_dict())
        return candidate

    async def _verify_candidates(
        self,
        verifier: FusionMember,
        round_index: int,
        messages: list[Message],
        selected_parts: list[str],
        segment_goal: _SegmentGoal | None,
        candidates: list[_Candidate],
        base_config: ChatConfig,
        *,
        trace: FusionTraceRecorder | None,
    ) -> _Verification:
        labels = list(string.ascii_uppercase[: len(candidates)])
        order = list(range(len(candidates)))
        random.Random(f"{round_index}:{verifier.id}").shuffle(order)
        label_to_candidate = {
            labels[position]: candidates[candidate_pos].index
            for position, candidate_pos in enumerate(order)
        }
        prompt = self._verification_prompt(
            messages,
            selected_parts,
            segment_goal,
            [
                (labels[position], candidates[candidate_pos].text)
                for position, candidate_pos in enumerate(order)
            ],
        )
        call_config = self._call_config(
            base_config,
            system_suffix=_VERIFY_SYSTEM,
            max_tokens=self._judge_max_tokens,
            temperature=self._judge_temperature,
            model_capabilities=verifier.model_capabilities,
        )
        if trace is not None:
            trace.emit(
                "fusion.anonymization",
                round_index=round_index,
                verifier_id=verifier.id,
                label_to_candidate=label_to_candidate,
            )
            trace.emit(
                "fusion.verify.request",
                round_index=round_index,
                verifier_id=verifier.id,
                provider=verifier.provider_id,
                model=verifier.model,
                label_to_candidate=label_to_candidate,
                request=(
                    _trace_request(call_config, [Message(role="user", content=prompt)])
                    if trace.settings.include_prompts
                    else None
                ),
        )
        started = time.perf_counter()
        harness_result = await collect_visible_text_with_harness(
            verifier.provider,
            [Message(role="user", content=prompt)],
            config=call_config,
        )
        text = harness_result.text
        usage = _CallUsage.from_harness(harness_result.usage)
        latency_ms = (time.perf_counter() - started) * 1000.0
        parsed = _parse_scores_detail(text, label_to_candidate)
        valid_ids = {candidate.index for candidate in candidates}
        normalized = _normalize_scores(parsed.scores, valid_ids)
        verification = _Verification(
            verifier_id=verifier.id,
            scores=parsed.scores,
            usage=usage,
            label_to_candidate=label_to_candidate,
            raw_response=text,
            label_scores=parsed.label_scores,
            ranking=parsed.ranking,
            parse_mode=parsed.mode,
            parse_fallback_used=parsed.fallback_used,
            normalized_scores=normalized,
            latency_ms=latency_ms,
        )
        if trace is not None:
            trace.emit(
                "fusion.verify.result",
                round_index=round_index,
                verifier_id=verifier.id,
                raw_response=(
                    verification.raw_response
                    if trace.settings.include_verifier_text
                    else None
                ),
                label_scores=verification.label_scores,
                candidate_scores=verification.scores,
                normalized_scores=verification.normalized_scores,
                ranking=verification.ranking,
                parse_mode=verification.parse_mode,
                parse_fallback_used=verification.parse_fallback_used,
                usage=verification.usage.as_dict(),
                latency_ms=verification.latency_ms,
                harness=harness_result.trace_payload(),
            )
            trace.record_usage(
                verifier.id,
                stage="verify",
                usage=verification.usage.as_dict(),
            )
        return verification

    def _call_config(
        self,
        base_config: ChatConfig,
        *,
        system_suffix: str,
        max_tokens: int,
        temperature: float | None,
        model_capabilities: ModelCapabilities | None = None,
    ) -> ChatConfig:
        existing_system = base_config.system or ""
        system = f"{existing_system}\n\n{system_suffix}".strip()
        update: dict[str, Any] = {
            "system": system,
            "max_tokens": (
                min(max_tokens, base_config.max_tokens)
                if base_config.max_tokens
                else max_tokens
            ),
            "thinking": False,
            "tool_choice": "none",
            "model_capabilities": model_capabilities or base_config.model_capabilities,
        }
        if temperature is not None:
            update["temperature"] = temperature
        return base_config.model_copy(update=update)

    def _round_messages(
        self,
        messages: list[Message],
        selected_parts: list[str],
        segment_plan: list[_SegmentGoal],
        round_index: int,
    ) -> list[Message]:
        if not selected_parts:
            return list(messages)
        segment_goal = self._segment_goal_for_round(segment_plan, round_index)
        goal_text = (
            f" Current segment goal: {segment_goal.title}. {segment_goal.instruction}"
            if segment_goal is not None
            else ""
        )
        return [
            *messages,
            Message(role="assistant", content="".join(selected_parts)),
            Message(
                role="user",
                content=(
                    "Continue with only the next coherent segment. "
                    "Do not repeat already selected text."
                    f"{goal_text}"
                ),
            ),
        ]

    def _draft_system(
        self,
        round_index: int,
        segment_plan: list[_SegmentGoal],
    ) -> str:
        if self._max_rounds <= 1:
            return f"{_DRAFT_SYSTEM}\nRound: {round_index + 1}."
        round_number = round_index + 1
        effective_min_rounds = self._effective_min_rounds(segment_plan)
        if round_number < effective_min_rounds:
            finish_rule = (
                f"This is round {round_number} of at least {effective_min_rounds}; "
                f"do not append {_DONE_SENTINEL} yet."
            )
        else:
            finish_rule = (
                f"This is round {round_number}; append {_DONE_SENTINEL} only if "
                "this segment completes the answer."
            )
        segment_goal = self._segment_goal_for_round(segment_plan, round_index)
        goal_block = (
            "\nCurrent segment goal:\n"
            f"- Title: {segment_goal.title}\n"
            f"- Instruction: {segment_goal.instruction}\n"
            f"- Target length: {segment_goal.target_words}\n"
            if segment_goal is not None
            else ""
        )
        plan_block = self._segment_plan_prompt(segment_plan)
        return (
            f"{_STEP_DRAFT_SYSTEM}\n"
            f"Round: {round_number} of at most {self._max_rounds}.\n"
            f"Minimum selected segments before completion: {effective_min_rounds}.\n"
            f"{finish_rule}"
            f"{goal_block}"
            f"{plan_block}"
        )

    def _segment_plan(self, messages: list[Message]) -> list[_SegmentGoal]:
        text = _latest_user_text(messages).lower()
        if _contains_any(
            text,
            (
                "研报",
                "调研",
                "报告",
                "股价",
                "估值",
                "新闻",
                "research",
                "report",
                "valuation",
                "stock",
            ),
        ):
            goals = [
                ("Conclusion", "Give the core answer and framing first."),
                ("Evidence", "Summarize the key facts, data, and sources already gathered."),
                ("Drivers", "Explain the main positive and negative drivers."),
                ("Estimate", "Develop the valuation, forecast, or reasoned estimate."),
                ("Risks", "State uncertainty, risk factors, and what would change the view."),
                ("Final synthesis", "Close with the actionable takeaway or deliverable status."),
            ]
            target = "100-180 words"
        elif _contains_any(
            text,
            (
                "代码",
                "实现",
                "开发",
                "报错",
                "日志",
                "修复",
                "前端",
                "后端",
                "trace",
                "bug",
                "debug",
                "implement",
                "fix",
            ),
        ):
            goals = [
                ("Diagnosis", "State what is happening and the most likely cause."),
                ("Design", "Explain the intended behavior and change boundary."),
                ("Implementation", "Describe the concrete code or configuration changes."),
                ("Verification", "Cover tests, traces, remaining risks, and next validation."),
            ]
            target = "80-160 words"
        elif _contains_any(
            text,
            (
                "方案",
                "设计",
                "计划",
                "架构",
                "spec",
                "plan",
                "strategy",
                "workflow",
            ),
        ):
            goals = [
                ("Goal and constraints", "Restate the goal, non-goals, and constraints."),
                ("Options", "Compare viable approaches and tradeoffs."),
                ("Recommended design", "Specify the preferred design and why."),
                ("Execution path", "Lay out implementation, validation, and observability."),
            ]
            target = "80-160 words"
        else:
            goals = [
                ("Direct answer", "Answer the user's question plainly."),
                ("Mechanics", "Explain the reasoning or mechanism with a compact example."),
                ("Implications", "State practical consequences, caveats, or next checks."),
            ]
            target = "60-140 words"

        desired = min(self._max_rounds, max(self._min_rounds, len(goals)))
        selected = list(goals[:desired])
        while len(selected) < desired:
            selected.append(
                (
                    f"Continuation {len(selected) + 1}",
                    "Continue with the next unrepeated semantic unit.",
                )
            )
        return [
            _SegmentGoal(index=index, title=title, instruction=instruction, target_words=target)
            for index, (title, instruction) in enumerate(selected)
        ]

    def _effective_min_rounds(self, segment_plan: list[_SegmentGoal]) -> int:
        planned_rounds = len(segment_plan) if self._adaptive_segments else 0
        return min(
            self._max_rounds,
            max(self._min_rounds, planned_rounds, 1),
        )

    def _segment_goal_for_round(
        self,
        segment_plan: list[_SegmentGoal],
        round_index: int,
    ) -> _SegmentGoal | None:
        if not segment_plan:
            return None
        if round_index < len(segment_plan):
            return segment_plan[round_index]
        return _SegmentGoal(
            index=round_index,
            title=f"Remaining gaps {round_index + 1}",
            instruction="Address only remaining unrepeated gaps and finish if complete.",
            target_words=segment_plan[-1].target_words,
        )

    def _segment_plan_prompt(self, segment_plan: list[_SegmentGoal]) -> str:
        if not segment_plan:
            return ""
        rows = "\n".join(
            f"{goal.index + 1}. {goal.title}: {goal.instruction}"
            for goal in segment_plan
        )
        return (
            "\nAdaptive segment plan for this answer. Do not expose these labels "
            "unless they are naturally useful to the user:\n"
            f"{rows}\n"
        )

    def _verification_prompt(
        self,
        messages: list[Message],
        selected_parts: list[str],
        segment_goal: _SegmentGoal | None,
        labelled_candidates: list[tuple[str, str]],
    ) -> str:
        conversation = _serialize_messages(messages[-8:])
        previous = "".join(selected_parts).strip()
        candidate_blocks = "\n\n".join(
            f"Candidate {label}:\n{text or '[empty response]'}"
            for label, text in labelled_candidates
        )
        previous_block = f"\nAlready selected response text:\n{previous}\n" if previous else ""
        goal_block = (
            "\nCurrent segment goal:\n"
            f"{segment_goal.title}: {segment_goal.instruction}\n"
            if segment_goal is not None
            else ""
        )
        labels = ", ".join(label for label, _text in labelled_candidates)
        return (
            f"Conversation:\n{conversation}\n"
            f"{previous_block}\n"
            f"{goal_block}\n"
            f"Score these anonymous candidate segments independently. "
            f"Use only these labels: {labels}. Higher is better.\n\n"
            f"{candidate_blocks}\n\n"
            'Return JSON exactly like {"scores":{"A":0.9,"B":0.4},"ranking":["A","B"]}.'
        )

    def _weighted_scores(
        self,
        candidates: list[_Candidate],
        verifications: list[_Verification],
        weights: dict[str, float],
    ) -> dict[int, float]:
        valid_ids = {candidate.index for candidate in candidates}
        scores = {candidate.index: 0.0 for candidate in candidates}
        for verification in verifications:
            normalized = _normalize_scores(verification.scores, valid_ids)
            verifier_weight = weights.get(verification.verifier_id, 1.0)
            for candidate_id, score in normalized.items():
                scores[candidate_id] += verifier_weight * score
        return scores

    def _select_candidate(
        self,
        candidates: list[_Candidate],
        scores: dict[int, float],
        weights: dict[str, float],
    ) -> _Candidate:
        return max(
            candidates,
            key=lambda candidate: (
                scores.get(candidate.index, 0.0),
                weights.get(candidate.member.id, 0.0),
                -candidate.index,
            ),
        )

    def _initial_weights(self) -> dict[str, float]:
        weights = {
            member.id: max(float(member.weight or 1.0), 0.001)
            for member in self._members
        }
        total = sum(weights.values()) or 1.0
        return {key: value / total for key, value in weights.items()}

    def _updated_weights(
        self,
        round_index: int,
        candidates: list[_Candidate],
        verifications: list[_Verification],
        weights: dict[str, float],
    ) -> dict[str, float]:
        rewards = _feedback_rewards(candidates, verifications)
        if not rewards:
            return weights
        learning_rate = self._feedback_alpha * math.sqrt(1.0 / (round_index + 1)) / max(
            len(candidates),
            1,
        )
        updated = dict(weights)
        for candidate in candidates:
            reward = max(rewards.get(candidate.index, 0.0), 0.0)
            updated[candidate.member.id] = max(
                0.001,
                weights.get(candidate.member.id, 0.001) * math.exp(learning_rate * reward),
            )
        total = sum(updated.values()) or 1.0
        return {key: value / total for key, value in updated.items()}

    def _new_trace(self, config: ChatConfig) -> FusionTraceRecorder | None:
        if not (self._trace_settings.enabled or self._trace_settings.expose_summary):
            return None
        return FusionTraceRecorder(
            self._trace_settings,
            metadata=config.metadata,
            members=[
                {
                    "id": member.id,
                    "provider": member.provider_id,
                    "model": member.model,
                    "initial_weight": member.weight,
                }
                for member in self._members
            ],
            action_model=self._action_model,
            max_rounds=self._max_rounds,
            min_rounds=self._min_rounds,
        )

    def _fusion_model_id(self) -> str:
        joined = "+".join(member.id for member in self._members)
        return f"fusion:{joined}" if joined else "fusion"


def _add_usage(base: _CallUsage, *items: _CallUsage) -> _CallUsage:
    cost_sources = [item.cost_source for item in (base, *items) if item.cost_source != "none"]
    if any(source == "provider" for source in cost_sources):
        cost_source = "provider"
    else:
        cost_source = cost_sources[0] if cost_sources else "none"
    return _CallUsage(
        input_tokens=base.input_tokens + sum(item.input_tokens for item in items),
        output_tokens=base.output_tokens + sum(item.output_tokens for item in items),
        reasoning_tokens=base.reasoning_tokens + sum(item.reasoning_tokens for item in items),
        cached_tokens=base.cached_tokens + sum(item.cached_tokens for item in items),
        cache_write_tokens=base.cache_write_tokens
        + sum(item.cache_write_tokens for item in items),
        billed_cost=base.billed_cost + sum(item.billed_cost for item in items),
        cost_source=cost_source,
    )


def _strip_done_sentinel(text: str) -> tuple[str, bool]:
    if _DONE_SENTINEL not in text:
        return text, False
    return text.replace(_DONE_SENTINEL, "").rstrip(), True


def _prepare_selected_segment(selected_parts: list[str], text: str) -> str:
    if not selected_parts or not text:
        return text
    previous = "".join(selected_parts)
    if not previous or previous[-1].isspace() or text[0].isspace():
        return text
    return f"\n\n{text}"


def _latest_user_text(messages: list[Message]) -> str:
    for message in reversed(messages):
        if str(message.role).lower() != "user":
            continue
        if _is_tool_result_content(message.content):
            continue
        text = _clean_user_request_text(_message_content_text(message.content))
        if text:
            return text
    return ""


def _message_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                value = item.get("text") or item.get("content") or ""
                if value:
                    parts.append(str(value))
            elif hasattr(item, "text") or hasattr(item, "content"):
                value = getattr(item, "text", None) or getattr(item, "content", None)
                if value:
                    parts.append(str(value))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content or "")


def _is_tool_result_content(content: Any) -> bool:
    if isinstance(content, list):
        typed_items = [item for item in content if _content_block_type(item)]
        return bool(typed_items) and all(
            _content_block_type(item) == "tool_result" for item in typed_items
        )
    if isinstance(content, dict):
        return str(content.get("type") or "") == "tool_result"
    if hasattr(content, "type"):
        return str(getattr(content, "type", "") or "") == "tool_result"
    return False


def _content_block_type(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("type") or "")
    return str(getattr(item, "type", "") or "")


def _clean_user_request_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    if stripped.startswith("[Request context for this turn]"):
        return ""
    runtime_marker = "\n[Runtime context for this turn]"
    if runtime_marker in stripped:
        stripped = stripped.split(runtime_marker, 1)[0].rstrip()
    return stripped


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _parse_scores(raw: str, label_to_candidate: dict[str, int]) -> dict[int, float]:
    return _parse_scores_detail(raw, label_to_candidate).scores


def _parse_scores_detail(raw: str, label_to_candidate: dict[str, int]) -> _ScoreParseResult:
    labels = set(label_to_candidate)
    try:
        payload = json.loads(_extract_json_object(raw))
        parsed, label_scores = _scores_from_payload(payload, label_to_candidate)
        if parsed:
            ranking = _ranking_from_payload_or_scores(payload, label_scores)
            return _ScoreParseResult(
                scores=parsed,
                label_scores=label_scores,
                ranking=ranking,
                mode="scores",
                fallback_used=False,
            )
        ranking = payload.get("ranking")
        if isinstance(ranking, list):
            ranking_labels = [
                str(label).strip().upper()
                for label in ranking
                if str(label).strip().upper() in labels
            ]
            parsed, label_scores = _scores_from_ranking(
                ranking_labels,
                label_to_candidate,
            )
            if parsed:
                return _ScoreParseResult(
                    scores=parsed,
                    label_scores=label_scores,
                    ranking=_dedupe_labels(ranking_labels),
                    mode="ranking",
                    fallback_used=False,
                )
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
        pass
    found = [
        label
        for label in re.findall(r"\b([A-Z])\b", raw.upper())
        if label in labels
    ]
    ranking = _dedupe_labels(found)
    parsed, label_scores = _scores_from_ranking(ranking, label_to_candidate)
    return _ScoreParseResult(
        scores=parsed,
        label_scores=label_scores,
        ranking=ranking,
        mode="regex_ranking" if parsed else "empty",
        fallback_used=True,
    )


def _scores_from_payload(
    payload: Any,
    label_to_candidate: dict[str, int],
) -> tuple[dict[int, float], dict[str, float]]:
    scores = payload.get("scores") if isinstance(payload, dict) else None
    if isinstance(scores, dict):
        output: dict[int, float] = {}
        label_scores: dict[str, float] = {}
        for raw_label, raw_score in scores.items():
            label = str(raw_label).strip().upper()
            if label not in label_to_candidate:
                continue
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue
            if math.isfinite(score):
                output[label_to_candidate[label]] = max(score, 0.0)
                label_scores[label] = max(score, 0.0)
        if output:
            return output, label_scores
    if isinstance(scores, list):
        output = {}
        label_scores = {}
        for item in scores:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "")).strip().upper()
            if label not in label_to_candidate:
                continue
            try:
                score = float(item.get("score"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(score):
                output[label_to_candidate[label]] = max(score, 0.0)
                label_scores[label] = max(score, 0.0)
        if output:
            return output, label_scores
    return {}, {}


def _scores_from_ranking(
    labels: list[str],
    label_to_candidate: dict[str, int],
) -> tuple[dict[int, float], dict[str, float]]:
    labels = _dedupe_labels(labels)
    total = len(labels)
    candidate_scores = {
        label_to_candidate[label]: float(total - position)
        for position, label in enumerate(labels)
        if label in label_to_candidate
    }
    label_scores = {
        label: float(total - position)
        for position, label in enumerate(labels)
        if label in label_to_candidate
    }
    return candidate_scores, label_scores


def _ranking_from_payload_or_scores(
    payload: Any,
    label_scores: dict[str, float],
) -> list[str]:
    ranking = payload.get("ranking") if isinstance(payload, dict) else None
    if isinstance(ranking, list):
        ranked = _dedupe_labels(
            [str(label).strip().upper() for label in ranking if str(label).strip()]
        )
        if ranked:
            return ranked
    return [
        label
        for label, _score in sorted(
            label_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]


def _normalize_scores(scores: dict[int, float], valid_ids: set[int]) -> dict[int, float]:
    cleaned = {
        idx: max(float(scores.get(idx, 0.0)), 0.0)
        for idx in valid_ids
    }
    total = sum(cleaned.values())
    if total <= 0:
        uniform = 1.0 / max(len(valid_ids), 1)
        return {idx: uniform for idx in valid_ids}
    return {idx: value / total for idx, value in cleaned.items()}


def _feedback_rewards(
    candidates: list[_Candidate],
    verifications: list[_Verification],
) -> dict[int, float]:
    valid_ids = {candidate.index for candidate in candidates}
    member_by_candidate = {candidate.index: candidate.member.id for candidate in candidates}
    wins = {candidate.index: 0.0 for candidate in candidates}
    total_wins = 0.0
    for verification in verifications:
        normalized = _normalize_scores(verification.scores, valid_ids)
        for candidate in candidates:
            if member_by_candidate[candidate.index] == verification.verifier_id:
                continue
            candidate_score = normalized.get(candidate.index, 0.0)
            for other in candidates:
                if other.index == candidate.index:
                    continue
                if candidate_score > normalized.get(other.index, 0.0):
                    wins[candidate.index] += 1.0
                    total_wins += 1.0
    if total_wins <= 0:
        return {}
    return {idx: value / total_wins for idx, value in wins.items()}


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def _dedupe_labels(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _trace_request(config: ChatConfig, messages: list[Message]) -> dict[str, Any]:
    return {
        "system": config.system or "",
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "thinking": config.thinking,
        "tool_choice": config.tool_choice,
        "messages": [
            {
                "role": message.role,
                "content": _message_content_for_trace(message),
            }
            for message in messages
        ],
    }


def _message_content_for_trace(message: Message) -> Any:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[dict[str, Any]] = []
        for block in content:
            block_type = getattr(block, "type", "")
            if block_type == "text":
                chunks.append({"type": "text", "text": getattr(block, "text", "")})
            elif block_type == "image":
                chunks.append({"type": "image"})
            elif block_type == "document":
                chunks.append({"type": "document", "title": getattr(block, "title", None)})
            elif block_type == "tool_result":
                chunks.append({"type": "tool_result"})
            elif block_type == "tool_use":
                chunks.append({"type": "tool_use", "name": getattr(block, "name", "")})
            else:
                chunks.append({"type": str(block_type or "unknown")})
        return chunks
    return str(content)


def _serialize_messages(messages: list[Message]) -> str:
    rendered: list[str] = []
    for message in messages:
        content = message.content
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            chunks: list[str] = []
            for block in content:
                block_text = getattr(block, "text", None)
                if isinstance(block_text, str):
                    chunks.append(block_text)
                elif getattr(block, "type", None) == "image":
                    chunks.append("[image]")
                elif getattr(block, "type", None) == "document":
                    chunks.append("[document]")
                elif getattr(block, "type", None) == "tool_result":
                    chunks.append("[tool result]")
                elif getattr(block, "type", None) == "tool_use":
                    chunks.append("[tool use]")
            text = "\n".join(chunks)
        else:
            text = str(content)
        rendered.append(f"{message.role}: {text}")
    return "\n".join(rendered)
