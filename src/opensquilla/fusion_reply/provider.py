"""Provider wrapper implementing OpenSquilla's Fusion Assist mode.

1. Fusion members draft and verify hidden next-step advice with tools disabled.
2. Each advice segment is selected by anonymous SpecEM-style verification.
3. The selected advice segments are injected into the next action-model request.
4. OpenSquilla's normal action model remains the only model that can call tools
   and author user-visible text.

The paper scores candidates with model logits and verify-in-line attention
masks. Hosted chat APIs generally do not expose comparable logits or attention
control, so the verification stage asks each model for normalized hidden-advice
candidate scores in strict JSON and aggregates anonymous verifier scores with
equal verifier influence.
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
from dataclasses import dataclass, replace
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
    StreamEvent,
    ToolDefinition,
)

_ASSIST_DRAFT_SYSTEM = """OpenSquilla Fusion Assist candidate generation.
Generate one hidden next-step advice segment for the action model in the current agent loop.
- Do not write the final user-visible answer.
- Do not call tools or emit tool-call syntax.
- The action model is the only model allowed to call tools.
- Focus only on the requested advice segment.
- If a tool appears useful, recommend it in prose only; do not produce JSON tool arguments.
- Be concise, concrete, and directly useful to the action model.
"""

_ASSIST_SEGMENT_PLANNER_SYSTEM = """OpenSquilla Fusion Assist semantic segment planner.
Plan hidden advice segments for the next action-model agent-loop call.
- You are the same action model that will later call tools or answer.
- Do not call tools, write tool-call JSON, or write the final user-visible answer.
- Break the next action-model call into semantic hidden-advice slots that would help you act well.
- Prefer 1-4 focused slots; never exceed the requested maximum.
- Avoid generic filler. Each slot should be useful for this exact turn.
- Return strict JSON only:
{"segments":[{"title":"Task state","instruction":"Summarize the request and current state.","target_words":"30-90 words"}]}
"""

_ASSIST_VERIFY_SYSTEM = """You are an anonymous SpecEM verifier for OpenSquilla Fusion Assist.
Score each hidden advice candidate by usefulness for the next action-model
agent-loop step: correctness, tool guidance, context awareness, risk handling,
and concision. Return strict JSON only:
{"scores":{"A":0.82,"B":0.55,"C":0.31},"ranking":["A","B","C"]}.
Scores may be any non-negative numbers; they will be normalized after parsing.
"""

_ASSIST_SEGMENT_FUSION_SYSTEM = """OpenSquilla Fusion Assist constrained segment fuser.
Fuse the anonymous candidate advice segments into one hidden advice segment.
- Use only information present in the provided candidates.
- Do not add new facts, sources, numbers, tool results, or tool arguments.
- Treat incomplete, truncated, or dangling candidates as raw material to repair, not text to copy.
- Produce a complete, clean advice segment that is safe to inject into the action model.
- If candidates are incomplete or thin, close the segment conservatively by stating uncertainty or missing evidence.
- Preserve useful concrete guidance and caveats.
- Remove contradictions by stating uncertainty or choosing the better-supported wording.
- Do not mention candidates, labels, judging, Fusion, or other models.
- Return only the fused hidden advice segment.
"""

_ASSIST_STITCH_SYSTEM = """OpenSquilla Fusion Assist constrained final stitch editor.
Stitch the selected hidden advice segments into one coherent hidden advice block.
- Use only the selected advice segments as source material.
- Do not add new facts, sources, numbers, tool results, or tool arguments.
- Only remove duplication, smooth transitions, and unify formatting.
- Preserve segment headings when useful for the action model.
- Do not mention candidates, judging, Fusion, or other models.
- Return only the stitched hidden advice block.
"""

_ASSIST_INJECTION_PREFIX = "Hidden Fusion Assist advice for this agent iteration:"


@dataclass(frozen=True)
class FusionMember:
    """One base model participating in fusion."""

    id: str
    provider_id: str
    model: str
    provider: LLMProvider
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


@dataclass(frozen=True)
class _SegmentPlanParseResult:
    goals: list[_SegmentGoal]
    mode: str
    fallback_used: bool


@dataclass(frozen=True)
class _AssistResult:
    text: str
    usage: _CallUsage
    summary: dict[str, Any]


@dataclass(frozen=True)
class _FusedAdviceResult:
    text: str
    usage: _CallUsage
    fuser_member: FusionMember
    member: FusionMember | None
    applied: bool
    seed_candidate: _Candidate
    fallback_reason: str = ""
    conservative_fallback_used: bool = False
    latency_ms: float = 0.0


@dataclass(frozen=True)
class _FusedAdviceSegment:
    goal: _SegmentGoal
    text: str
    member: FusionMember | None


class FusionReplyProvider:
    """LLMProvider that keeps tools single-model and adds Fusion Assist."""

    provider_name = "fusion_reply"
    finalize_artifact_delivery_with_provider = True

    def __init__(
        self,
        members: list[FusionMember],
        *,
        action_provider: LLMProvider | None = None,
        action_model: str = "",
        assist_max_segments: int = 4,
        step_max_tokens: int = 1024,
        judge_max_tokens: int = 512,
        temperature: float | None = 0.7,
        judge_temperature: float | None = 0.0,
        trace_settings: FusionTraceSettings | None = None,
    ) -> None:
        self._members = list(members)
        self._action_provider = action_provider
        self._action_model = action_model
        self._assist_max_segments = max(1, int(assist_max_segments or 1))
        self._step_max_tokens = max(1, int(step_max_tokens or 1024))
        self._judge_max_tokens = max(1, int(judge_max_tokens or 512))
        self._temperature = temperature
        self._judge_temperature = judge_temperature
        self._trace_settings = trace_settings or FusionTraceSettings()
        self.model = self._fusion_model_id()
        self._assist_trace: FusionTraceRecorder | None = None
        self._assist_iterations = 0

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
        if self._action_provider is None:
            yield ErrorEvent(
                message="Fusion Assist requires an action provider.",
                code="fusion_reply_no_action_provider",
            )
            return
        async for event in self._chat_action_with_assist(
            messages,
            tools=tools,
            config=base_config,
        ):
            yield event

    async def _chat_action_with_assist(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDefinition] | None,
        config: ChatConfig,
    ) -> AsyncIterator[StreamEvent]:
        """Run Fusion Assist before one normal action-model provider call."""

        assist = await self._run_fusion_assist(messages, tools=tools, config=config)
        action_messages = _messages_with_fusion_advice(messages, assist.text)
        if assist.text:
            trace = self._assist_trace
            if trace is not None:
                trace.emit(
                    "fusion.assist.inject",
                    fusion_iteration=self._assist_iterations,
                    advice=(assist.text if trace.settings.include_candidate_text else None),
                    action_model=self._action_model,
                )
        try:
            stream = self._action_provider.chat(  # type: ignore[union-attr]
                action_messages,
                tools=tools,
                config=config,
            )
            if inspect.isawaitable(stream):
                stream = await stream
            async for event in stream:
                if isinstance(event, DoneEvent):
                    summary = assist.summary
                    action_usage = _CallUsage.from_done(event)
                    combined_usage = _add_usage(assist.usage, action_usage)
                    metadata = dict(event.metadata or {})
                    if summary and self._trace_settings.expose_summary:
                        metadata["fusion_summary"] = summary
                    yield replace(
                        event,
                        input_tokens=combined_usage.input_tokens,
                        output_tokens=combined_usage.output_tokens,
                        reasoning_tokens=combined_usage.reasoning_tokens,
                        cached_tokens=combined_usage.cached_tokens,
                        billed_cost=combined_usage.billed_cost,
                        cache_write_tokens=combined_usage.cache_write_tokens,
                        cost_source=combined_usage.cost_source,
                        metadata=metadata,
                    )
                    return
                yield event
        except Exception as exc:  # noqa: BLE001 - provider errors surface as stream events.
            yield ErrorEvent(message=str(exc), code="fusion_reply_action_error")
            return

    async def _run_fusion_assist(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDefinition] | None,
        config: ChatConfig,
    ) -> _AssistResult:
        if not self._members:
            return _AssistResult(text="", usage=_CallUsage(), summary={})

        trace = self._assist_trace_for(config)
        aggregate_usage = _CallUsage()
        selected_advices: list[str] = []
        fused_segments: list[_FusedAdviceSegment] = []
        fusion_iteration = self._assist_iterations + 1

        if trace is not None:
            trace.emit(
                "fusion.assist.start",
                fusion_iteration=fusion_iteration,
                action_model=self._action_model,
                tool_names=[tool.name for tool in tools or []],
            )

        segment_plan, plan_usage = await self._assist_segment_plan(
            messages,
            tools,
            config,
            fusion_iteration,
            trace=trace,
        )
        aggregate_usage = _add_usage(aggregate_usage, plan_usage)

        if trace is not None:
            trace.emit(
                "fusion.assist.segment_plan",
                fusion_iteration=fusion_iteration,
                mode="action_model",
                max_segments=self._assist_max_segments,
                segments=[goal.as_dict() for goal in segment_plan],
            )

        try:
            for segment_index, segment_goal in enumerate(segment_plan):
                if trace is not None:
                    trace.emit(
                        "fusion.assist.segment.start",
                        fusion_iteration=fusion_iteration,
                        segment_index=segment_index,
                        segment_goal=segment_goal.as_dict(),
                    )
                candidates = await asyncio.gather(
                    *[
                        self._draft_advice_candidate(
                            member,
                            index,
                            fusion_iteration,
                            segment_index,
                            segment_goal,
                            messages,
                            tools,
                            selected_advices,
                            config,
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
                    verifications: list[_Verification] = []
                    scores = {selected.index: 1.0}
                else:
                    verifications = await asyncio.gather(
                        *[
                            self._verify_advice_candidates(
                                member,
                                fusion_iteration,
                                segment_index,
                                segment_goal,
                                messages,
                                tools,
                                selected_advices,
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
                    scores = self._anonymous_scores(candidates, verifications)
                    if trace is not None:
                        trace.emit(
                            "fusion.assist.aggregate",
                            fusion_iteration=fusion_iteration,
                            segment_index=segment_index,
                            segment_goal=segment_goal.as_dict(),
                            verifier_results=[
                                {
                                    "verifier_id": verification.verifier_id,
                                    "normalized_scores": verification.normalized_scores,
                                }
                                for verification in verifications
                            ],
                            anonymous_scores=scores,
                        )
                    selected = self._select_candidate(candidates, scores)

                fused = await self._fuse_advice_segment(
                    fusion_iteration,
                    segment_index,
                    segment_goal,
                    messages,
                    tools,
                    selected_advices,
                    candidates,
                    scores,
                    selected,
                    config,
                    trace=trace,
                )
                aggregate_usage = _add_usage(aggregate_usage, fused.usage)
                selected_text = fused.text.strip()
                selected_advices.append(selected_text)
                if selected_text:
                    fused_segments.append(
                        _FusedAdviceSegment(
                            goal=segment_goal,
                            text=selected_text,
                            member=fused.member,
                        )
                    )
                if trace is not None:
                    trace.emit(
                        "fusion.assist.select",
                        fusion_iteration=fusion_iteration,
                        segment_index=segment_index,
                        segment_goal=segment_goal.as_dict(),
                        segment_fusion_applied=fused.applied,
                        fuser_member_id=fused.fuser_member.id,
                        selected_member_id=fused.member.id if fused.member else None,
                        fuser_seed_candidate_id=fused.seed_candidate.index,
                        fuser_seed_member_id=fused.seed_candidate.member.id,
                        fallback_reason=fused.fallback_reason,
                        conservative_fallback_used=fused.conservative_fallback_used,
                        selected_advice=(
                            selected_text if trace.settings.include_candidate_text else None
                        ),
                        selection_reason=(
                            "constrained segment fusion"
                            if fused.applied
                            else "conservative complete fallback"
                        ),
                        anonymous_scores=scores,
                    )
                    if fused.member is not None:
                        trace.record_selected_advice(fused.member.id, selected_text)
        except Exception as exc:  # noqa: BLE001 - assist failures degrade to action-only.
            summary = {}
            if trace is not None:
                trace.emit(
                    "fusion.assist.error",
                    fusion_iteration=fusion_iteration,
                    error=str(exc),
                )
                summary = trace.finish(
                    status="error",
                    rounds_completed=self._assist_iterations,
                    error=str(exc),
                )
            return _AssistResult(text="", usage=aggregate_usage, summary=summary)

        advice_text, stitch_usage = await self._stitch_fused_advices(
            fusion_iteration,
            messages,
            fused_segments,
            config,
            trace=trace,
        )
        aggregate_usage = _add_usage(aggregate_usage, stitch_usage)
        self._assist_iterations = fusion_iteration
        summary = (
            trace.finish(status="ok", rounds_completed=self._assist_iterations)
            if trace is not None
            else {}
        )
        return _AssistResult(
            text=advice_text,
            usage=aggregate_usage,
            summary=summary,
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

    async def _draft_advice_candidate(
        self,
        member: FusionMember,
        index: int,
        fusion_iteration: int,
        segment_index: int,
        segment_goal: _SegmentGoal,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
        selected_advices: list[str],
        base_config: ChatConfig,
        *,
        trace: FusionTraceRecorder | None,
    ) -> _Candidate:
        advice_messages = [
            Message(
                role="user",
                content=self._assist_draft_prompt(
                    messages,
                    tools,
                    selected_advices,
                    segment_goal,
                ),
            )
        ]
        call_config = self._call_config(
            base_config,
            system_suffix=_ASSIST_DRAFT_SYSTEM,
            max_tokens=self._step_max_tokens,
            temperature=self._temperature,
            model_capabilities=member.model_capabilities,
        )
        if trace is not None:
            trace.emit(
                "fusion.assist.draft.request",
                fusion_iteration=fusion_iteration,
                segment_index=segment_index,
                segment_goal=segment_goal.as_dict(),
                member_id=member.id,
                provider=member.provider_id,
                model=member.model,
                request=(
                    _trace_request(call_config, advice_messages)
                    if trace.settings.include_prompts
                    else None
                ),
            )
        started = time.perf_counter()
        harness_result = await collect_visible_text_with_harness(
            member.provider,
            advice_messages,
            config=call_config,
        )
        usage = _CallUsage.from_harness(harness_result.usage)
        candidate = _Candidate(
            index=index,
            member=member,
            text=harness_result.text.strip(),
            usage=usage,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
        if trace is not None:
            trace.emit(
                "fusion.assist.draft.result",
                fusion_iteration=fusion_iteration,
                segment_index=segment_index,
                segment_goal=segment_goal.as_dict(),
                candidate_id=index,
                member_id=member.id,
                text=(candidate.text if trace.settings.include_candidate_text else None),
                usage=candidate.usage.as_dict(),
                latency_ms=candidate.latency_ms,
                harness=harness_result.trace_payload(),
            )
            trace.record_usage(member.id, stage="draft", usage=candidate.usage.as_dict())
        return candidate

    async def _verify_advice_candidates(
        self,
        verifier: FusionMember,
        fusion_iteration: int,
        segment_index: int,
        segment_goal: _SegmentGoal,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
        selected_advices: list[str],
        candidates: list[_Candidate],
        base_config: ChatConfig,
        *,
        trace: FusionTraceRecorder | None,
    ) -> _Verification:
        labels = list(string.ascii_uppercase[: len(candidates)])
        order = list(range(len(candidates)))
        random.Random(
            f"assist:{fusion_iteration}:{segment_index}:{verifier.id}"
        ).shuffle(order)
        label_to_candidate = {
            labels[position]: candidates[candidate_pos].index
            for position, candidate_pos in enumerate(order)
        }
        prompt = self._assist_verification_prompt(
            messages,
            tools,
            selected_advices,
            segment_goal,
            [
                (labels[position], candidates[candidate_pos].text)
                for position, candidate_pos in enumerate(order)
            ],
        )
        call_config = self._call_config(
            base_config,
            system_suffix=_ASSIST_VERIFY_SYSTEM,
            max_tokens=self._judge_max_tokens,
            temperature=self._judge_temperature,
            model_capabilities=verifier.model_capabilities,
        )
        if trace is not None:
            trace.emit(
                "fusion.assist.anonymization",
                fusion_iteration=fusion_iteration,
                segment_index=segment_index,
                segment_goal=segment_goal.as_dict(),
                verifier_id=verifier.id,
                label_to_candidate=label_to_candidate,
            )
            trace.emit(
                "fusion.assist.verify.request",
                fusion_iteration=fusion_iteration,
                segment_index=segment_index,
                segment_goal=segment_goal.as_dict(),
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
        usage = _CallUsage.from_harness(harness_result.usage)
        parsed = _parse_scores_detail(harness_result.text, label_to_candidate)
        valid_ids = {candidate.index for candidate in candidates}
        verification = _Verification(
            verifier_id=verifier.id,
            scores=parsed.scores,
            usage=usage,
            label_to_candidate=label_to_candidate,
            raw_response=harness_result.text,
            label_scores=parsed.label_scores,
            ranking=parsed.ranking,
            parse_mode=parsed.mode,
            parse_fallback_used=parsed.fallback_used,
            normalized_scores=_normalize_scores(parsed.scores, valid_ids),
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
        if trace is not None:
            trace.emit(
                "fusion.assist.verify.result",
                fusion_iteration=fusion_iteration,
                segment_index=segment_index,
                segment_goal=segment_goal.as_dict(),
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

    async def _fuse_advice_segment(
        self,
        fusion_iteration: int,
        segment_index: int,
        segment_goal: _SegmentGoal,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
        selected_advices: list[str],
        candidates: list[_Candidate],
        scores: dict[int, float],
        seed_candidate: _Candidate,
        base_config: ChatConfig,
        *,
        trace: FusionTraceRecorder | None,
    ) -> _FusedAdviceResult:
        fuser = seed_candidate.member
        prompt = self._assist_segment_fusion_prompt(
            messages,
            tools,
            selected_advices,
            segment_goal,
            candidates,
            scores,
        )
        call_config = self._call_config(
            base_config,
            system_suffix=_ASSIST_SEGMENT_FUSION_SYSTEM,
            max_tokens=self._step_max_tokens,
            temperature=0.0,
            model_capabilities=fuser.model_capabilities,
        )
        if trace is not None:
            trace.emit(
                "fusion.assist.segment_fuse.request",
                fusion_iteration=fusion_iteration,
                segment_index=segment_index,
                segment_goal=segment_goal.as_dict(),
                fuser_member_id=fuser.id,
                fuser_seed_candidate_id=seed_candidate.index,
                fuser_seed_member_id=seed_candidate.member.id,
                anonymous_scores=scores,
                request=(
                    _trace_request(call_config, [Message(role="user", content=prompt)])
                    if trace.settings.include_prompts
                    else None
                ),
            )

        started = time.perf_counter()
        conservative_fallback = _conservative_advice_fallback(segment_goal)
        try:
            harness_result = await collect_visible_text_with_harness(
                fuser.provider,
                [Message(role="user", content=prompt)],
                config=call_config,
            )
        except Exception as exc:  # noqa: BLE001 - fusion failure falls back to safe advice.
            latency_ms = (time.perf_counter() - started) * 1000.0
            if trace is not None:
                trace.emit(
                    "fusion.assist.segment_fuse.result",
                    fusion_iteration=fusion_iteration,
                    segment_index=segment_index,
                    segment_goal=segment_goal.as_dict(),
                    fuser_member_id=fuser.id,
                    status="fallback",
                    applied=False,
                    fallback_reason=str(exc),
                    fuser_seed_candidate_id=seed_candidate.index,
                    conservative_fallback_used=True,
                    fused_advice=(
                        conservative_fallback
                        if trace.settings.include_candidate_text
                        else None
                    ),
                    latency_ms=latency_ms,
                )
            return _FusedAdviceResult(
                text=conservative_fallback,
                usage=_CallUsage(),
                fuser_member=fuser,
                member=None,
                applied=False,
                seed_candidate=seed_candidate,
                fallback_reason=str(exc),
                conservative_fallback_used=True,
                latency_ms=latency_ms,
            )

        usage = _CallUsage.from_harness(harness_result.usage)
        text = harness_result.text.strip()
        applied = bool(text)
        fallback_reason = "" if applied else "empty_segment_fusion"
        final_text = text if applied else conservative_fallback
        latency_ms = (time.perf_counter() - started) * 1000.0
        if trace is not None:
            trace.emit(
                "fusion.assist.segment_fuse.result",
                fusion_iteration=fusion_iteration,
                segment_index=segment_index,
                segment_goal=segment_goal.as_dict(),
                fuser_member_id=fuser.id,
                status="ok" if applied else "fallback",
                applied=applied,
                fallback_reason=fallback_reason,
                fuser_seed_candidate_id=seed_candidate.index,
                conservative_fallback_used=not applied,
                fused_advice=(
                    final_text if trace.settings.include_candidate_text else None
                ),
                usage=usage.as_dict(),
                latency_ms=latency_ms,
                harness=harness_result.trace_payload(),
            )
            trace.record_usage(fuser.id, stage="fusion", usage=usage.as_dict())
        return _FusedAdviceResult(
            text=final_text,
            usage=usage,
            fuser_member=fuser,
            member=fuser if applied else None,
            applied=applied,
            seed_candidate=seed_candidate,
            fallback_reason=fallback_reason,
            conservative_fallback_used=not applied,
            latency_ms=latency_ms,
        )

    async def _stitch_fused_advices(
        self,
        fusion_iteration: int,
        messages: list[Message],
        fused_segments: list[_FusedAdviceSegment],
        base_config: ChatConfig,
        *,
        trace: FusionTraceRecorder | None,
    ) -> tuple[str, _CallUsage]:
        non_empty = [segment for segment in fused_segments if segment.text.strip()]
        if not non_empty:
            return "", _CallUsage()
        if len(non_empty) == 1:
            text = _format_fused_segments(non_empty)
            if trace is not None:
                trace.emit(
                    "fusion.assist.stitch.skipped",
                    fusion_iteration=fusion_iteration,
                    reason="single_segment",
                    final_advice=(
                        text if trace.settings.include_candidate_text else None
                    ),
                )
            return text, _CallUsage()

        editor = _stitch_editor_member(non_empty)
        if editor is None:
            text = _format_fused_segments(non_empty)
            if trace is not None:
                trace.emit(
                    "fusion.assist.stitch.skipped",
                    fusion_iteration=fusion_iteration,
                    reason="no_model_authored_segments",
                    final_advice=(
                        text if trace.settings.include_candidate_text else None
                    ),
                )
            return text, _CallUsage()
        prompt = self._assist_stitch_prompt(messages, non_empty)
        call_config = self._call_config(
            base_config,
            system_suffix=_ASSIST_STITCH_SYSTEM,
            max_tokens=_advice_stitch_max_tokens(self._step_max_tokens, non_empty),
            temperature=0.0,
            model_capabilities=editor.model_capabilities,
        )
        if trace is not None:
            trace.emit(
                "fusion.assist.stitch.request",
                fusion_iteration=fusion_iteration,
                editor_member_id=editor.id,
                selected_segment_count=len(non_empty),
                selected_segments=[
                    {
                        "segment_index": segment.goal.index,
                        "title": segment.goal.title,
                        "member_id": segment.member.id if segment.member else None,
                        "chars": len(segment.text),
                    }
                    for segment in non_empty
                ],
                request=(
                    _trace_request(call_config, [Message(role="user", content=prompt)])
                    if trace.settings.include_prompts
                    else None
                ),
            )

        fallback = _format_fused_segments(non_empty)
        started = time.perf_counter()
        try:
            harness_result = await collect_visible_text_with_harness(
                editor.provider,
                [Message(role="user", content=prompt)],
                config=call_config,
            )
        except Exception as exc:  # noqa: BLE001 - stitch failure falls back to concat.
            if trace is not None:
                trace.emit(
                    "fusion.assist.stitch.result",
                    fusion_iteration=fusion_iteration,
                    editor_member_id=editor.id,
                    status="error",
                    applied=False,
                    fallback_reason=str(exc),
                    final_advice=(
                        fallback if trace.settings.include_candidate_text else None
                    ),
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                )
            return fallback, _CallUsage()

        usage = _CallUsage.from_harness(harness_result.usage)
        text = harness_result.text.strip()
        applied = bool(text)
        final_text = text if applied else fallback
        fallback_reason = "" if applied else "empty_stitch_output"
        if trace is not None:
            trace.emit(
                "fusion.assist.stitch.result",
                fusion_iteration=fusion_iteration,
                editor_member_id=editor.id,
                status="ok" if applied else "fallback",
                applied=applied,
                fallback_reason=fallback_reason,
                final_advice=(
                    final_text if trace.settings.include_candidate_text else None
                ),
                usage=usage.as_dict(),
                latency_ms=(time.perf_counter() - started) * 1000.0,
                harness=harness_result.trace_payload(),
            )
            trace.record_usage(editor.id, stage="stitch", usage=usage.as_dict())
        return final_text, usage

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

    async def _assist_segment_plan(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
        base_config: ChatConfig,
        fusion_iteration: int,
        *,
        trace: FusionTraceRecorder | None,
    ) -> tuple[list[_SegmentGoal], _CallUsage]:
        prompt = self._assist_segment_plan_prompt(messages, tools)
        call_config = self._call_config(
            base_config,
            system_suffix=_ASSIST_SEGMENT_PLANNER_SYSTEM,
            max_tokens=min(self._judge_max_tokens, 768),
            temperature=0.0,
            model_capabilities=base_config.model_capabilities,
        )
        planner_messages = [Message(role="user", content=prompt)]
        if trace is not None:
            trace.emit(
                "fusion.assist.segment_plan.request",
                fusion_iteration=fusion_iteration,
                action_model=self._action_model,
                max_segments=self._assist_max_segments,
                request=(
                    _trace_request(call_config, planner_messages)
                    if trace.settings.include_prompts
                    else None
                ),
            )

        started = time.perf_counter()
        usage = _CallUsage()
        raw_response = ""
        harness_payload: dict[str, Any] | None = None
        error = ""
        try:
            harness_result = await collect_visible_text_with_harness(
                self._action_provider,  # type: ignore[arg-type]
                planner_messages,
                config=call_config,
            )
            usage = _CallUsage.from_harness(harness_result.usage)
            raw_response = harness_result.text
            harness_payload = harness_result.trace_payload()
            parsed = _parse_segment_plan(raw_response, self._assist_max_segments)
        except Exception as exc:  # noqa: BLE001 - planner failure falls back to safe slots.
            error = str(exc)
            parsed = _SegmentPlanParseResult(
                goals=_fallback_segment_plan(self._assist_max_segments),
                mode="planner_error_fallback",
                fallback_used=True,
            )

        if trace is not None:
            trace.emit(
                "fusion.assist.segment_plan.result",
                fusion_iteration=fusion_iteration,
                action_model=self._action_model,
                raw_response=(
                    raw_response if trace.settings.include_candidate_text else None
                ),
                parse_mode=parsed.mode,
                fallback_used=parsed.fallback_used,
                error=error,
                usage=usage.as_dict(),
                latency_ms=(time.perf_counter() - started) * 1000.0,
                harness=harness_payload,
                segments=[goal.as_dict() for goal in parsed.goals],
            )
        return parsed.goals, usage

    def _assist_segment_plan_prompt(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
    ) -> str:
        user_request = _latest_user_text(messages)
        conversation = _serialize_messages(messages[-8:])
        return (
            "Plan semantic hidden-advice segments for your next OpenSquilla "
            "agent-loop call.\n\n"
            f"Maximum segment count: {self._assist_max_segments}\n\n"
            f"Original user request:\n{user_request or '[not available]'}\n\n"
            f"Recent conversation and tool state:\n{conversation}\n\n"
            f"Available tools for your later action call:\n{_tool_summary(tools)}\n\n"
            "Return strict JSON only. Each segment needs title, instruction, "
            "and target_words. The segment list is the complete plan for this "
            "one upcoming action-model call."
        )

    def _assist_draft_prompt(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
        selected_advices: list[str],
        segment_goal: _SegmentGoal,
    ) -> str:
        user_request = _latest_user_text(messages)
        conversation = _serialize_messages(messages[-8:])
        tool_summary = _tool_summary(tools)
        previous = "\n\n".join(selected_advices).strip()
        previous_block = (
            f"\nPreviously selected Fusion Assist advice in this turn:\n{previous}\n"
            if previous
            else ""
        )
        return (
            "You are advising the OpenSquilla action model before its next "
            "agent-loop LLM call. The advice is hidden from the user.\n\n"
            "Current hidden advice segment:\n"
            f"- Title: {segment_goal.title}\n"
            f"- Instruction: {segment_goal.instruction}\n"
            f"- Target length: {segment_goal.target_words}\n\n"
            f"Original user request:\n{user_request or '[not available]'}\n\n"
            f"Recent conversation and tool state:\n{conversation}\n\n"
            f"Available tools for the action model:\n{tool_summary}\n"
            f"{previous_block}\n"
            "Produce only this segment of hidden advice. Do not cover other "
            "segment topics unless required for coherence."
        )

    def _assist_verification_prompt(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
        selected_advices: list[str],
        segment_goal: _SegmentGoal,
        labelled_candidates: list[tuple[str, str]],
    ) -> str:
        conversation = _serialize_messages(messages[-8:])
        previous = "\n\n".join(selected_advices).strip()
        previous_block = (
            f"\nPreviously selected Fusion Assist advice in this turn:\n{previous}\n"
            if previous
            else ""
        )
        candidate_blocks = "\n\n".join(
            f"Candidate {label}:\n{text or '[empty advice]'}"
            for label, text in labelled_candidates
        )
        labels = ", ".join(label for label, _text in labelled_candidates)
        return (
            f"Recent conversation and tool state:\n{conversation}\n\n"
            f"Available tools for the action model:\n{_tool_summary(tools)}\n"
            f"{previous_block}\n"
            "Current hidden advice segment:\n"
            f"{segment_goal.title}: {segment_goal.instruction}\n\n"
            f"Score these anonymous hidden advice candidates for this segment "
            f"of the next action-model agent-loop step. Use only these labels: {labels}. "
            "Higher is better. Prefer advice that helps the action model decide "
            "whether to call a tool, interpret tool results, or answer now.\n\n"
            f"{candidate_blocks}\n\n"
            'Return JSON exactly like {"scores":{"A":0.9,"B":0.4},"ranking":["A","B"]}.'
        )

    def _assist_segment_fusion_prompt(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
        selected_advices: list[str],
        segment_goal: _SegmentGoal,
        candidates: list[_Candidate],
        scores: dict[int, float],
    ) -> str:
        conversation = _serialize_messages(messages[-8:])
        previous = "\n\n".join(
            advice.strip() for advice in selected_advices if advice.strip()
        )
        previous_block = (
            f"\nPreviously fused hidden advice segments:\n{previous}\n"
            if previous
            else ""
        )
        labels = list(string.ascii_uppercase[: len(candidates)])
        candidate_blocks = "\n\n".join(
            (
                f"Candidate {labels[index]} "
                f"(score {scores.get(candidate.index, 0.0):.3f}):\n"
                f"{candidate.text or '[empty advice]'}"
            )
            for index, candidate in enumerate(candidates)
        )
        return (
            "Fuse the anonymous hidden advice candidates for one semantic segment.\n"
            "Use only candidate content. Do not add facts, sources, numbers, "
            "tool results, or tool arguments that are not already present. "
            "If candidates look incomplete or truncated, repair them into a "
            "short complete advice segment by preserving only supported guidance "
            "and explicitly noting uncertainty or missing evidence when needed.\n\n"
            "Current hidden advice segment:\n"
            f"{segment_goal.title}: {segment_goal.instruction}\n"
            f"Target length: {segment_goal.target_words}\n\n"
            f"Recent conversation and tool state:\n{conversation}\n\n"
            f"Available tools for the action model:\n{_tool_summary(tools)}\n"
            f"{previous_block}\n"
            f"{candidate_blocks}\n\n"
            "Return only the fused hidden advice segment."
        )

    def _assist_stitch_prompt(
        self,
        messages: list[Message],
        fused_segments: list[_FusedAdviceSegment],
    ) -> str:
        conversation = _serialize_messages(messages[-8:])
        segment_blocks = "\n\n".join(
            (
                f"Selected hidden advice segment {index + 1} "
                f"({segment.goal.title}):\n{segment.text.strip()}"
            )
            for index, segment in enumerate(fused_segments)
        )
        return (
            "Stitch the selected hidden advice segments into one concise hidden "
            "advice block for the action model.\n"
            "Only remove duplication, smooth transitions, and normalize format. "
            "Do not add facts, sources, numbers, tool results, or tool arguments.\n\n"
            f"Recent conversation for context only:\n{conversation}\n\n"
            f"{segment_blocks}\n\n"
            "Return only the stitched hidden advice block."
        )

    def _anonymous_scores(
        self,
        candidates: list[_Candidate],
        verifications: list[_Verification],
    ) -> dict[int, float]:
        valid_ids = {candidate.index for candidate in candidates}
        scores = {candidate.index: 0.0 for candidate in candidates}
        for verification in verifications:
            normalized = _normalize_scores(verification.scores, valid_ids)
            for candidate_id, score in normalized.items():
                scores[candidate_id] += score
        divisor = max(len(verifications), 1)
        return {candidate_id: score / divisor for candidate_id, score in scores.items()}

    def _select_candidate(
        self,
        candidates: list[_Candidate],
        scores: dict[int, float],
    ) -> _Candidate:
        return max(
            candidates,
            key=lambda candidate: (
                scores.get(candidate.index, 0.0),
                -candidate.index,
            ),
        )

    def _assist_trace_for(self, config: ChatConfig) -> FusionTraceRecorder | None:
        if self._assist_trace is None:
            self._assist_trace = self._new_trace(config)
        return self._assist_trace

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
                }
                for member in self._members
            ],
            action_model=self._action_model,
            assist_max_segments=self._assist_max_segments,
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


def _messages_with_fusion_advice(messages: list[Message], advice: str) -> list[Message]:
    cleaned = advice.strip()
    if not cleaned:
        return list(messages)
    return [
        *messages,
        Message(
            role="user",
            content=(
                f"{_ASSIST_INJECTION_PREFIX}\n"
                f"{cleaned}\n\n"
                "This advice is not user-visible. You are the only model allowed "
                "to call tools. Use or ignore the advice as appropriate."
            ),
        ),
    ]


def _format_fused_segments(segments: list[_FusedAdviceSegment]) -> str:
    blocks: list[str] = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        blocks.append(f"{segment.goal.title}:\n{text}")
    return "\n\n".join(blocks).strip()


def _stitch_editor_member(segments: list[_FusedAdviceSegment]) -> FusionMember | None:
    selected_chars: dict[str, int] = {}
    members_by_id: dict[str, FusionMember] = {}
    order: dict[str, int] = {}
    for index, segment in enumerate(segments):
        if segment.member is None:
            continue
        member_id = segment.member.id
        selected_chars[member_id] = selected_chars.get(member_id, 0) + len(segment.text)
        members_by_id[member_id] = segment.member
        order.setdefault(member_id, index)
    if not selected_chars:
        return None
    editor_id = max(
        selected_chars,
        key=lambda member_id: (
            selected_chars.get(member_id, 0),
            -order.get(member_id, 0),
        ),
    )
    return members_by_id[editor_id]


def _conservative_advice_fallback(segment_goal: _SegmentGoal) -> str:
    return (
        f"For '{segment_goal.title}', Fusion could not form a reliable complete "
        "advice segment. The action model should rely on the original user "
        "request, current conversation, and verified tool results; avoid "
        "assuming unsupported facts, numbers, sources, or tool outputs."
    )


def _advice_stitch_max_tokens(
    step_max_tokens: int,
    segments: list[_FusedAdviceSegment],
) -> int:
    return max(step_max_tokens, min(4096, step_max_tokens * max(len(segments), 1)))


def _tool_summary(tools: list[ToolDefinition] | None) -> str:
    if not tools:
        return "[no tools available]"
    rows: list[str] = []
    for tool in tools[:30]:
        description = str(getattr(tool, "description", "") or "").strip()
        rows.append(f"- {tool.name}: {description or 'no description'}")
    if len(tools) > 30:
        rows.append(f"- ... {len(tools) - 30} additional tools omitted")
    return "\n".join(rows)


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


def _parse_segment_plan(raw: str, max_segments: int) -> _SegmentPlanParseResult:
    try:
        payload = json.loads(_extract_json_value(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return _SegmentPlanParseResult(
            goals=_fallback_segment_plan(max_segments),
            mode="json_parse_fallback",
            fallback_used=True,
        )
    raw_segments = payload.get("segments") if isinstance(payload, dict) else payload
    if not isinstance(raw_segments, list):
        return _SegmentPlanParseResult(
            goals=_fallback_segment_plan(max_segments),
            mode="missing_segments_fallback",
            fallback_used=True,
        )

    goals: list[_SegmentGoal] = []
    seen_titles: set[str] = set()
    for item in raw_segments:
        if not isinstance(item, dict):
            continue
        title = _clean_segment_field(item.get("title"), limit=64)
        instruction = _clean_segment_field(item.get("instruction"), limit=360)
        target_words = _clean_segment_field(item.get("target_words"), limit=40)
        if not title or not instruction:
            continue
        normalized_title = title.casefold()
        if normalized_title in seen_titles:
            continue
        seen_titles.add(normalized_title)
        goals.append(
            _SegmentGoal(
                index=len(goals),
                title=title,
                instruction=instruction,
                target_words=target_words or "30-90 words",
            )
        )
        if len(goals) >= max(1, max_segments):
            break

    if not goals:
        return _SegmentPlanParseResult(
            goals=_fallback_segment_plan(max_segments),
            mode="empty_segments_fallback",
            fallback_used=True,
        )
    return _SegmentPlanParseResult(
        goals=goals,
        mode="json_segments",
        fallback_used=False,
    )


def _fallback_segment_plan(max_segments: int) -> list[_SegmentGoal]:
    fallback = [
        (
            "Task state",
            "Summarize the request, current context, and what the next action-model call must decide.",
        ),
        (
            "Next action",
            "Advise whether to answer now or gather more context, without producing tool arguments.",
        ),
        (
            "Risks",
            "Call out uncertainty, missing evidence, and constraints the action model should preserve.",
        ),
    ]
    return [
        _SegmentGoal(
            index=index,
            title=title,
            instruction=instruction,
            target_words="30-90 words",
        )
        for index, (title, instruction) in enumerate(fallback[: max(1, max_segments)])
    ]


def _clean_segment_field(value: Any, *, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].strip() or text[:limit].strip()


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


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def _extract_json_value(text: str) -> str:
    object_start = text.find("{")
    if object_start >= 0:
        end = text.rfind("}")
        if end > object_start:
            return text[object_start : end + 1]
        return text[object_start:]
    array_start = text.find("[")
    if array_start < 0:
        return text
    end = text.rfind("]")
    if end > array_start:
        return text[array_start : end + 1]
    return text[array_start:]


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
