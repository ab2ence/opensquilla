from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator

from opensquilla.fusion_reply.provider import FusionMember, FusionReplyProvider
from opensquilla.fusion_reply.trace import FusionTraceSettings
from opensquilla.provider.types import (
    ChatConfig,
    DoneEvent,
    Message,
    StreamEvent,
    TextDeltaEvent,
    ToolDefinition,
    ToolInputSchema,
    ToolUseDeltaEvent,
    ToolUseEndEvent,
    ToolUseStartEvent,
)


class _FakeProvider:
    provider_name = "fake"

    def __init__(self, model: str, draft: str | list[str]) -> None:
        self.model = model
        self.drafts = [draft] if isinstance(draft, str) else list(draft)
        self._draft_index = 0
        self.calls: list[tuple[list[Message], object, ChatConfig | None]] = []

    async def chat(
        self,
        messages: list[Message],
        tools: object = None,
        config: ChatConfig | None = None,
    ) -> AsyncIterator[StreamEvent]:
        self.calls.append((messages, tools, config))
        system = (config.system if config else "") or ""
        if "anonymous specem verifier" in system.lower():
            prompt = str(messages[-1].content)
            scores = _scores_from_prompt(prompt)
            ranking = [
                label
                for label, _score in sorted(
                    scores.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ]
            yield TextDeltaEvent(text=json.dumps({"scores": scores, "ranking": ranking}))
            yield DoneEvent(input_tokens=1, output_tokens=1, model=f"judge:{self.model}")
            return

        draft = self.drafts[min(self._draft_index, len(self.drafts) - 1)]
        self._draft_index += 1
        yield TextDeltaEvent(text=draft)
        yield DoneEvent(input_tokens=2, output_tokens=3, model=self.model)

    async def list_models(self) -> list[object]:
        return []


class _ActionProvider:
    provider_name = "action"

    def __init__(self, *, text: str = "", use_tool: bool = False) -> None:
        self.text = text
        self.use_tool = use_tool
        self.calls: list[tuple[list[Message], object, ChatConfig | None]] = []

    async def chat(
        self,
        messages: list[Message],
        tools: object = None,
        config: ChatConfig | None = None,
    ) -> AsyncIterator[StreamEvent]:
        self.calls.append((messages, tools, config))
        if self.text:
            yield TextDeltaEvent(text=self.text)
        if self.use_tool:
            yield ToolUseStartEvent(tool_use_id="tool-1", tool_name="search")
            yield ToolUseDeltaEvent(tool_use_id="tool-1", json_fragment='{"q":"glm"}')
            yield ToolUseEndEvent(
                tool_use_id="tool-1",
                tool_name="search",
                arguments={"q": "glm"},
            )
            yield DoneEvent(
                stop_reason="tool_use",
                input_tokens=5,
                output_tokens=6,
                model="action",
            )
            return
        yield DoneEvent(stop_reason="end_turn", input_tokens=5, output_tokens=6, model="action")


class _ReasoningOnlyThenDraftProvider(_FakeProvider):
    def __init__(self, model: str, recovered_draft: str) -> None:
        super().__init__(model, recovered_draft)
        self._reasoning_only_sent = False

    async def chat(
        self,
        messages: list[Message],
        tools: object = None,
        config: ChatConfig | None = None,
    ) -> AsyncIterator[StreamEvent]:
        system = (config.system if config else "") or ""
        if "anonymous specem verifier" in system.lower():
            async for event in super().chat(messages, tools=tools, config=config):
                yield event
            return

        self.calls.append((messages, tools, config))
        if not self._reasoning_only_sent:
            self._reasoning_only_sent = True
            yield DoneEvent(
                stop_reason="stop",
                input_tokens=35_000,
                output_tokens=2,
                reasoning_tokens=2,
                reasoning_content="internal reasoning only",
                model=self.model,
            )
            return

        yield TextDeltaEvent(text=self.drafts[0])
        yield DoneEvent(input_tokens=4, output_tokens=5, model=self.model)


def _tool_defs() -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="search",
            description="Search the web",
            input_schema=ToolInputSchema(properties={"q": {"type": "string"}}),
        )
    ]


def _scores_from_prompt(prompt: str) -> dict[str, float]:
    labelled: list[tuple[str, str]] = []
    for match in re.finditer(
        r"Candidate ([A-Z]):\n(.*?)(?=\n\nCandidate [A-Z]:|\n\nReturn JSON|\Z)",
        prompt,
        flags=re.S,
    ):
        labelled.append((match.group(1), match.group(2)))
    return {
        label: (1.0 if "GOOD" in text else 0.1)
        for label, text in labelled
    }


def test_fusion_reply_selects_weighted_anonymous_winner() -> None:
    asyncio.run(_run_fusion_reply_selection_case())


async def _run_fusion_reply_selection_case() -> None:
    weak = _FakeProvider("weak", "BAD candidate")
    strong = _FakeProvider("strong", "GOOD candidate")
    provider = FusionReplyProvider(
        [
            FusionMember("weak", "fake", "weak", weak),
            FusionMember("strong", "fake", "strong", strong),
        ],
        max_rounds=1,
    )

    text = ""
    done: DoneEvent | None = None
    async for event in provider.chat([Message(role="user", content="answer")], tools=[]):
        if isinstance(event, TextDeltaEvent):
            text += event.text
        elif isinstance(event, DoneEvent):
            done = event

    assert text == "GOOD candidate"
    assert done is not None
    assert done.model == "fusion:weak+strong"
    assert done.input_tokens == 6
    assert done.output_tokens == 8
    assert all(call[1] is None for call in [*weak.calls, *strong.calls])


def test_fusion_reply_writes_detailed_trace_and_compact_summary(tmp_path) -> None:
    asyncio.run(_run_fusion_trace_case(tmp_path))


async def _run_fusion_trace_case(tmp_path) -> None:
    weak = _FakeProvider("weak", "BAD candidate")
    strong = _FakeProvider("strong", "GOOD candidate")
    provider = FusionReplyProvider(
        [
            FusionMember("weak", "fake", "weak", weak),
            FusionMember("strong", "fake", "strong", strong),
        ],
        max_rounds=1,
        trace_settings=FusionTraceSettings(log_dir=str(tmp_path)),
    )

    done: DoneEvent | None = None
    async for event in provider.chat(
        [Message(role="user", content="answer")],
        tools=[],
        config=ChatConfig(metadata={"fusion_trace_session_key": "session-1"}),
    ):
        if isinstance(event, DoneEvent):
            done = event

    assert done is not None
    summary = done.metadata["fusion_summary"]
    assert summary["algorithm"] == "specem_weighted_pairwise"
    members = summary["output_contribution"]["members"]
    assert members[0]["member_id"] == "strong"
    assert members[0]["share"] == 1.0
    assert members[1]["member_id"] == "weak"
    assert members[1]["share"] == 0.0

    trace_files = list(tmp_path.glob("fusion-traces-*.jsonl"))
    assert len(trace_files) == 1
    events = [
        json.loads(line)
        for line in trace_files[0].read_text(encoding="utf-8").splitlines()
    ]
    kinds = [event["event"] for event in events]
    assert "fusion.draft.result" in kinds
    assert "fusion.verify.result" in kinds
    assert "fusion.aggregate" in kinds
    assert "fusion.select" in kinds
    assert kinds[-1] == "fusion.end"
    assert all(event["session_key"] == "session-1" for event in events)
    verify_result = next(event for event in events if event["event"] == "fusion.verify.result")
    assert verify_result["label_scores"]
    assert verify_result["normalized_scores"]
    assert verify_result["parse_mode"] == "scores"


def test_fusion_reply_harness_retries_reasoning_only_draft(tmp_path) -> None:
    asyncio.run(_run_reasoning_only_draft_retry_case(tmp_path))


async def _run_reasoning_only_draft_retry_case(tmp_path) -> None:
    recovered = _ReasoningOnlyThenDraftProvider("deepseek", "GOOD recovered candidate")
    weak = _FakeProvider("weak", "BAD candidate")
    provider = FusionReplyProvider(
        [
            FusionMember("deepseek", "fake", "deepseek", recovered),
            FusionMember("weak", "fake", "weak", weak),
        ],
        max_rounds=1,
        trace_settings=FusionTraceSettings(log_dir=str(tmp_path)),
    )

    text = ""
    async for event in provider.chat([Message(role="user", content="answer")], tools=[]):
        if isinstance(event, TextDeltaEvent):
            text += event.text

    assert text == "GOOD recovered candidate"
    assert len(recovered.calls) == 3  # reasoning-only draft, retried draft, verifier

    trace_files = list(tmp_path.glob("fusion-traces-*.jsonl"))
    events = [
        json.loads(line)
        for line in trace_files[0].read_text(encoding="utf-8").splitlines()
    ]
    draft_result = next(
        event
        for event in events
        if event["event"] == "fusion.draft.result"
        and event["member_id"] == "deepseek"
    )
    harness = draft_result["harness"]
    assert harness["classification"] == "ok"
    assert harness["attempt_count"] == 2
    assert [attempt["classification"] for attempt in harness["attempts"]] == [
        "reasoning_only",
        "ok",
    ]
    assert harness["warnings"][0]["code"] == "provider_reasoning_only_retry"


def test_fusion_reply_runs_multiple_segments_before_accepting_done(tmp_path) -> None:
    asyncio.run(_run_multi_segment_done_gate_case(tmp_path))


async def _run_multi_segment_done_gate_case(tmp_path) -> None:
    weak = _FakeProvider(
        "weak",
        ["BAD first segment <fusion_done/>", "BAD second segment <fusion_done/>"],
    )
    strong = _FakeProvider(
        "strong",
        ["GOOD first segment <fusion_done/>", "GOOD second segment <fusion_done/>"],
    )
    provider = FusionReplyProvider(
        [
            FusionMember("weak", "fake", "weak", weak),
            FusionMember("strong", "fake", "strong", strong),
        ],
        max_rounds=2,
        min_rounds=2,
        trace_settings=FusionTraceSettings(log_dir=str(tmp_path)),
    )

    text = ""
    done: DoneEvent | None = None
    async for event in provider.chat([Message(role="user", content="answer")], tools=[]):
        if isinstance(event, TextDeltaEvent):
            text += event.text
        elif isinstance(event, DoneEvent):
            done = event

    assert text == "GOOD first segment\n\nGOOD second segment"
    assert done is not None
    summary = done.metadata["fusion_summary"]
    assert summary["rounds_completed"] == 2
    members = summary["output_contribution"]["members"]
    assert members[0]["member_id"] == "strong"
    assert members[0]["selected_segments"] == 2
    assert members[0]["share"] == 1.0

    trace_files = list(tmp_path.glob("fusion-traces-*.jsonl"))
    events = [
        json.loads(line)
        for line in trace_files[0].read_text(encoding="utf-8").splitlines()
    ]
    plan = next(event for event in events if event["event"] == "fusion.segment_plan")
    assert plan["mode"] == "adaptive_heuristic"
    assert plan["effective_min_rounds"] == 2
    assert len(plan["segments"]) == 2
    selects = [event for event in events if event["event"] == "fusion.select"]
    assert len(selects) == 2
    assert selects[0]["candidate_is_done"] is True
    assert selects[0]["is_done"] is False
    assert selects[0]["done_ignored_reason"] == "min_rounds_not_reached"
    assert selects[1]["candidate_is_done"] is True
    assert selects[1]["is_done"] is True


def test_fusion_reply_adaptive_segment_plan_expands_report_tasks(tmp_path) -> None:
    asyncio.run(_run_adaptive_report_segment_plan_case(tmp_path))


async def _run_adaptive_report_segment_plan_case(tmp_path) -> None:
    weak = _FakeProvider(
        "weak",
        [f"BAD report segment {index} <fusion_done/>" for index in range(1, 7)],
    )
    strong = _FakeProvider(
        "strong",
        [f"GOOD report segment {index} <fusion_done/>" for index in range(1, 7)],
    )
    provider = FusionReplyProvider(
        [
            FusionMember("weak", "fake", "weak", weak),
            FusionMember("strong", "fake", "strong", strong),
        ],
        max_rounds=6,
        min_rounds=2,
        trace_settings=FusionTraceSettings(log_dir=str(tmp_path)),
    )

    text = ""
    done: DoneEvent | None = None
    messages = [
        Message(
            role="user",
            content=(
                "[Request context for this turn]\n"
                "This request-scoped context is not a user request."
            ),
        ),
        Message(
            role="user",
            content=(
                "写一份公司股价研报，包含新闻、估值和风险\n"
                "[Runtime context for this turn]\n"
                "Current local date/time: 2026-06-23T14:24+08:00"
            ),
        ),
        Message(
            role="assistant",
            content=[
                {
                    "type": "tool_use",
                    "id": "tool-1",
                    "name": "web_search",
                    "input": {"q": "glm"},
                }
            ],
        ),
        Message(
            role="user",
            content=[
                {
                    "type": "tool_result",
                    "tool_use_id": "tool-1",
                    "content": "result",
                }
            ],
        ),
    ]
    async for event in provider.chat(messages, tools=[]):
        if isinstance(event, TextDeltaEvent):
            text += event.text
        elif isinstance(event, DoneEvent):
            done = event

    assert "GOOD report segment 1" in text
    assert "GOOD report segment 6" in text
    assert done is not None
    assert done.metadata["fusion_summary"]["rounds_completed"] == 6

    trace_files = list(tmp_path.glob("fusion-traces-*.jsonl"))
    events = [
        json.loads(line)
        for line in trace_files[0].read_text(encoding="utf-8").splitlines()
    ]
    plan = next(event for event in events if event["event"] == "fusion.segment_plan")
    assert plan["mode"] == "adaptive_heuristic"
    assert plan["effective_min_rounds"] == 6
    assert [segment["title"] for segment in plan["segments"]] == [
        "Conclusion",
        "Evidence",
        "Drivers",
        "Estimate",
        "Risks",
        "Final synthesis",
    ]


def test_fusion_reply_passes_tool_calls_through_action_model() -> None:
    asyncio.run(_run_action_tool_passthrough_case())


async def _run_action_tool_passthrough_case() -> None:
    action = _ActionProvider(text="Looking first. ", use_tool=True)
    weak = _FakeProvider("weak", "BAD candidate")
    strong = _FakeProvider("strong", "GOOD candidate")
    provider = FusionReplyProvider(
        [
            FusionMember("weak", "fake", "weak", weak),
            FusionMember("strong", "fake", "strong", strong),
        ],
        action_provider=action,
        action_member_id="weak",
        max_rounds=1,
    )

    events: list[StreamEvent] = []
    async for event in provider.chat(
        [Message(role="user", content="research")],
        tools=_tool_defs(),
    ):
        events.append(event)

    assert [type(event) for event in events] == [
        TextDeltaEvent,
        ToolUseStartEvent,
        ToolUseDeltaEvent,
        ToolUseEndEvent,
        DoneEvent,
    ]
    assert action.calls and action.calls[0][1] == _tool_defs()
    assert weak.calls == []
    assert strong.calls == []
    assert isinstance(events[-1], DoneEvent)
    assert events[-1].stop_reason == "tool_use"


def test_fusion_reply_fuses_only_after_action_final_text() -> None:
    asyncio.run(_run_action_final_fusion_case())


async def _run_action_final_fusion_case() -> None:
    action = _ActionProvider(text="BAD action answer", use_tool=False)
    weak = _FakeProvider("weak", "SHOULD NOT DRAFT")
    strong = _FakeProvider("strong", "GOOD fused answer")
    provider = FusionReplyProvider(
        [
            FusionMember("weak", "fake", "weak", weak),
            FusionMember("strong", "fake", "strong", strong),
        ],
        action_provider=action,
        action_member_id="weak",
        max_rounds=1,
    )

    text = ""
    done: DoneEvent | None = None
    async for event in provider.chat(
        [Message(role="user", content="answer with evidence")],
        tools=_tool_defs(),
    ):
        if isinstance(event, TextDeltaEvent):
            text += event.text
        elif isinstance(event, DoneEvent):
            done = event

    assert text == "GOOD fused answer"
    assert done is not None
    assert done.model == "fusion:weak+strong"
    assert action.calls and action.calls[0][1] == _tool_defs()
    assert len(weak.calls) == 1  # verifier only; the action text seeded weak's draft
    assert len(strong.calls) == 2  # draft + verifier
    assert all(call[1] is None for call in [*weak.calls, *strong.calls])


def test_fusion_reply_multi_segment_does_not_seed_action_full_answer() -> None:
    asyncio.run(_run_action_final_multi_segment_no_seed_case())


async def _run_action_final_multi_segment_no_seed_case() -> None:
    action = _ActionProvider(text="GOOD action full answer", use_tool=False)
    weak = _FakeProvider(
        "weak",
        ["BAD first segment <fusion_done/>", "BAD second segment <fusion_done/>"],
    )
    strong = _FakeProvider(
        "strong",
        ["GOOD first segment <fusion_done/>", "GOOD second segment <fusion_done/>"],
    )
    provider = FusionReplyProvider(
        [
            FusionMember("weak", "fake", "weak", weak),
            FusionMember("strong", "fake", "strong", strong),
        ],
        action_provider=action,
        action_member_id="weak",
        max_rounds=2,
        min_rounds=2,
    )

    text = ""
    async for event in provider.chat(
        [Message(role="user", content="answer with evidence")],
        tools=_tool_defs(),
    ):
        if isinstance(event, TextDeltaEvent):
            text += event.text

    assert text == "GOOD first segment\n\nGOOD second segment"
    assert action.calls and action.calls[0][1] == _tool_defs()
    assert len(weak.calls) == 4  # two drafts + two verifications; no action seed candidate
    assert len(strong.calls) == 4
    assert all(call[1] is None for call in [*weak.calls, *strong.calls])
