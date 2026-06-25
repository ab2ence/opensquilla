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

    def __init__(
        self,
        model: str,
        draft: str | list[str],
        *,
        fuse_text: str | None = None,
    ) -> None:
        self.model = model
        self.drafts = [draft] if isinstance(draft, str) else list(draft)
        self.fuse_text = fuse_text
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
        if "constrained segment fuser" in system.lower():
            if self.fuse_text is None:
                yield TextDeltaEvent(text=_fuse_from_prompt(str(messages[-1].content)))
            elif self.fuse_text:
                yield TextDeltaEvent(text=self.fuse_text)
            yield DoneEvent(input_tokens=3, output_tokens=4, model=f"fuse:{self.model}")
            return
        if "constrained final stitch editor" in system.lower():
            yield TextDeltaEvent(text=_stitch_from_prompt(str(messages[-1].content)))
            yield DoneEvent(input_tokens=4, output_tokens=5, model=f"stitch:{self.model}")
            return

        draft = self.drafts[min(self._draft_index, len(self.drafts) - 1)]
        self._draft_index += 1
        yield TextDeltaEvent(text=draft)
        yield DoneEvent(input_tokens=2, output_tokens=3, model=self.model)

    async def list_models(self) -> list[object]:
        return []


class _FailingProvider:
    provider_name = "fake"

    def __init__(self, model: str) -> None:
        self.model = model
        self.calls: list[tuple[list[Message], object, ChatConfig | None]] = []

    async def chat(
        self,
        messages: list[Message],
        tools: object = None,
        config: ChatConfig | None = None,
    ) -> AsyncIterator[StreamEvent]:
        self.calls.append((messages, tools, config))
        raise RuntimeError(f"{self.model} unavailable")

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
        if _is_segment_planner_config(config):
            yield TextDeltaEvent(text=_planner_json())
            yield DoneEvent(input_tokens=2, output_tokens=2, model="action:planner")
            return
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


class _SequencedActionProvider:
    provider_name = "action"

    def __init__(self) -> None:
        self.calls: list[tuple[list[Message], object, ChatConfig | None]] = []
        self.action_calls = 0

    async def chat(
        self,
        messages: list[Message],
        tools: object = None,
        config: ChatConfig | None = None,
    ) -> AsyncIterator[StreamEvent]:
        self.calls.append((messages, tools, config))
        if _is_segment_planner_config(config):
            yield TextDeltaEvent(text=_planner_json())
            yield DoneEvent(input_tokens=2, output_tokens=2, model="action:planner")
            return
        self.action_calls += 1
        if self.action_calls == 1:
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
        yield TextDeltaEvent(text="final action answer")
        yield DoneEvent(stop_reason="end_turn", input_tokens=7, output_tokens=8, model="action")


def _tool_defs() -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="search",
            description="Search the web",
            input_schema=ToolInputSchema(properties={"q": {"type": "string"}}),
        )
    ]


def _planner_json() -> str:
    return json.dumps(
        {
            "segments": [
                {
                    "title": "Task state",
                    "instruction": "Summarize the task and current agent state.",
                    "target_words": "30-90 words",
                },
                {
                    "title": "Tool plan",
                    "instruction": "Advise whether the next action call should use tools.",
                    "target_words": "30-90 words",
                },
                {
                    "title": "Evidence constraints",
                    "instruction": "Identify evidence, constraints, and uncertainty to preserve.",
                    "target_words": "30-90 words",
                },
            ]
        }
    )


def _is_segment_planner_config(config: ChatConfig | None) -> bool:
    return "semantic segment planner" in ((config.system if config else "") or "").lower()


def _actual_action_calls(
    calls: list[tuple[list[Message], object, ChatConfig | None]],
) -> list[tuple[list[Message], object, ChatConfig | None]]:
    return [call for call in calls if not _is_segment_planner_config(call[2])]


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


def _fuse_from_prompt(prompt: str) -> str:
    candidates: list[str] = []
    for match in re.finditer(
        r"Candidate [A-Z](?: \(score [0-9.]+\))?:\n(.*?)(?=\n\nCandidate [A-Z]|\n\nReturn only|\Z)",
        prompt,
        flags=re.S,
    ):
        text = match.group(1).strip()
        if text:
            candidates.append(text)
    useful = [text for text in candidates if "GOOD" in text] or candidates
    return "FUSED " + " | ".join(useful)


def _stitch_from_prompt(prompt: str) -> str:
    segments: list[str] = []
    for match in re.finditer(
        r"Selected hidden advice segment \d+ \((.*?)\):\n(.*?)(?=\n\nSelected hidden advice segment \d+|\n\nReturn only|\Z)",
        prompt,
        flags=re.S,
    ):
        title = match.group(1).strip()
        text = match.group(2).strip()
        if text:
            segments.append(f"{title}:\n{text}")
    return "STITCHED\n" + "\n\n".join(segments)


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
        assist_max_segments=1,
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
    action_calls = _actual_action_calls(action.calls)
    assert action_calls and action_calls[0][1] == _tool_defs()
    assert len(weak.calls) == 2
    assert len(strong.calls) == 3
    assert all(call[1] is None for call in [*weak.calls, *strong.calls])
    action_messages = action_calls[0][0]
    assert "Hidden Fusion Assist advice" in str(action_messages[-1].content)
    assert isinstance(events[-1], DoneEvent)
    assert events[-1].stop_reason == "tool_use"
    assert events[-1].metadata["fusion_summary"]["architecture"] == "agent_loop_assist"


def test_fusion_reply_assists_each_agent_loop_iteration() -> None:
    asyncio.run(_run_multi_iteration_assist_case())


async def _run_multi_iteration_assist_case() -> None:
    action = _SequencedActionProvider()
    weak = _FakeProvider("weak", ["BAD first advice", "BAD second advice"])
    strong = _FakeProvider("strong", ["GOOD first advice", "GOOD second advice"])
    provider = FusionReplyProvider(
        [
            FusionMember("weak", "fake", "weak", weak),
            FusionMember("strong", "fake", "strong", strong),
        ],
        action_provider=action,
        assist_max_segments=1,
    )

    first_events: list[StreamEvent] = []
    async for event in provider.chat(
        [Message(role="user", content="research")],
        tools=_tool_defs(),
    ):
        first_events.append(event)

    second_text = ""
    second_done: DoneEvent | None = None
    async for event in provider.chat(
        [
            Message(role="user", content="research"),
            Message(
                role="user",
                content=[
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-1",
                        "content": "search result",
                    }
                ],
            ),
        ],
        tools=_tool_defs(),
    ):
        if isinstance(event, TextDeltaEvent):
            second_text += event.text
        elif isinstance(event, DoneEvent):
            second_done = event

    assert isinstance(first_events[-1], DoneEvent)
    assert first_events[-1].stop_reason == "tool_use"
    assert second_text == "final action answer"
    assert second_done is not None
    assert second_done.metadata["fusion_summary"]["assist_iterations"] == 2
    action_calls = _actual_action_calls(action.calls)
    assert len(action.calls) == 4
    assert len(action_calls) == 2
    assert all(
        "Hidden Fusion Assist advice" in str(call[0][-1].content)
        for call in action_calls
    )
    assert len(weak.calls) == 4
    assert len(strong.calls) == 6
    assert all(call[1] is None for call in [*weak.calls, *strong.calls])


def test_fusion_reply_assists_before_action_final_text() -> None:
    asyncio.run(_run_action_final_assist_case())


async def _run_action_final_assist_case() -> None:
    action = _ActionProvider(text="BAD action answer", use_tool=False)
    weak = _FakeProvider("weak", "SHOULD NOT DRAFT")
    strong = _FakeProvider("strong", "GOOD hidden advice")
    provider = FusionReplyProvider(
        [
            FusionMember("weak", "fake", "weak", weak),
            FusionMember("strong", "fake", "strong", strong),
        ],
        action_provider=action,
        assist_max_segments=1,
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

    assert text == "BAD action answer"
    assert done is not None
    assert done.model == "action"
    action_calls = _actual_action_calls(action.calls)
    assert action_calls and action_calls[0][1] == _tool_defs()
    action_messages = action_calls[0][0]
    assert "Hidden Fusion Assist advice" in str(action_messages[-1].content)
    assert "FUSED GOOD hidden advice" in str(action_messages[-1].content)
    assert "STITCHED" not in str(action_messages[-1].content)
    assert len(weak.calls) == 2
    assert len(strong.calls) == 3
    assert all(call[1] is None for call in [*weak.calls, *strong.calls])
    summary = done.metadata["fusion_summary"]
    assert summary["architecture"] == "agent_loop_assist"
    assert summary["final_answer_author"] == "action_model"
    assert summary["assist_participation"]["members"][0]["selected_advice"] == 1


def test_fusion_reply_assist_failure_is_traced_and_action_continues() -> None:
    asyncio.run(_run_assist_failure_action_continues_case())


async def _run_assist_failure_action_continues_case() -> None:
    action = _ActionProvider(text="action still answers", use_tool=False)
    failing = _FailingProvider("weak")
    provider = FusionReplyProvider(
        [FusionMember("weak", "fake", "weak", failing)],
        action_provider=action,
        assist_max_segments=1,
    )

    text = ""
    done: DoneEvent | None = None
    async for event in provider.chat(
        [Message(role="user", content="answer despite assist failure")],
        tools=_tool_defs(),
    ):
        if isinstance(event, TextDeltaEvent):
            text += event.text
        elif isinstance(event, DoneEvent):
            done = event

    assert text == "action still answers"
    assert done is not None
    assert done.model == "action"
    action_calls = _actual_action_calls(action.calls)
    assert action_calls and action_calls[0][1] == _tool_defs()
    assert "Hidden Fusion Assist advice" not in str(action_calls[0][0][-1].content)
    assert failing.calls and failing.calls[0][1] is None
    summary = done.metadata["fusion_summary"]
    assert summary["architecture"] == "agent_loop_assist"
    assert summary["status"] == "error"
    assert summary["assist_iterations"] == 0


def test_fusion_reply_uses_semantic_advice_segments_fusion_and_stitch(tmp_path) -> None:
    asyncio.run(_run_semantic_advice_segments_case(tmp_path))


async def _run_semantic_advice_segments_case(tmp_path) -> None:
    action = _ActionProvider(text="action final answer", use_tool=False)
    weak = _FakeProvider(
        "weak",
        ["BAD state", "BAD tool plan", "BAD evidence"],
    )
    strong = _FakeProvider(
        "strong",
        ["GOOD state", "GOOD tool plan", "GOOD evidence"],
    )
    provider = FusionReplyProvider(
        [
            FusionMember("weak", "fake", "weak", weak),
            FusionMember("strong", "fake", "strong", strong),
        ],
        action_provider=action,
        assist_max_segments=3,
        trace_settings=FusionTraceSettings(log_dir=str(tmp_path)),
    )

    done: DoneEvent | None = None
    async for event in provider.chat(
        [Message(role="user", content="research this before answering")],
        tools=_tool_defs(),
        config=ChatConfig(metadata={"fusion_trace_session_key": "session-1"}),
    ):
        if isinstance(event, DoneEvent):
            done = event

    assert done is not None
    action_advice = str(_actual_action_calls(action.calls)[0][0][-1].content)
    assert "STITCHED" in action_advice
    assert "Task state:\nFUSED GOOD state" in action_advice
    assert "Tool plan:\nFUSED GOOD tool plan" in action_advice
    assert "Evidence constraints:\nFUSED GOOD evidence" in action_advice
    summary = done.metadata["fusion_summary"]
    members = summary["assist_participation"]["members"]
    assert members[0]["member_id"] == "strong"
    assert members[0]["selected_advice"] == 3
    assert members[0]["fusion_calls"] == 3
    assert members[0]["stitch_calls"] == 1
    assert len(weak.calls) == 6
    assert len(strong.calls) == 10

    trace_files = list(tmp_path.glob("fusion-traces-*.jsonl"))
    assert len(trace_files) == 1
    events = [
        json.loads(line)
        for line in trace_files[0].read_text(encoding="utf-8").splitlines()
    ]
    kinds = [event["event"] for event in events]
    assert "fusion.assist.segment_plan" in kinds
    assert "fusion.assist.segment_plan.request" in kinds
    assert "fusion.assist.segment_plan.result" in kinds
    assert "fusion.assist.segment_fuse.request" in kinds
    assert "fusion.assist.segment_fuse.result" in kinds
    assert "fusion.assist.stitch.request" in kinds
    assert "fusion.assist.stitch.result" in kinds
    assert "fusion.segment_plan" not in kinds
    assert "fusion.stitch.request" not in kinds
    plan = next(event for event in events if event["event"] == "fusion.assist.segment_plan")
    assert plan["mode"] == "action_model"
    assert [segment["title"] for segment in plan["segments"]] == [
        "Task state",
        "Tool plan",
        "Evidence constraints",
    ]
    selects = [event for event in events if event["event"] == "fusion.assist.select"]
    assert [event["fuser_member_id"] for event in selects] == [
        "strong",
        "strong",
        "strong",
    ]
    assert all(event["segment_fusion_applied"] is True for event in selects)


def test_fusion_reply_does_not_inject_raw_candidate_when_segment_fuser_empty(
    tmp_path,
) -> None:
    asyncio.run(_run_empty_segment_fuser_fallback_case(tmp_path))


async def _run_empty_segment_fuser_fallback_case(tmp_path) -> None:
    action = _ActionProvider(text="action final answer", use_tool=False)
    weak = _FakeProvider("weak", "BAD candidate")
    strong = _FakeProvider(
        "strong",
        "GOOD but truncated candidate that should not be injected raw",
        fuse_text="",
    )
    provider = FusionReplyProvider(
        [
            FusionMember("weak", "fake", "weak", weak),
            FusionMember("strong", "fake", "strong", strong),
        ],
        action_provider=action,
        assist_max_segments=1,
        trace_settings=FusionTraceSettings(log_dir=str(tmp_path)),
    )

    done: DoneEvent | None = None
    async for event in provider.chat(
        [Message(role="user", content="research this before answering")],
        tools=_tool_defs(),
        config=ChatConfig(metadata={"fusion_trace_session_key": "session-1"}),
    ):
        if isinstance(event, DoneEvent):
            done = event

    assert done is not None
    action_advice = str(_actual_action_calls(action.calls)[0][0][-1].content)
    assert "GOOD but truncated candidate" not in action_advice
    assert "Fusion could not form a reliable complete advice segment" in action_advice

    summary = done.metadata["fusion_summary"]
    members = summary["assist_participation"]["members"]
    assert all(member["selected_advice"] == 0 for member in members)

    trace_files = list(tmp_path.glob("fusion-traces-*.jsonl"))
    events = [
        json.loads(line)
        for line in trace_files[0].read_text(encoding="utf-8").splitlines()
    ]
    select = next(event for event in events if event["event"] == "fusion.assist.select")
    assert select["segment_fusion_applied"] is False
    assert select["conservative_fallback_used"] is True
    assert select["selected_member_id"] is None
    assert select["selection_reason"] == "conservative complete fallback"
