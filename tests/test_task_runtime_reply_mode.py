from __future__ import annotations

from types import SimpleNamespace

from opensquilla.gateway.boot import build_task_runtime_run_kwargs


def test_task_runtime_run_kwargs_include_reply_mode() -> None:
    run = SimpleNamespace(
        agent_id="main",
        attachments=[],
        input_provenance={"kind": "test"},
        run_kind="session_turn",
        no_memory_capture=False,
        fresh_user_session=False,
        ingress_pipeline_steps=(),
        semantic_message=None,
        reply_mode="fusion",
    )

    kwargs = build_task_runtime_run_kwargs(
        run,
        tool_context=object(),
        model=None,
    )

    assert kwargs["reply_mode"] == "fusion"
