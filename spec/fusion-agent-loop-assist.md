# Fusion Agent Loop Assist Spec

Status: Draft for review
Date: 2026-06-24

## Summary

The current Fusion Reply implementation fuses only the final visible answer after
OpenSquilla's action model has completed the agent loop. That makes fusion a
final-answer generator/reranker, not an in-loop agent capability.

This spec changes the target architecture: Fusion becomes an advisory layer
inside the existing OpenSquilla agent loop. The OpenSquilla harness remains the
owner of the turn lifecycle, tool execution, context management, retries,
budgets, memory, traces, and persistence. The action model remains the only
model allowed to call tools and the only model that authors the user-visible
assistant response. Fusion models run with tools disabled and provide hidden
next-step advice to the action model before each action-model LLM call.

## Problem

The current behavior is effectively:

```text
OpenSquilla Agent loop runs to completion
->
Action model reaches final text
->
Fusion drafts/verifies/selects final answer segments
->
Fusion emits final assistant response
```

This is not the desired experimental shape. It does not let fusion influence
which tool to call, whether another tool step is needed, how to interpret tool
results, or whether the agent should continue. It only changes the final
wording after the action phase is already over.

The desired behavior is:

```text
OpenSquilla Agent loop
->
Before each action-model LLM request, Fusion generates hidden advice
->
Action model sees current context + hidden fusion advice
->
Action model either calls tools or writes the final answer
->
Tool results return to the same Agent loop
->
The next loop iteration receives fresh Fusion advice over the updated context
```

## Goals

- Keep one canonical OpenSquilla agent loop.
- Reuse OpenSquilla's existing Agent harness, tool execution, context
  compaction, retry, budget, memory, and finalization behavior.
- Run Fusion before each action-model provider call, not after the agent loop
  has completed.
- Make Fusion a hidden advisory layer for the action model.
- Keep tool calls single-model: only the action model may emit tool calls.
- Run all Fusion model calls with tools disabled.
- Preserve `reply_mode = "fusion"` as the user-facing CLI/WebUI entry point.
- Record detailed backend trace for every fusion advisory step.
- Avoid claiming final output character contribution from fusion members when
  the final visible answer is authored by the action model.

## Non-Goals

- Do not create a second independent agent loop outside OpenSquilla.
- Do not let Fusion models call tools.
- Do not let Fusion models directly emit the final user-visible assistant
  message.
- Do not route through the router during fusion mode.
- Do not persist hidden Fusion advice as a normal user or assistant transcript
  message.
- Do not optimize cost in this spec. Cost-reduction strategies can be layered on
  after the architecture is corrected.

## Definitions

- Agent loop: The existing `Agent.run_turn(...)` loop that alternates between
  LLM calls, tool calls, tool results, compaction, and final answer.
- Action model: The single model used by OpenSquilla to decide tool calls and
  author user-visible text.
- Fusion member: A configured model that participates in hidden advisory
  generation and verification.
- Fusion advice: A hidden text block selected by the fusion algorithm and
  injected into the next action-model request.
- Fusion iteration: One advisory run before one action-model LLM call.
- Draft: A Fusion member's proposed next-step advice.
- Verifier: A Fusion member scoring anonymous draft advice candidates.
- Selected advice: The draft chosen by anonymous verification for the current
  Fusion iteration.

## Architecture

### Current incorrect shape

```text
WebUI / CLI
  ->
TurnRunner / Agent Harness
  ->
Action phase: action model uses tools until final text
  ->
Fusion Reply phase: models draft/verify final answer segments
  ->
Fusion emits final response
```

### Target shape

```text
WebUI / CLI
  ->
TurnRunner / Agent Harness
  ->
Agent.run_turn loop
  ->
For each provider call:
    Fusion Assist phase:
      - tools disabled
      - members draft next-step advice
      - members anonymously verify advice candidates
      - anonymous score selection chooses hidden advice
    Action phase:
      - action model sees current context + hidden selected advice
      - tools enabled
      - action model decides tool_call or final text
    If tool_call:
      - OpenSquilla executes tool
      - tool_result is appended to the same agent context
      - loop continues
    Else:
      - action model final text is persisted as assistant response
```

## Responsibility Boundaries

### OpenSquilla Agent Harness

Owns:

- turn lifecycle
- prompt assembly
- message history
- tool definitions
- tool execution
- tool result insertion
- provider retry policy
- invalid-response handling
- context compaction
- budget enforcement
- transcript persistence
- memory capture
- session usage rollup

### Action Model

Owns:

- final decision for the next step
- all tool calls
- all tool arguments
- final user-visible answer
- decision to stop or continue

The action model may use Fusion advice, but it is not required to follow it.

### Fusion Layer

Owns:

- hidden advisory generation
- anonymous advice verification
- anonymous advice scoring and selection
- per-iteration Fusion trace
- contribution/participation telemetry for Fusion members

Fusion must not:

- call tools
- emit tool calls
- directly persist transcript messages
- directly emit final user-visible answer text
- bypass OpenSquilla's Agent harness

## Per-Iteration Flow

Each Agent loop LLM request should follow this sequence:

```text
1. Agent has current turn_messages, available tools, and runtime metadata.
2. Fusion wrapper receives provider.chat(messages, tools, config).
3. If tools are enabled and fusion mode is active:
   3.1 Build a Fusion Assist prompt from:
       - original user request or turn metadata
       - current visible conversation
       - recent tool results
       - available tool summary
       - prior selected Fusion advice summaries for this turn
   3.2 Ask each Fusion member to draft next-step advice.
   3.3 Anonymize advice candidates.
   3.4 Ask Fusion members to verify candidates with tools disabled.
   3.5 Aggregate anonymous verifier scores with equal verifier influence.
   3.6 Select one advice candidate.
   3.7 Emit detailed backend trace.
4. Inject selected advice as an ephemeral hidden context block into the action
   model request.
5. Call the action provider with original tools enabled.
6. Stream action model events back to Agent.
7. If the action model calls a tool, Agent executes it and loops.
8. If the action model returns final text, Agent finalizes the turn normally.
```

If tools are not provided, Fusion may still run as hidden answer guidance, but
the final visible text must still come from the action model. Fusion should not
switch back into final-answer segment generation.

## Fusion Advice Contract

Fusion members should be asked for next-step advice, not final answer prose.

Recommended draft schema:

```json
{
  "task_state": "Brief state of the current task.",
  "recommended_next_step": "What the action model should do next.",
  "tool_guidance": {
    "should_call_tool": true,
    "tool_name": "optional tool name if obvious",
    "reason": "Why a tool is or is not needed.",
    "argument_guidance": "High-level argument guidance, not mandatory JSON."
  },
  "answer_guidance": "If no tool is needed, how to answer.",
  "risks": ["Potential mistake or missing context."],
  "confidence": 0.0
}
```

The selected advice should be injected as text, for example:

```text
Hidden Fusion Assist advice for this agent iteration:
- Current state: ...
- Recommended next step: ...
- Tool guidance: ...
- Risks: ...

This advice is not user-visible. You are the only model allowed to call tools.
Use or ignore the advice as appropriate.
```

The injection should be ephemeral:

- included in the provider request to the action model;
- available in backend turn-call logs or fusion trace if configured;
- not appended as a normal transcript message;
- not shown in WebUI chat as user-visible content.

## SpecEM Mapping

The SpecEM-like algorithm should operate over next-step advice rather than
final answer segments:

```text
Each Fusion member drafts one advice candidate
->
Candidates are anonymized
->
Fusion members verify anonymous candidates
->
Anonymous verifier scores are aggregated equally
->
Best advice is selected for this agent iteration
->
The Agent loop proceeds with the selected hidden advice
```

The "segment" unit becomes "agent iteration advice". A multi-step task naturally
creates multiple fused units because the Agent loop calls the provider multiple
times after tool results.

## Anonymous Score Semantics

Fusion selection should be based only on anonymous verifier scores:

- each verifier receives anonymized candidate labels;
- each verifier returns normalized scores/ranking for the candidates;
- all verifier score sheets have equal influence;
- model configuration `weight` values are ignored by selection;
- no online weight update is performed between Agent loop iterations.

Tie-break order should be deterministic: highest anonymous aggregate score,
then earliest candidate index.

## Trace Semantics

Backend trace must distinguish these concepts:

- action model LLM call
- action model tool call
- tool result
- Fusion draft participation
- Fusion verify participation
- selected Fusion advice
- injected hidden advice
- final answer author

Recommended event kinds:

- `fusion.assist.start`
- `fusion.assist.draft.request`
- `fusion.assist.draft.result`
- `fusion.assist.anonymization`
- `fusion.assist.verify.request`
- `fusion.assist.verify.result`
- `fusion.assist.aggregate`
- `fusion.assist.select`
- `fusion.assist.inject`
- `fusion.assist.end`

Each event should include:

- `trace_id`
- `turn_id`
- `session_key`
- `agent_id`
- `agent_iteration`
- `fusion_iteration`
- `action_model`
- `member_id` or `verifier_id` when applicable
- usage, latency, cost, status, parse mode, and errors when applicable

The trace may include full prompts and full advice text only when full trace mode
is enabled. It must never record API keys, auth headers, proxy credentials, or
environment variables.

## WebUI Semantics

The old final-output contribution chart is misleading for this architecture
because the action model writes the final visible answer.

For Fusion Assist mode, WebUI should display a compact indicator such as:

```text
Fusion assist: enabled
Action model: deepseek/deepseek-v4-flash
Fusion iterations: 4
```

Optional compact display:

| Model | Draft calls | Verify calls | Selected advice |
| --- | ---: | ---: | ---: |
| deepseek_v4_flash | 4 | 4 | 1 |
| gemini_3_flash | 4 | 4 | 2 |
| kimi_k2_7_code | 4 | 4 | 1 |

WebUI must not label Fusion member participation as final answer contribution.
If final output authorship is shown, it should be:

```text
Final answer author: action model
```

## Configuration

Keep the user-facing mode:

```toml
[reply]
default_mode = "fusion"
```

Add or reinterpret Fusion config with an explicit architecture field:

```toml
[fusion_reply]
enabled = true
architecture = "agent_loop_assist"
action_provider = "openrouter"
action_model = "deepseek/deepseek-v4-flash"
assist_max_rounds = 1
judge_max_tokens = 512
temperature = 0.4
judge_temperature = 0.0
inject_strategy = "ephemeral_hidden_context"
```

Supported `architecture` values:

- `agent_loop_assist`: target architecture from this spec.
- `final_answer_fusion`: legacy behavior, deprecated for this experiment.

For this experiment, `agent_loop_assist` should be the default whenever
`reply_mode = "fusion"`.

## Implementation Plan

### Step 1: Rename the conceptual provider role

Keep compatibility with existing imports if needed, but the implementation
should behave as a Fusion Assist provider, not a final Fusion Reply provider.

Possible names:

- `FusionAssistProvider`
- `FusionAugmentedActionProvider`

The external `reply_mode = "fusion"` can remain unchanged.

### Step 2: Move Fusion before the action provider call

Current behavior waits until action model final text before calling SpecEM.

Required behavior:

```text
provider.chat(messages, tools, config)
  ->
run_fusion_assist(messages, tools, config)
  ->
inject hidden advice into messages
  ->
action_provider.chat(messages_with_advice, tools, config)
  ->
return action provider events directly to Agent
```

Do not call final-answer `_chat_specem(...)` after action final text.

### Step 3: Preserve native Agent loop behavior

No new outer loop should be introduced in Fusion.

The existing Agent loop already calls `provider.chat(...)` once per LLM
iteration. Therefore, putting Fusion before each wrapped action provider call
naturally makes Fusion run once per Agent iteration.

### Step 4: Add an advice prompt builder

Build prompts from:

- original user request or stable turn metadata;
- current conversation state;
- recent tool results;
- available tool names and short descriptions;
- previous selected advice summaries;
- current action-model role.

Avoid using the last raw user/tool message as the task identity when original
user intent metadata is available.

### Step 5: Update trace schema

Extend the existing Fusion trace to record assist events and injection events.

The trace must show:

- which models drafted advice;
- which models verified advice;
- anonymous scores;
- which advice was injected;
- which action-model call followed the injected advice;
- whether that action-model call produced tool calls or final text.

### Step 6: Update WebUI compact display

Replace final output contribution for `agent_loop_assist` with assist
participation metrics. Keep the detailed trace backend-only.

### Step 7: Tests

Add tests for:

- Fusion runs before action provider call.
- Fusion selected advice is present in the action provider request.
- Fusion advice is not persisted as a transcript message.
- Fusion calls have tools disabled.
- Action provider still receives tools enabled.
- Action model tool calls are executed by the existing Agent loop.
- After tool results, the next Agent iteration runs Fusion again.
- Final visible answer comes from the action model.
- Router is not invoked in fusion mode.
- Trace records each Fusion assist iteration.

## Acceptance Criteria

- In `reply_mode = "fusion"`, Fusion runs before each action-model LLM call.
- A task with two tool iterations produces at least two Fusion assist traces.
- The action model remains the only model that emits tool calls.
- Tool results are appended by the existing Agent loop and feed the next
  iteration.
- The final assistant response is authored by the action model, not by Fusion.
- Fusion advice is visible in backend trace but not in the chat transcript.
- WebUI does not show Fusion member final-output contribution for this mode.
- Existing direct/direct-router behavior is unchanged outside fusion mode.
- Legacy final-answer fusion can be disabled or marked deprecated.

## Open Questions

- Should `final_answer_fusion` remain available behind an explicit config flag,
  or should it be removed entirely for this experiment?
- Should Fusion run before every action-model request, or only when the task is
  classified as complex or uncertain?
- Should verifier members include the action model itself, or only non-action
  Fusion members?
- Should selected advice be one candidate verbatim, or should a final synthesis
  model compress multiple useful candidates into one advice block?
- Should WebUI show only an enabled badge, or also compact draft/verify/selected
  counts?
- How much of the available tool schema should be included in Fusion prompts to
  balance usefulness and cost?

## Recommended Default for the Next Experiment

- Architecture: `agent_loop_assist`.
- Action model: `deepseek/deepseek-v4-flash`.
- Fusion members:
  - `deepseek/deepseek-v4-flash`
  - `google/gemini-3-flash-preview`
  - `moonshotai/kimi-k2.7-code`
- Fusion calls: tools disabled.
- Action calls: tools enabled.
- Assist rounds per Agent iteration: 1.
- Verifier mode: anonymous pairwise/ranking scores.
- Trace level: full.
- WebUI: show Fusion assist participation, not final output contribution.
- Router: disabled.
