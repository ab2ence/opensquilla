# Fusion Trace Observability Spec

Status: Draft
Date: 2026-06-23

## Summary

OpenSquilla Fusion Reply currently records the final turn as a composite model,
for example `fusion:deepseek_v4_flash+gemini_3_flash+kimi_k2_7_code`, but it
does not persist enough detail to audit how each fusion member participated.

This spec adds a detailed backend trace for the fusion phase while keeping the
WebUI intentionally simple. The WebUI should only show final output
contribution. The backend trace must retain the full fusion audit chain:
candidate generation, anonymous verification, normalized scoring, weighted
aggregation, selection, weight update, usage, latency, and cost per member.

## Goals

- Make every Fusion Reply turn auditable after completion.
- Record exactly which model generated each candidate segment.
- Record exactly which model verified each anonymous candidate set.
- Record raw verifier output, parsed scores, normalized scores, rankings, and
  parse fallback behavior.
- Record weights before and after each fusion round.
- Record final selected member per round and final output contribution per
  member.
- Record per-member usage: input tokens, output tokens, reasoning tokens,
  cached tokens, cache write tokens, billed cost, cost source, and latency.
- Keep WebUI display minimal: only final output contribution by model.
- Keep detailed trace available on disk for debugging, research, and evaluation.

## Non-Goals

- Do not show verifier scores, candidate text, or raw model outputs in WebUI by
  default.
- Do not change the Fusion Reply algorithm itself.
- Do not fuse tool calls. Tool calls remain single-action-model behavior.
- Do not make router runtime a dependency of fusion tracing.
- Do not require a database migration for the full detailed trace.

## Definitions

- Action model: The single model used by OpenSquilla to call tools and perform
  actions. In the current config this is `deepseek/deepseek-v4-flash`.
- Fusion member: A model configured under `[[fusion_reply.models]]`.
- Candidate: One member's proposed next answer segment during a fusion round.
- Verifier: A fusion member acting as judge over anonymous candidates.
- Output contribution: The fraction of final visible answer text selected from
  each model's candidates.
- Vote weight: The current algorithmic verifier/candidate weight for a member.
- Score share: A candidate's weighted score divided by the total weighted score
  for the round.

## User-Facing Behavior

### WebUI

WebUI must only display final output contribution, not internal scoring.

Example compact display:

| Model | Final output contribution |
| --- | ---: |
| deepseek_v4_flash | 0% |
| gemini_3_flash | 100% |
| kimi_k2_7_code | 0% |

Fusion experiments should run in multi-segment mode. `min_rounds` prevents the
first selected candidate from ending the whole answer immediately, so the trace
can validate iterative segment generation, anonymous comparison, selection, and
weight update across rounds. Contribution is accumulated over selected
segments:

| Model | Final output contribution |
| --- | ---: |
| deepseek_v4_flash | 38% |
| gemini_3_flash | 42% |
| kimi_k2_7_code | 20% |

Default contribution unit: characters.

Future optional unit: tokenizer-estimated output tokens. If token contribution is
added later, WebUI must label the unit clearly.

### Transcript Summary

The transcript `turn_usage` should include only a compact fusion summary:

```json
{
  "fusion_summary": {
    "fusion_trace_id": "fus_20260623_021349_dde6da8b",
    "round_count": 1,
    "selected_members": ["gemini_3_flash"],
    "output_contribution": {
      "unit": "chars",
      "members": {
        "deepseek_v4_flash": {
          "selected_chars": 0,
          "share": 0.0
        },
        "gemini_3_flash": {
          "selected_chars": 1141,
          "share": 1.0
        },
        "kimi_k2_7_code": {
          "selected_chars": 0,
          "share": 0.0
        }
      }
    }
  }
}
```

The transcript summary should not contain full candidate text, full prompts, raw
verifier responses, or detailed per-call request data.

## Backend Trace Storage

Detailed trace should be written to daily JSONL files:

```text
C:\Users\10104\.opensquilla\logs\fusion-traces-YYYYMMDD.jsonl
```

Each line is one event. All events for a turn share:

- `schema_version`
- `ts`
- `kind`
- `fusion_trace_id`
- `trace_id`
- `turn_id`
- `session_key`
- `session_id`
- `round`

The detailed trace is append-only and should not block the user turn if writing
fails. Trace write failures should emit a warning log but must not fail the
fusion response.

## Trace Configuration

Add:

```toml
[fusion_reply.trace]
enabled = true
level = "full"
write_jsonl = true
include_full_prompts = true
include_full_candidate_text = true
include_full_verifier_response = true
include_full_selected_text = true
include_usage = true
include_latency = true
include_cost = true
expose_webui_summary_only = true
```

Supported levels:

- `off`: no fusion trace beyond existing turn usage.
- `summary`: only `fusion.start`, `fusion.select`, and `fusion.end`.
- `full`: all events and full text fields.

For this experiment, default should be `full`.

Security rule: never record API keys, authorization headers, environment
variables, proxy credentials, or provider request headers.

## Event Schema

### fusion.start

Emitted once before fusion drafting begins.

```json
{
  "schema_version": 1,
  "kind": "fusion.start",
  "fusion_trace_id": "fus_20260623_021349_dde6da8b",
  "trace_id": "dde6da8bbfbd45ceb5f92b7e9066ec13",
  "turn_id": "74f44afd9bdc4ec585e9e13e887af24f",
  "session_key": "agent:main:webchat:bhbww8sk",
  "session_id": "cc6178d3-a2eb-461b-8f56-282780cee2d2",
  "config": {
    "max_rounds": 6,
    "min_rounds": 2,
    "step_max_tokens": 384,
    "judge_max_tokens": 512,
    "temperature": 0.7,
    "judge_temperature": 0.0,
    "feedback_alpha": 1.0,
    "adaptive_segments": true
  },
  "action": {
    "provider": "openrouter",
    "model": "deepseek/deepseek-v4-flash",
    "member_id": "deepseek_v4_flash"
  },
  "members": [
    {
      "id": "deepseek_v4_flash",
      "provider": "openrouter",
      "model": "deepseek/deepseek-v4-flash",
      "configured_weight": 1.0,
      "initial_weight": 0.333333
    },
    {
      "id": "gemini_3_flash",
      "provider": "openrouter",
      "model": "google/gemini-3-flash-preview",
      "configured_weight": 1.0,
      "initial_weight": 0.333333
    },
    {
      "id": "kimi_k2_7_code",
      "provider": "openrouter",
      "model": "moonshotai/kimi-k2.7-code",
      "configured_weight": 1.0,
      "initial_weight": 0.333333
    }
  ]
}
```

### fusion.segment_plan

Emitted once before the first round when adaptive segment planning is enabled.

```json
{
  "kind": "fusion.segment_plan",
  "mode": "adaptive_heuristic",
  "effective_min_rounds": 6,
  "segments": [
    {
      "index": 0,
      "title": "Conclusion",
      "instruction": "Give the core answer and framing first.",
      "target_words": "100-180 words"
    },
    {
      "index": 1,
      "title": "Evidence",
      "instruction": "Summarize the key facts, data, and sources already gathered.",
      "target_words": "100-180 words"
    }
  ]
}
```

### fusion.round.start

Emitted once per round.

```json
{
  "kind": "fusion.round.start",
  "round": 1,
  "weights_before": {
    "deepseek_v4_flash": 0.333333,
    "gemini_3_flash": 0.333333,
    "kimi_k2_7_code": 0.333333
  },
  "selected_text_chars_before_round": 0,
  "segment_goal": {
    "index": 0,
    "title": "Conclusion",
    "instruction": "Give the core answer and framing first.",
    "target_words": "100-180 words"
  }
}
```

### fusion.draft.request

Emitted before each member draft request.

```json
{
  "kind": "fusion.draft.request",
  "round": 1,
  "member_id": "gemini_3_flash",
  "candidate_index": 1,
  "request": {
    "system_prompt": "full system prompt when include_full_prompts=true",
    "messages": [
      {
        "role": "user",
        "content": "full request content when include_full_prompts=true"
      }
    ],
    "max_tokens": 1024,
    "temperature": 0.7
  }
}
```

If `include_full_prompts=false`, store:

```json
{
  "request": {
    "system_prompt_sha256": "...",
    "messages_sha256": "...",
    "system_prompt_chars": 1234,
    "messages_chars": 5678
  }
}
```

### fusion.draft.result

Emitted after each draft request succeeds or fails.

```json
{
  "kind": "fusion.draft.result",
  "round": 1,
  "member_id": "gemini_3_flash",
  "provider": "openrouter",
  "model": "google/gemini-3-flash-preview",
  "candidate_index": 1,
  "anonymous_label": "B",
  "status": "ok",
  "text": "full candidate text when include_full_candidate_text=true",
  "text_sha256": "...",
  "chars": 1180,
  "done_sentinel_present": true,
  "usage": {
    "input_tokens": 52000,
    "output_tokens": 900,
    "reasoning_tokens": 0,
    "cached_tokens": 0,
    "cache_write_tokens": 0,
    "billed_cost": 0.0123,
    "cost_source": "provider_billed"
  },
  "latency_ms": 18320,
  "error": null
}
```

Failure example:

```json
{
  "kind": "fusion.draft.result",
  "round": 1,
  "member_id": "gemini_3_flash",
  "candidate_index": 1,
  "status": "error",
  "text": "",
  "usage": {
    "input_tokens": 0,
    "output_tokens": 0
  },
  "latency_ms": 30000,
  "error": {
    "code": "provider_timeout",
    "message": "request timed out"
  }
}
```

### fusion.anonymization

Emitted once after candidates are labeled.

```json
{
  "kind": "fusion.anonymization",
  "round": 1,
  "label_to_candidate": {
    "A": {
      "candidate_index": 0,
      "member_id": "deepseek_v4_flash"
    },
    "B": {
      "candidate_index": 1,
      "member_id": "gemini_3_flash"
    },
    "C": {
      "candidate_index": 2,
      "member_id": "kimi_k2_7_code"
    }
  }
}
```

This event is backend-only. WebUI must not expose anonymous label mapping by
default.

### fusion.verify.request

Emitted before each verifier request.

```json
{
  "kind": "fusion.verify.request",
  "round": 1,
  "verifier_id": "kimi_k2_7_code",
  "verifier_weight": 0.333333,
  "candidate_labels": ["A", "B", "C"],
  "request": {
    "system_prompt": "full verifier system prompt",
    "messages": [
      {
        "role": "user",
        "content": "full anonymous candidate scoring prompt"
      }
    ],
    "max_tokens": 512,
    "temperature": 0.0
  }
}
```

### fusion.verify.result

Emitted after each verifier request succeeds or fails.

```json
{
  "kind": "fusion.verify.result",
  "round": 1,
  "verifier_id": "kimi_k2_7_code",
  "provider": "openrouter",
  "model": "moonshotai/kimi-k2.7-code",
  "verifier_weight": 0.333333,
  "status": "ok",
  "raw_response": "{\"scores\":{\"A\":0.72,\"B\":0.91,\"C\":0.55},\"ranking\":[\"B\",\"A\",\"C\"]}",
  "parsed_scores": {
    "A": 0.72,
    "B": 0.91,
    "C": 0.55
  },
  "normalized_scores": {
    "A": 0.330,
    "B": 0.417,
    "C": 0.252
  },
  "candidate_scores_by_member": {
    "deepseek_v4_flash": 0.330,
    "gemini_3_flash": 0.417,
    "kimi_k2_7_code": 0.252
  },
  "ranking": ["B", "A", "C"],
  "parse_mode": "json_scores",
  "parse_fallback_used": false,
  "usage": {
    "input_tokens": 53000,
    "output_tokens": 120,
    "reasoning_tokens": 0,
    "cached_tokens": 0,
    "cache_write_tokens": 0,
    "billed_cost": 0.0041,
    "cost_source": "provider_billed"
  },
  "latency_ms": 9200,
  "error": null
}
```

Supported `parse_mode` values:

- `json_scores`
- `json_ranking`
- `text_ranking_fallback`
- `uniform_fallback`
- `error`

### fusion.aggregate

Emitted once scores are aggregated.

```json
{
  "kind": "fusion.aggregate",
  "round": 1,
  "weights_before": {
    "deepseek_v4_flash": 0.333333,
    "gemini_3_flash": 0.333333,
    "kimi_k2_7_code": 0.333333
  },
  "candidate_scores": {
    "0": {
      "member_id": "deepseek_v4_flash",
      "weighted_score": 0.301
    },
    "1": {
      "member_id": "gemini_3_flash",
      "weighted_score": 0.421
    },
    "2": {
      "member_id": "kimi_k2_7_code",
      "weighted_score": 0.278
    }
  },
  "score_share": {
    "deepseek_v4_flash": 0.301,
    "gemini_3_flash": 0.421,
    "kimi_k2_7_code": 0.278
  },
  "verifier_count": 3,
  "candidate_count": 3
}
```

### fusion.select

Emitted once selected candidate is known.

```json
{
  "kind": "fusion.select",
  "round": 1,
  "selected_candidate_index": 1,
  "selected_label": "B",
  "selected_member": "gemini_3_flash",
  "selection_reason": "highest_weighted_score",
  "selected_weighted_score": 0.421,
  "tie_break": null,
  "selected_text": "full selected text when include_full_selected_text=true",
  "selected_text_chars": 1141,
  "selected_text_sha256": "...",
  "done_sentinel_present": true
}
```

Supported `selection_reason` values:

- `highest_weighted_score`
- `tie_break_member_weight`
- `tie_break_candidate_index`
- `single_candidate`

### fusion.weights.update

Emitted once online feedback updates weights.

```json
{
  "kind": "fusion.weights.update",
  "round": 1,
  "learning_rate": 0.333333,
  "feedback_alpha": 1.0,
  "rewards": {
    "deepseek_v4_flash": 0.25,
    "gemini_3_flash": 0.50,
    "kimi_k2_7_code": 0.25
  },
  "weights_before": {
    "deepseek_v4_flash": 0.333333,
    "gemini_3_flash": 0.333333,
    "kimi_k2_7_code": 0.333333
  },
  "weights_after": {
    "deepseek_v4_flash": 0.324,
    "gemini_3_flash": 0.352,
    "kimi_k2_7_code": 0.324
  }
}
```

For explicit one-round diagnostic runs, `weights_after` is recorded for audit,
but it does not affect the already selected final answer. Fusion experiments
should use multi-segment settings such as `max_rounds = 6`, `min_rounds = 2`,
and `adaptive_segments = true`.

### fusion.end

Emitted once after all fusion rounds finish.

```json
{
  "kind": "fusion.end",
  "fusion_trace_id": "fus_20260623_021349_dde6da8b",
  "round_count": 1,
  "final_text": "full final answer text",
  "final_text_sha256": "...",
  "final_text_chars": 1141,
  "selected_segments": [
    {
      "round": 1,
      "member_id": "gemini_3_flash",
      "candidate_index": 1,
      "chars": 1141,
      "share": 1.0
    }
  ],
  "output_contribution": {
    "unit": "chars",
    "members": {
      "deepseek_v4_flash": {
        "selected_chars": 0,
        "share": 0.0
      },
      "gemini_3_flash": {
        "selected_chars": 1141,
        "share": 1.0
      },
      "kimi_k2_7_code": {
        "selected_chars": 0,
        "share": 0.0
      }
    }
  },
  "member_totals": {
    "deepseek_v4_flash": {
      "draft_calls": 1,
      "verify_calls": 1,
      "draft_successes": 1,
      "verify_successes": 1,
      "selected_rounds": 0,
      "selected_chars": 0,
      "input_tokens": 104000,
      "output_tokens": 1020,
      "reasoning_tokens": 0,
      "cached_tokens": 0,
      "cache_write_tokens": 0,
      "billed_cost": 0.016,
      "latency_ms": 27520
    },
    "gemini_3_flash": {
      "draft_calls": 1,
      "verify_calls": 1,
      "draft_successes": 1,
      "verify_successes": 1,
      "selected_rounds": 1,
      "selected_chars": 1141,
      "input_tokens": 103000,
      "output_tokens": 1060,
      "reasoning_tokens": 0,
      "cached_tokens": 0,
      "cache_write_tokens": 0,
      "billed_cost": 0.019,
      "latency_ms": 28110
    },
    "kimi_k2_7_code": {
      "draft_calls": 1,
      "verify_calls": 1,
      "draft_successes": 1,
      "verify_successes": 1,
      "selected_rounds": 0,
      "selected_chars": 0,
      "input_tokens": 102000,
      "output_tokens": 990,
      "reasoning_tokens": 0,
      "cached_tokens": 0,
      "cache_write_tokens": 0,
      "billed_cost": 0.015,
      "latency_ms": 27130
    }
  },
  "aggregate_usage": {
    "input_tokens": 426473,
    "output_tokens": 11269,
    "reasoning_tokens": 3932,
    "cached_tokens": 236520,
    "cache_write_tokens": 2676,
    "billed_cost": 0.0894410504,
    "cost_source": "provider_billed"
  }
}
```

## Contribution Semantics

The trace must keep these concepts separate:

- Tool/action participation: single action model participation in tool calls.
- Draft participation: candidate generation calls per fusion member.
- Verify participation: judge calls per fusion member.
- Vote weight: current weight used to aggregate verifier scores.
- Score share: how much support each candidate received in the round.
- Final output contribution: how many final answer characters came from each
  selected member.

WebUI displays only final output contribution.

Backend trace records all concepts.

## Failure Semantics

If a member fails draft:

- Emit `fusion.draft.result` with `status = "error"`.
- Exclude that member's candidate from candidate aggregation for the round.
- Still allow the member to verify if its verifier call can run.

If a verifier fails:

- Emit `fusion.verify.result` with `status = "error"` and `parse_mode = "error"`.
- Exclude that verifier from weighted aggregation.
- Continue if at least one verifier result remains.

If all verifiers fail:

- Emit `fusion.aggregate` with `fallback = "uniform_candidate_scores"`.
- Select by deterministic fallback:
  1. highest current member weight
  2. lowest candidate index

If all drafts fail:

- Emit `fusion.end` with `status = "error"`.
- Surface provider error to Agent as today.

## Privacy and Retention

The experiment requires full backend trace, including full prompts and full
model text. This is useful for debugging but sensitive.

Rules:

- Store traces only in local OpenSquilla logs.
- Never record API keys, auth headers, proxy credentials, or environment
  variables.
- Never include provider request headers.
- Record full prompts and full candidate/verifier text only under
  `[fusion_reply.trace] level = "full"`.
- WebUI must not fetch or display full fusion trace by default.
- Add a future cleanup command or retention setting if trace volume becomes too
  large.

Suggested future retention config:

```toml
[fusion_reply.trace]
retention_days = 14
max_file_mb = 256
```

## Implementation Plan

### Step 1: Add trace config

Files:

- `src/opensquilla/gateway/config.py`
- `src/opensquilla/fusion_reply/factory.py`

Add `FusionTraceConfig` fields:

- `enabled`
- `level`
- `write_jsonl`
- `include_full_prompts`
- `include_full_candidate_text`
- `include_full_verifier_response`
- `include_full_selected_text`
- `include_usage`
- `include_latency`
- `include_cost`
- `expose_webui_summary_only`

Pass config into `FusionReplyProvider`.

### Step 2: Add trace data model and writer

New file:

- `src/opensquilla/fusion_reply/trace.py`

Responsibilities:

- Generate `fusion_trace_id`.
- Write JSONL events.
- Redact secrets.
- Build compact summary for transcript.
- Track per-member totals.
- Track selected segment contribution.
- Avoid failing the user turn when trace writes fail.

### Step 3: Instrument FusionReplyProvider

File:

- `src/opensquilla/fusion_reply/provider.py`

Instrument:

- `_chat_specem`
- `_draft_candidate`
- `_verify_candidates`
- `_weighted_scores`
- `_select_candidate`
- `_updated_weights`

Required events:

- `fusion.start`
- `fusion.segment_plan`
- `fusion.round.start`
- `fusion.draft.request`
- `fusion.draft.result`
- `fusion.anonymization`
- `fusion.verify.request`
- `fusion.verify.result`
- `fusion.aggregate`
- `fusion.select`
- `fusion.weights.update`
- `fusion.end`

### Step 4: Attach compact summary to DoneEvent

Current provider `DoneEvent` does not carry arbitrary metadata. Add one of:

Option A: Extend provider `DoneEvent` with `metadata: dict[str, Any]`.

Option B: Add a provider-side `FusionTraceSummaryEvent` before `DoneEvent`, then
Agent captures it into turn usage.

Preferred: Option A if compatibility impact is low. Option B if preserving
provider event shape is safer.

### Step 5: Persist compact summary into transcript usage

Files likely involved:

- `src/opensquilla/engine/agent.py`
- `src/opensquilla/engine/turn_runner/stream_consumer_stage.py`
- `src/opensquilla/engine/turn_runner/turn_finalizer_stage.py`

Persist:

```json
turn_usage.fusion_summary
```

Do not persist full detailed trace into transcript.

### Step 6: WebUI display

Files likely involved:

- `opensquilla-webui/src/views/ChatView.vue`
- `opensquilla-webui/src/types/rpc.ts`

Display only:

- `fusion_summary.output_contribution.members`
- model IDs
- percentages

Do not expose:

- scores
- candidate text
- raw verifier output
- prompt text
- weight updates

### Step 7: Tests

Add tests for:

- Trace config parsing.
- JSONL trace event sequence for one-round fusion.
- Draft success and verifier success record usage and latency.
- Verifier parse fallback is recorded.
- Weighted aggregate and selected member are recorded.
- `fusion_summary.output_contribution` is stored in transcript usage.
- WebUI uses only compact contribution fields.
- Tool calls do not get fused and remain action-model only.
- Artifact final response still enters fusion and records selected contribution.

## Acceptance Criteria

- A completed Fusion Reply turn writes a `fusion-traces-YYYYMMDD.jsonl` file.
- The file contains all required event kinds for a successful one-round fusion.
- Each fusion member has draft and verify participation recorded.
- Each verifier's raw response, parsed scores, normalized scores, and ranking are
  recorded.
- Aggregate weighted scores and selection reason are recorded.
- `fusion.end` includes per-member totals and final output contribution.
- Transcript `turn_usage` includes compact `fusion_summary`.
- WebUI displays only final output contribution by model.
- WebUI does not display raw scores, weights, candidate text, or verifier text.
- If one verifier fails, trace records the failure and fusion still completes.
- If all verifiers fail, trace records deterministic fallback behavior.
- Existing fusion behavior and final answer text remain unchanged.

## Open Questions

- Should output contribution be based on characters only, or should token-based
  contribution be added immediately?
- Should full prompt recording include entire tool result content, or should
  large tool results be stored as `sha256 + preview + size` even in full mode?
- Should there be a UI affordance for opening the local trace file, or should it
  remain backend-only for now?
- Should detailed trace retention be enforced immediately or deferred until trace
  volume becomes a problem?

## Recommended Default for This Experiment

- Backend trace: full.
- WebUI: final output contribution only.
- Transcript: compact summary only.
- Contribution unit: characters.
- Full candidate text: enabled.
- Full verifier response: enabled.
- Full prompts: enabled, with secret redaction.
- Router: disabled.
- Tool calls: single action model only.
