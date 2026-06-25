# Fusion Assist Segment Trace Observability Spec

Status: Updated
Date: 2026-06-25

## Summary

Fusion mode is now an OpenSquilla-native assist layer, not a final-answer
generation layer. The action model remains the only model that can call tools
and author user-visible text. Before each action-model call, Fusion members
produce hidden advice segments, anonymously verify candidates for each segment,
and inject the selected advice segments into the action-model request.

## Current Algorithm

```text
OpenSquilla Agent loop
  -> Fusion Assist before one action-model call
  -> Semantic hidden advice segment plan
  -> For each advice segment:
       each Fusion member drafts one candidate
       candidates are anonymized as A/B/C
       Fusion members score anonymous candidates
       scores are normalized and averaged with equal verifier influence
       constrained segment fusion produces one fused segment
       empty/error fusion falls back to conservative complete advice
  -> If only one fused segment: inject it directly
  -> If 2+ fused segments: constrained final stitch
       only deduplicate, connect, and normalize formatting
       do not add facts, sources, numbers, or tool results
       empty/error stitch falls back to concat
  -> Action model receives tools and decides whether to call tools or answer
```

There is no final-answer Fusion stage, no model weights, and no online weight
update. The only stitch pass is a constrained hidden-advice stitch, not a
user-visible final answer writer.

## Segment Semantics

Segments are semantic hidden-advice slots, not fixed token windows.

At the start of each Fusion Assist pass, OpenSquilla asks the action model to
plan the semantic hidden-advice slots for its own upcoming agent-loop call.
This planner call uses the normal OpenSquilla text-call harness with tools
disabled. It must return strict JSON containing `segments`, where each segment
has `title`, `instruction`, and `target_words`.

The plan is capped by `fusion_reply.assist_max_segments`. The segment loop stops
when the action-model-planned segments are exhausted. It does not rely on a
model-emitted done sentinel. If more work is needed, the normal OpenSquilla
agent loop will run another action iteration after tool results or updated
context.

If the action-model planner fails, returns invalid JSON, or returns no usable
segments, Fusion may use a small generic safety fallback so the agent loop can
continue. This fallback is not an alternate baseline strategy and should be
visible in trace as `fallback_used=true`.

## Single Segment Selection

Each individual segment is constrained-fused, not picked verbatim.

The verifier scores still matter: they identify the fallback candidate and the
segment fuser. The fuser receives anonymous candidates plus scores and must
produce one complete hidden advice segment using only candidate content. The
fuser is also the completeness gate: if candidates are truncated or thin, it
should close the advice conservatively by preserving supported guidance and
noting uncertainty. If the fusion call fails or returns empty text, the segment
falls back to conservative complete advice, not a raw candidate.

## Multi Segment Combination

If there is one fused segment, it is injected directly. If there are two or more
fused segments, Fusion runs constrained final stitch over the selected segment
texts:

```text
Task state:
...

Tool plan:
...

Evidence constraints:
...
```

The stitch model may only remove duplication, smooth transitions, and normalize
formatting. It may not add facts, sources, numbers, tool results, or tool
arguments. If stitch fails or returns empty text, Fusion falls back to the
structured concat form above.

## Backend Trace

Backend JSONL trace must retain the full audit chain:

- `fusion.start`
- `fusion.assist.start`
- `fusion.assist.segment_plan.request`
- `fusion.assist.segment_plan.result`
- `fusion.assist.segment_plan`
- `fusion.assist.segment.start`
- `fusion.assist.draft.request`
- `fusion.assist.draft.result`
- `fusion.assist.anonymization`
- `fusion.assist.verify.request`
- `fusion.assist.verify.result`
- `fusion.assist.aggregate`
- `fusion.assist.segment_fuse.request`
- `fusion.assist.segment_fuse.result`
- `fusion.assist.select`
- `fusion.assist.stitch.skipped`
- `fusion.assist.stitch.request`
- `fusion.assist.stitch.result`
- `fusion.assist.inject`
- `fusion.assist.end`

Trace events should include `fusion_iteration`, `segment_index`,
`segment_goal`, member IDs, anonymized label maps, raw verifier output when
enabled, parsed scores, normalized scores, aggregate scores, fuser member,
fuser seed candidate, stitch editor, usage, latency, and harness retry metadata.

## WebUI Summary

WebUI should only show compact assist participation, not candidate text or raw
scores.

Compact transcript summary:

```json
{
  "fusion_summary": {
    "architecture": "agent_loop_assist",
    "algorithm": "specem_anonymous_advice_score",
    "assist_iterations": 2,
    "final_answer_author": "action_model",
    "assist_participation": {
      "basis": "draft_verify_selected_advice_segments",
      "members": [
        {
          "member_id": "gemini_3_flash",
          "draft_calls": 6,
          "verify_calls": 6,
          "fusion_calls": 2,
          "stitch_calls": 1,
          "selected_advice": 2
        }
      ]
    }
  }
}
```

Detailed trace remains backend-only.
