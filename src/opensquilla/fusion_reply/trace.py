"""Trace recording for OpenSquilla Fusion Reply experiments."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class FusionTraceSettings:
    """Runtime trace controls for Fusion Reply."""

    enabled: bool = True
    level: str = "full"
    write_jsonl: bool = True
    log_dir: str = ""
    include_prompts: bool = True
    include_candidate_text: bool = True
    include_verifier_text: bool = True
    expose_summary: bool = True

    @classmethod
    def from_config(cls, config: Any | None) -> "FusionTraceSettings":
        if config is None:
            return cls()
        return cls(
            enabled=bool(getattr(config, "enabled", True)),
            level=str(getattr(config, "level", "full") or "full"),
            write_jsonl=bool(getattr(config, "write_jsonl", True)),
            log_dir=str(getattr(config, "log_dir", "") or ""),
            include_prompts=bool(getattr(config, "include_prompts", True)),
            include_candidate_text=bool(
                getattr(config, "include_candidate_text", True)
            ),
            include_verifier_text=bool(
                getattr(config, "include_verifier_text", True)
            ),
            expose_summary=bool(getattr(config, "expose_summary", True)),
        )


class FusionTraceRecorder:
    """Append-only JSONL trace writer plus compact final contribution summary."""

    def __init__(
        self,
        settings: FusionTraceSettings,
        *,
        metadata: Mapping[str, Any] | None,
        members: list[Mapping[str, Any]],
        action_model: str,
        max_rounds: int,
        architecture: str = "final_answer_fusion",
        min_rounds: int = 1,
    ) -> None:
        self.settings = settings
        self.trace_id = str(uuid.uuid4())
        self.metadata = dict(metadata or {})
        self.path = _trace_path(settings) if settings.enabled and settings.write_jsonl else None
        self._members = [dict(member) for member in members]
        self._architecture = str(architecture or "final_answer_fusion")
        self._action_model = action_model
        self._member_stats: dict[str, dict[str, Any]] = {}
        for member in self._members:
            member_id = str(member.get("id") or "")
            self._member_stats[member_id] = {
                "member_id": member_id,
                "provider": str(member.get("provider") or ""),
                "model": str(member.get("model") or ""),
                "draft_calls": 0,
                "verify_calls": 0,
                "stitch_calls": 0,
                "selected_advice": 0,
                "selected_segments": 0,
                "selected_chars": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "cached_tokens": 0,
                "cache_write_tokens": 0,
                "billed_cost": 0.0,
            }
        self.emit(
            "fusion.start",
            architecture=self._architecture,
            action_model=action_model,
            max_rounds=max_rounds,
            min_rounds=min_rounds,
            members=self._members,
            trace_level=settings.level,
        )

    def emit(self, event: str, **payload: Any) -> None:
        """Write one trace event if JSONL tracing is enabled."""

        if not self.settings.enabled or self.path is None:
            return
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "trace_id": self.trace_id,
            "event": event,
            "session_key": self.metadata.get("fusion_trace_session_key")
            or self.metadata.get("session_key"),
            "agent_id": self.metadata.get("fusion_trace_agent_id")
            or self.metadata.get("agent_id"),
            "reply_mode": self.metadata.get("reply_mode"),
            **_redact_sensitive(payload),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(record), ensure_ascii=False) + "\n")

    def record_usage(
        self,
        member_id: str,
        *,
        stage: str,
        usage: Mapping[str, Any],
    ) -> None:
        stats = self._member_stats.get(member_id)
        if stats is None:
            return
        if stage == "draft":
            stats["draft_calls"] += 1
        elif stage == "verify":
            stats["verify_calls"] += 1
        elif stage == "stitch":
            stats["stitch_calls"] += 1
        for key in (
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "cached_tokens",
            "cache_write_tokens",
        ):
            stats[key] = int(stats.get(key, 0) or 0) + int(usage.get(key, 0) or 0)
        stats["billed_cost"] = float(stats.get("billed_cost", 0.0) or 0.0) + float(
            usage.get("billed_cost", 0.0) or 0.0
        )

    def record_selected(self, member_id: str, text: str) -> None:
        stats = self._member_stats.get(member_id)
        if stats is None:
            return
        stats["selected_segments"] += 1
        stats["selected_chars"] += len(text or "")

    def record_selected_advice(self, member_id: str, text: str) -> None:
        stats = self._member_stats.get(member_id)
        if stats is None:
            return
        stats["selected_advice"] += 1
        stats["selected_chars"] += len(text or "")

    def summary(self, *, rounds_completed: int) -> dict[str, Any]:
        if self._architecture == "agent_loop_assist":
            return self._assist_summary(rounds_completed=rounds_completed)
        total_chars = sum(
            int(stats.get("selected_chars", 0) or 0)
            for stats in self._member_stats.values()
        )
        members: list[dict[str, Any]] = []
        for member in self._members:
            member_id = str(member.get("id") or "")
            stats = self._member_stats.get(member_id, {})
            selected_chars = int(stats.get("selected_chars", 0) or 0)
            members.append(
                {
                    "member_id": member_id,
                    "provider": str(member.get("provider") or ""),
                    "model": str(member.get("model") or ""),
                    "selected_segments": int(stats.get("selected_segments", 0) or 0),
                    "selected_chars": selected_chars,
                    "share": (selected_chars / total_chars) if total_chars > 0 else 0.0,
                }
            )
        members.sort(key=lambda item: (-float(item["share"]), str(item["member_id"])))
        return {
            "trace_id": self.trace_id,
            "architecture": self._architecture,
            "algorithm": "specem_anonymous_segment_score",
            "rounds_completed": rounds_completed,
            "member_count": len(members),
            "output_contribution": {
                "basis": "selected_output_chars",
                "members": members,
            },
        }

    def _assist_summary(self, *, rounds_completed: int) -> dict[str, Any]:
        members: list[dict[str, Any]] = []
        for member in self._members:
            member_id = str(member.get("id") or "")
            stats = self._member_stats.get(member_id, {})
            members.append(
                {
                    "member_id": member_id,
                    "provider": str(member.get("provider") or ""),
                    "model": str(member.get("model") or ""),
                    "draft_calls": int(stats.get("draft_calls", 0) or 0),
                    "verify_calls": int(stats.get("verify_calls", 0) or 0),
                    "selected_advice": int(stats.get("selected_advice", 0) or 0),
                    "input_tokens": int(stats.get("input_tokens", 0) or 0),
                    "output_tokens": int(stats.get("output_tokens", 0) or 0),
                    "reasoning_tokens": int(stats.get("reasoning_tokens", 0) or 0),
                    "cached_tokens": int(stats.get("cached_tokens", 0) or 0),
                    "cache_write_tokens": int(stats.get("cache_write_tokens", 0) or 0),
                    "billed_cost": float(stats.get("billed_cost", 0.0) or 0.0),
                }
            )
        members.sort(
            key=lambda item: (
                -int(item["selected_advice"]),
                -int(item["draft_calls"]) - int(item["verify_calls"]),
                str(item["member_id"]),
            )
        )
        return {
            "trace_id": self.trace_id,
            "architecture": "agent_loop_assist",
            "algorithm": "specem_anonymous_advice_score",
            "assist_iterations": rounds_completed,
            "rounds_completed": rounds_completed,
            "member_count": len(members),
            "action_model": self._action_model,
            "final_answer_author": "action_model",
            "assist_participation": {
                "basis": "draft_verify_selected_advice",
                "members": members,
            },
        }

    def finish(self, *, status: str, rounds_completed: int, error: str = "") -> dict[str, Any]:
        summary = self.summary(rounds_completed=rounds_completed)
        summary["status"] = status
        event = "fusion.assist.end" if self._architecture == "agent_loop_assist" else "fusion.end"
        self.emit(
            event,
            status=status,
            error=error,
            rounds_completed=rounds_completed,
            summary=summary,
            member_usage=list(self._member_stats.values()),
        )
        return summary


def _trace_path(settings: FusionTraceSettings) -> Path:
    env_log_dir = os.environ.get("OPENSQUILLA_LOG_DIR", "")
    if settings.log_dir:
        root = Path(settings.log_dir).expanduser()
    elif env_log_dir:
        root = Path(env_log_dir).expanduser()
    else:
        root = Path.home() / ".opensquilla" / "logs"
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    return root / f"fusion-traces-{stamp}.jsonl"


_SENSITIVE_KEYS = {
    "api_key",
    "authorization",
    "password",
    "proxy",
    "secret",
    "access_token",
    "refresh_token",
    "credential",
    "credentials",
}


def _redact_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, child in value.items():
            if str(key).strip().lower() in _SENSITIVE_KEYS:
                output[str(key)] = "[redacted]"
            else:
                output[str(key)] = _redact_sensitive(child)
        return output
    if isinstance(value, list):
        return [_redact_sensitive(item) for item in value]
    return value


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)
