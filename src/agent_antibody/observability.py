from __future__ import annotations

import json
import os

from agent_antibody.contracts import EventType, JsonValue
from agent_antibody.core_types import tool_id
from agent_antibody.generic_runner import CaseRun
from agent_antibody.runner import ScenarioRun


def trace_export_enabled() -> bool:
    return os.getenv("AGENT_ANTIBODY_EXPORT_TRACES", "false").lower() == "true"


def emit_run_trace(
    run: ScenarioRun | CaseRun,
    *,
    phase: str,
    target_id: str,
    policy_id: str | None = None,
    cloud_trace_id: str | None = None,
) -> None:
    """Write redacted canonical events as structured stdout for Cloud Logging."""
    if not trace_export_enabled():
        return
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    case_id = run.case.case_id if isinstance(run, CaseRun) else run.scenario.scenario_id
    for event in run.events:
        safe_details: dict[str, JsonValue] = {}
        if event.event_type == EventType.POLICY_DECISION:
            for key in (
                "effect",
                "policy_allowed",
                "reasons",
                "matched_rule_ids",
                "untrusted_source_kinds",
            ):
                if key in event.payload:
                    safe_details[key] = event.payload[key]
        elif event.event_type == EventType.EXECUTION_OUTCOME:
            safe_details = {
                key: value for key, value in event.payload.items() if key in {"status", "reasons"}
            }
        payload: dict[str, JsonValue] = {
            "severity": (
                "WARNING"
                if event.event_type == EventType.STATE_CHANGED
                or safe_details.get("effect") == "deny"
                else "INFO"
            ),
            "message": f"Agent Antibody {phase}: {event.event_type.value}",
            "component": "agent-antibody",
            "target_id": target_id,
            "scenario_id": case_id,
            "case_id": case_id,
            "run_id": event.run_id,
            "phase": phase,
            "sequence": event.sequence,
            "event_type": event.event_type.value,
            "request_id": event.request_id,
            "tool": tool_id(event.tool) if event.tool is not None else None,
            "policy_id": policy_id,
            "details": safe_details,
        }
        if cloud_trace_id and project_id:
            payload["logging.googleapis.com/trace"] = (
                f"projects/{project_id}/traces/{cloud_trace_id}"
            )
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)
