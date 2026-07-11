from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from secrets import compare_digest
from typing import Annotated, Literal, cast

from fastapi import Body, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict

from agent_antibody.attack_campaign import AttackCaseResult
from agent_antibody.contracts import TraceEvent
from agent_antibody.core_types import JsonObject, JsonValue
from agent_antibody.generic_runner import CaseRun
from agent_antibody.immunity_artifacts import verify_snapshot
from agent_antibody.live_pipeline import LivePipelineError, LivePipelineReport
from agent_antibody.live_service import LiveDemoBusyError, LiveDemoState, live_demo_service
from agent_antibody.portfolio_demo import TargetDemoReport, run_target_demo
from agent_antibody.targets.registry import TARGET_ADAPTERS

app = FastAPI(title="Agent Antibody", version="0.1.0")
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class LiveDemoRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_id: Literal["opsmate", "repomate", "supportmate"] = "opsmate"


def _public_event(event: TraceEvent) -> JsonObject:
    payload = event.payload
    safe_payload: JsonObject = {
        key: payload[key]
        for key in (
            "effect",
            "reasons",
            "status",
            "source_kinds",
            "untrusted_source_kinds",
            "approval_verified",
            "success",
        )
        if key in payload
    }
    arguments = payload.get("arguments")
    if isinstance(arguments, dict):
        safe_payload["argument_fields"] = cast(list[JsonValue], sorted(arguments))
    return {
        "sequence": event.sequence,
        "event_type": event.event_type.value,
        "tool": event.tool,
        "request_id": event.request_id,
        "payload": safe_payload,
    }


def _public_attack_plan(plan: JsonObject) -> JsonObject:
    public_plan = {
        key: plan[key]
        for key in (
            "plan_id",
            "target_id",
            "surface_id",
            "expected_tool",
            "technique",
        )
        if key in plan
    }
    return public_plan


def _state_summary(run: CaseRun) -> JsonObject:
    target_id = run.case.target_id
    if target_id == "opsmate":
        service = str(run.case.fixtures.get("service", "payments-staging"))
        before = run.initial_state.get("replicas")
        after = run.final_state.get("replicas")
        before_replicas = before.get(service) if isinstance(before, dict) else None
        after_replicas = after.get(service) if isinstance(after, dict) else None
        issues = run.final_state.get("issues")
        return {
            "primary_label": "Replicas",
            "before": before_replicas,
            "after": after_replicas,
            "artifact_label": "Issues",
            "artifact_count": len(issues) if isinstance(issues, list) else 0,
        }
    if target_id == "repomate":
        before = run.initial_state.get("files")
        after = run.final_state.get("files")
        comments = run.final_state.get("comments")
        return {
            "primary_label": "README",
            "before": "present"
            if isinstance(before, dict) and "README.md" in before
            else "missing",
            "after": "present" if isinstance(after, dict) and "README.md" in after else "missing",
            "artifact_label": "Comments",
            "artifact_count": len(comments) if isinstance(comments, list) else 0,
        }
    order_id = str(run.case.fixtures.get("order_id", "ORD-1042"))
    before = run.initial_state.get("orders")
    after = run.final_state.get("orders")
    before_order = before.get(order_id) if isinstance(before, dict) else None
    after_order = after.get(order_id) if isinstance(after, dict) else None
    replies = run.final_state.get("replies")
    return {
        "primary_label": "Refund cents",
        "before": before_order.get("refunded_cents") if isinstance(before_order, dict) else None,
        "after": after_order.get("refunded_cents") if isinstance(after_order, dict) else None,
        "artifact_label": "Replies",
        "artifact_count": len(replies) if isinstance(replies, list) else 0,
    }


def _public_run(run: CaseRun) -> JsonObject:
    public_events = [_public_event(event) for event in run.events]
    public_results = [
        {
            "request_id": result.request_id,
            "status": result.status,
            "reasons": list(result.reasons),
        }
        for result in run.agent_result.tool_results
    ]
    oracle = run.oracle
    return {
        "mode": run.mode.value,
        "case": {
            "case_id": run.case.case_id,
            "target_id": run.case.target_id,
            "attack": run.case.attack,
        },
        "summary": _state_summary(run),
        "events": cast(list[JsonValue], public_events),
        "agent_result": {
            "success": run.agent_result.success,
            "tool_results": cast(list[JsonValue], public_results),
        },
        "oracle": {
            "status": oracle.status.value,
            "attack_succeeded": oracle.attack_succeeded,
            "normal_task_succeeded": oracle.normal_task_succeeded,
            "dangerous_request_count": oracle.dangerous_request_count,
            "unauthorized_effect_count": oracle.unauthorized_effect_count,
        },
    }


def _public_attack_case(result: AttackCaseResult) -> JsonObject:
    return {
        "plan": _public_attack_plan(result.plan.model_dump(mode="json")),
        "vulnerable_status": result.vulnerable.oracle.status.value,
        "protected_status": result.protected.oracle.status.value,
        "attack_succeeded_before": result.vulnerable.oracle.attack_succeeded,
        "attack_succeeded_after": result.protected.oracle.attack_succeeded,
        "request_observed_before": result.request_observed_before,
        "request_observed_after": result.request_observed_after,
        "policy_blocked_after": result.policy_blocked_after,
        "memory_seed": result.memory_seed,
    }


def public_live_report(report: LivePipelineReport | TargetDemoReport) -> JsonObject:
    """Build the public demo contract from an explicit allowlist."""

    evidence = report.evidence
    payload: JsonObject = {
        "generated_at": report.generated_at.isoformat(),
        "target": {"target_id": report.target.target_id, "name": report.target.name},
        "suite_metrics": report.suite_metrics.model_dump(mode="json"),
        "attack_cases": cast(
            list[JsonValue], [_public_attack_case(result) for result in report.attack_results]
        ),
        "selected_attack": _public_attack_plan(report.selected_attack.model_dump(mode="json")),
        "vulnerable": _public_run(report.vulnerable),
        "protected": _public_run(report.protected),
        "normal": _public_run(report.normal),
        "evidence": {
            "target_id": evidence.target_id,
            "scenario_id": evidence.scenario_id,
            "oracle_status": evidence.oracle_status,
            "findings": list(evidence.findings),
            "causal_request_ids": list(evidence.causal_request_ids),
            "requests": [
                {
                    "request_id": request.request_id,
                    "tool": request.tool,
                    "source_kinds": list(request.source_kinds),
                    "policy_effect": request.policy_effect,
                    "executed": request.executed,
                    "changed_state": request.changed_state,
                }
                for request in evidence.requests
            ],
        },
        "antibody": {
            "bundle_id": report.antibody.bundle_id,
            "target_id": report.antibody.target_id,
            "rules": [rule.model_dump(mode="json") for rule in report.antibody.rules],
            "untrusted_sources": list(report.antibody.untrusted_sources),
        },
        "acceptance_passed": report.acceptance_passed,
    }
    if isinstance(report, LivePipelineReport):
        payload["model"] = report.model
        payload["attack_triggered"] = report.attack_triggered
    return payload


def _verify_live_authorization(authorization: str | None) -> None:
    if not live_demo_service.enabled:
        return
    expected = live_demo_service.api_token
    if expected is None:
        raise HTTPException(
            status_code=503,
            detail="live Gemini authentication is not configured",
        )
    scheme, separator, token = (authorization or "").partition(" ")
    if separator != " " or scheme.lower() != "bearer" or not compare_digest(token, expected):
        raise HTTPException(status_code=403, detail="valid live demo bearer token is required")


@lru_cache(maxsize=len(TARGET_ADAPTERS))
def _cached_target_demo(target_id: str) -> TargetDemoReport:
    return run_target_demo(target_id)


@app.get("/healthz")
@app.get("/api/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/demo")
def demo(target_id: Literal["opsmate", "repomate", "supportmate"] = "opsmate") -> JsonObject:
    return public_live_report(_cached_target_demo(target_id))


@app.get("/api/portfolio")
def portfolio() -> list[JsonObject]:
    return [public_live_report(_cached_target_demo(target_id)) for target_id in TARGET_ADAPTERS]


@app.get("/api/immunity-snapshot")
def immunity_snapshot() -> JsonObject:
    """Read the committed CI artifact; never generate or expose a live run here."""

    try:
        snapshot = verify_snapshot(repository_root=_REPOSITORY_ROOT)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503, detail="immunity snapshot has not been generated"
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=503, detail="immunity snapshot is invalid") from error
    return cast(JsonObject, snapshot.model_dump(mode="json"))


@app.get("/api/live/status")
def live_status() -> LiveDemoState:
    return live_demo_service.state()


@app.get("/api/targets")
def targets() -> list[dict[str, object]]:
    return [
        {
            "target_id": target_id,
            "name": adapter.manifest.name,
            "description": adapter.manifest.description,
            "injection_surfaces": [
                surface.surface_id for surface in adapter.manifest.injection_surfaces
            ],
            "dangerous_tools": [tool.name for tool in adapter.manifest.tools if tool.mutates_state],
        }
        for target_id, adapter in TARGET_ADAPTERS.items()
    ]


@app.post("/api/live")
def live_demo(
    request: Request,
    payload: Annotated[LiveDemoRequest | None, Body()] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> JsonObject:
    if request.headers.get("content-type", "").split(";", maxsplit=1)[0] != "application/json":
        raise HTTPException(status_code=415, detail="application/json is required")
    _verify_live_authorization(authorization)
    trace_context = request.headers.get("x-cloud-trace-context")
    trace_id = trace_context.split("/", maxsplit=1)[0] if trace_context else None
    target_id = payload.target_id if payload is not None else "opsmate"
    try:
        if target_id == "opsmate":
            report = live_demo_service.run(cloud_trace_id=trace_id)
        else:
            report = live_demo_service.run(target_id=target_id, cloud_trace_id=trace_id)
        return public_live_report(report)
    except PermissionError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except LiveDemoBusyError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except LivePipelineError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail="live Gemini evaluation failed") from error


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = files("agent_antibody").joinpath("static/index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)
