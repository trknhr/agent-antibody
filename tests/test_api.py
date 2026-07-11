import json
from copy import deepcopy
from typing import cast

from fastapi.testclient import TestClient
from httpx import Response
from pytest import MonkeyPatch

from agent_antibody import api
from agent_antibody.api import app
from agent_antibody.portfolio_demo import run_target_demo

client = TestClient(app)


def test_healthz() -> None:
    response = cast(
        Response,
        client.get("/healthz"),  # pyright: ignore[reportUnknownMemberType]
    )
    api_response = cast(
        Response,
        client.get("/api/healthz"),  # pyright: ignore[reportUnknownMemberType]
    )

    assert response.json() == {"status": "ok"}
    assert api_response.json() == {"status": "ok"}


def test_demo_endpoint() -> None:
    response = cast(
        Response,
        client.get("/api/demo"),  # pyright: ignore[reportUnknownMemberType]
    )

    assert response.status_code == 200
    payload = cast(dict[str, object], response.json())
    vulnerable = cast(dict[str, object], payload["vulnerable"])
    protected = cast(dict[str, object], payload["protected"])
    normal = cast(dict[str, object], payload["normal"])
    assert cast(dict[str, object], vulnerable["oracle"])["status"] == "INFECTED"
    assert cast(dict[str, object], protected["oracle"])["status"] == "IMMUNE"
    assert cast(dict[str, object], normal["oracle"])["status"] == "HEALTHY"
    suite_metrics = cast(dict[str, object], payload["suite_metrics"])
    attack_cases = cast(list[dict[str, object]], payload["attack_cases"])
    assert suite_metrics["total"] == 10
    assert suite_metrics["success_before"] == 10
    assert suite_metrics["success_after"] == 0
    assert len(attack_cases) == 10
    assert all("vulnerable" not in attack for attack in attack_cases)
    assert all("protected" not in attack for attack in attack_cases)
    assert len([attack for attack in attack_cases if attack["memory_seed"] is True]) == 1
    assert all(attack["policy_blocked_after"] is True for attack in attack_cases)


def test_target_registry_and_portfolio_endpoints() -> None:
    targets_response = cast(
        Response,
        client.get("/api/targets"),  # pyright: ignore[reportUnknownMemberType]
    )
    portfolio_response = cast(
        Response,
        client.get("/api/portfolio"),  # pyright: ignore[reportUnknownMemberType]
    )

    assert [target["target_id"] for target in targets_response.json()] == [
        "opsmate",
        "repomate",
        "supportmate",
    ]
    reports = portfolio_response.json()
    assert len(reports) == 3
    assert all(report["vulnerable"]["oracle"]["status"] == "INFECTED" for report in reports)
    assert all(report["protected"]["oracle"]["status"] == "IMMUNE" for report in reports)
    assert all(report["normal"]["oracle"]["status"] == "HEALTHY" for report in reports)
    assert all(report["suite_metrics"]["total"] == 10 for report in reports)
    assert all(len(report["attack_cases"]) == 10 for report in reports)


def test_index() -> None:
    response = cast(
        Response,
        client.get("/"),  # pyright: ignore[reportUnknownMemberType]
    )

    assert response.status_code == 200
    assert "Agent Antibody" in response.text
    assert "SupportMate · refunds" in response.text
    assert "Attack suite" in response.text
    assert "Generating 10 target-specific attacks" in response.text


def test_live_endpoint_is_disabled_and_requires_json() -> None:
    missing_content_type = cast(
        Response,
        client.post("/api/live"),  # pyright: ignore[reportUnknownMemberType]
    )
    disabled = cast(
        Response,
        client.post(  # pyright: ignore[reportUnknownMemberType]
            "/api/live",
            json={},
        ),
    )

    assert missing_content_type.status_code == 415
    assert disabled.status_code == 503


def test_live_endpoint_sanitizes_unexpected_runtime_errors(monkeypatch: MonkeyPatch) -> None:
    def fail_run(*, cloud_trace_id: str | None = None) -> object:
        raise RuntimeError("provider request failed with internal details")

    monkeypatch.setattr(api.live_demo_service, "run", fail_run)
    response = cast(
        Response,
        client.post(  # pyright: ignore[reportUnknownMemberType]
            "/api/live",
            json={},
        ),
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "live Gemini evaluation failed"}


def test_live_endpoint_redacts_target_fixtures_and_tool_outputs(
    monkeypatch: MonkeyPatch,
) -> None:
    report = run_target_demo("repomate")

    def cached_run(*, cloud_trace_id: str | None = None) -> object:
        del cloud_trace_id
        return report

    monkeypatch.setattr(api.live_demo_service, "run", cached_run)
    response = cast(
        Response,
        client.post(  # pyright: ignore[reportUnknownMemberType]
            "/api/live",
            json={},
            headers={
                "Content-Type": "application/json",
            },
        ),
    )

    assert response.status_code == 200
    serialized = response.text
    assert "Run `uv sync" not in serialized
    assert '"output"' not in serialized
    assert "super-secret-value" not in serialized


def test_live_endpoint_requires_a_server_configured_bearer_token(
    monkeypatch: MonkeyPatch,
) -> None:
    report = run_target_demo("opsmate")

    def cached_run(*, cloud_trace_id: str | None = None) -> object:
        del cloud_trace_id
        return report

    monkeypatch.setenv("AGENT_ANTIBODY_LIVE_ENABLED", "true")
    monkeypatch.setenv("AGENT_ANTIBODY_LIVE_API_TOKEN", "presenter-secret")
    monkeypatch.setattr(api.live_demo_service, "run", cached_run)

    missing = cast(Response, client.post("/api/live", json={}))  # pyright: ignore[reportUnknownMemberType]
    wrong = cast(
        Response,
        client.post(  # pyright: ignore[reportUnknownMemberType]
            "/api/live",
            json={},
            headers={"Authorization": "Bearer wrong"},
        ),
    )
    accepted = cast(
        Response,
        client.post(  # pyright: ignore[reportUnknownMemberType]
            "/api/live",
            json={},
            headers={"Authorization": "Bearer presenter-secret"},
        ),
    )
    status = cast(Response, client.get("/api/live/status"))  # pyright: ignore[reportUnknownMemberType]

    assert missing.status_code == 403
    assert wrong.status_code == 403
    assert accepted.status_code == 200
    assert status.json()["configured"] is True
    assert "presenter-secret" not in status.text


def test_public_report_allowlist_removes_agent_and_state_free_text() -> None:
    secret = "token=super-secret-value"
    report = run_target_demo("supportmate")
    final_state = deepcopy(report.vulnerable.final_state)
    final_state["refunds"] = [{"refund_id": "refund-1", "reason": secret}]
    final_state["unexpected_secret"] = secret
    initial_state = {**report.vulnerable.initial_state, "email": secret}
    agent_result = report.vulnerable.agent_result.model_copy(update={"reply": secret})
    events = tuple(
        event.model_copy(update={"payload": {**event.payload, "arguments": {"reason": secret}}})
        for event in report.vulnerable.events
    )
    vulnerable = report.vulnerable.model_copy(
        update={
            "initial_state": initial_state,
            "final_state": final_state,
            "agent_result": agent_result,
            "events": events,
        }
    )
    request = report.evidence.requests[0].model_copy(
        update={"arguments": {"body": secret, "reason": secret}}
    )
    evidence = report.evidence.model_copy(
        update={"requests": (request, *report.evidence.requests[1:])}
    )
    unsafe_report = report.model_copy(
        update={
            "vulnerable": vulnerable,
            "evidence": evidence,
            "antibody_proposal": report.antibody_proposal.model_copy(update={"rationale": secret}),
            "antibody": report.antibody.model_copy(update={"rationale": secret}),
        }
    )
    unsafe_plan = report.attack_results[0].plan.model_copy(update={"payload": secret})
    unsafe_attack_result = report.attack_results[0].model_copy(
        update={"plan": unsafe_plan, "vulnerable": vulnerable}
    )
    unsafe_report = unsafe_report.model_copy(
        update={
            "attack_results": (unsafe_attack_result, *report.attack_results[1:]),
            "selected_attack": unsafe_plan,
        }
    )

    public = api.public_live_report(unsafe_report)
    serialized = json.dumps(public)

    assert secret not in serialized
    public_vulnerable = cast(dict[str, object], public["vulnerable"])
    public_agent_result = cast(dict[str, object], public_vulnerable["agent_result"])
    assert "reply" not in public_agent_result
    public_evidence = cast(dict[str, object], public["evidence"])
    assert "arguments" not in cast(list[dict[str, object]], public_evidence["requests"])[0]
    public_cases = cast(list[dict[str, object]], public["attack_cases"])
    assert len(public_cases) == 10
    assert "agent_result" not in public_cases[0]
    assert "events" not in public_cases[0]
    assert "initial_state" not in public_vulnerable
    assert "final_state" not in public_vulnerable
    assert set(cast(dict[str, object], public["target"])) == {"target_id", "name"}
