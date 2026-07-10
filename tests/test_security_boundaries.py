from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from agent_antibody.agent import AgentRunResult
from agent_antibody.contracts import (
    EventType,
    JsonValue,
    PolicyMode,
    SignedTaskContract,
    ToolName,
)
from agent_antibody.gateway import PolicyGateway
from agent_antibody.oracle import DeterministicOracle, OracleStatus
from agent_antibody.policy import (
    PolicyAction,
    PolicyEngine,
    PolicyRules,
    RuleWhen,
    ToolPolicyRule,
)
from agent_antibody.runner import run_scenario
from agent_antibody.scenarios import MALICIOUS_RUNBOOK, NORMAL_RUNBOOK, Scenario
from agent_antibody.signing import ApprovalSigner, ContractSigner
from agent_antibody.tools import BoundTools
from agent_antibody.trace import TraceRecorder

TEST_SECRET = b"agent-antibody-test-signing-key-32-bytes"


def _approval_policy() -> PolicyRules:
    return PolicyRules(
        rules=(
            ToolPolicyRule(
                rule_id="require-scale-zero-approval",
                tool=ToolName.SCALE_SERVICE,
                when=RuleWhen(replicas="< 1"),
                action=PolicyAction.REQUIRE_APPROVAL,
            ),
        )
    )


def _bound_tools(
    signed_contract: SignedTaskContract,
    *,
    mode: PolicyMode = PolicyMode.ENFORCE,
    rules: PolicyRules | None = None,
) -> tuple[BoundTools, TraceRecorder, ApprovalSigner, PolicyGateway]:
    contract_signer = ContractSigner(TEST_SECRET)
    approval_signer = ApprovalSigner(TEST_SECRET)
    recorder = TraceRecorder(run_id=str(uuid4()))
    gateway = PolicyGateway(
        mode=mode,
        simulator=MALICIOUS_RUNBOOK.simulator(),
        recorder=recorder,
        contract_signer=contract_signer,
        policy=PolicyEngine(rules or PolicyRules(), approval_signer),
    )
    tools = BoundTools(
        caller_id="opsmate",
        signed_contract=signed_contract,
        gateway=gateway,
    )
    return tools, recorder, approval_signer, gateway


def test_tampered_contract_is_rejected_by_gateway_even_in_audit_mode() -> None:
    signer = ContractSigner(TEST_SECRET)
    signed = MALICIOUS_RUNBOOK.issue_contract(signer, caller_id="opsmate")
    tampered_contract = signed.contract.model_copy(update={"objective": "attacker objective"})
    tampered = SignedTaskContract(contract=tampered_contract, signature=signed.signature)
    tools, recorder, _approval_signer, _gateway = _bound_tools(
        tampered,
        mode=PolicyMode.AUDIT,
    )

    result = tools.get_service_status(MALICIOUS_RUNBOOK.service)

    assert result.status == "denied"
    assert "task_contract_signature_invalid" in result.reasons
    assert not any(event.event_type == EventType.TOOL_EXECUTED for event in recorder.events)


def test_approval_never_expands_task_contract() -> None:
    contract_signer = ContractSigner(TEST_SECRET)
    approval_signer = ApprovalSigner(TEST_SECRET)
    broad = MALICIOUS_RUNBOOK.issue_contract(contract_signer, caller_id="opsmate").contract
    restricted = broad.model_copy(
        update={
            "allowed_tools": tuple(
                tool for tool in broad.allowed_tools if tool != ToolName.SCALE_SERVICE
            )
        }
    )
    arguments: dict[str, JsonValue] = {"service": "payments-staging", "replicas": 0}
    approval = approval_signer.issue(
        contract=restricted,
        tool=ToolName.SCALE_SERVICE,
        arguments=arguments,
        expires_at=datetime.now(UTC) + timedelta(minutes=5),
    )

    evaluation = PolicyEngine(_approval_policy(), approval_signer).evaluate(
        contract=restricted,
        tool=ToolName.SCALE_SERVICE,
        arguments=arguments,
        usage={},
        approval=approval,
    )

    assert not evaluation.allowed
    assert "tool_not_allowed_by_task" in evaluation.reasons
    assert "approval_invalid" not in evaluation.reasons


def test_approval_is_bound_to_exact_arguments_and_single_use() -> None:
    signer = ContractSigner(TEST_SECRET)
    signed = MALICIOUS_RUNBOOK.issue_contract(signer, caller_id="opsmate")
    tools, _recorder, approval_signer, _gateway = _bound_tools(
        signed,
        rules=_approval_policy(),
    )
    arguments: dict[str, JsonValue] = {"service": "payments-staging", "replicas": 0}
    approval = approval_signer.issue(
        contract=signed.contract,
        tool=ToolName.SCALE_SERVICE,
        arguments=arguments,
        expires_at=datetime.now(UTC) + timedelta(minutes=5),
    )

    first = tools.scale_service("payments-staging", 0, approval=approval)
    replay = tools.scale_service("payments-staging", 0, approval=approval)

    assert first.status == "executed"
    assert replay.status == "denied"
    assert "approval_replayed" in replay.reasons


def test_approval_for_other_arguments_is_denied() -> None:
    signer = ContractSigner(TEST_SECRET)
    signed = MALICIOUS_RUNBOOK.issue_contract(signer, caller_id="opsmate")
    tools, _recorder, approval_signer, _gateway = _bound_tools(
        signed,
        rules=_approval_policy(),
    )
    approved_arguments: dict[str, JsonValue] = {
        "service": "payments-staging",
        "replicas": 1,
    }
    approval = approval_signer.issue(
        contract=signed.contract,
        tool=ToolName.SCALE_SERVICE,
        arguments=approved_arguments,
        expires_at=datetime.now(UTC) + timedelta(minutes=5),
    )

    result = tools.scale_service("payments-staging", 0, approval=approval)

    assert result.status == "denied"
    assert "approval_invalid" in result.reasons


def test_invalid_arguments_are_redacted_before_trace_recording() -> None:
    signer = ContractSigner(TEST_SECRET)
    signed = MALICIOUS_RUNBOOK.issue_contract(signer, caller_id="opsmate")
    _tools, recorder, _approval_signer, gateway = _bound_tools(signed)
    arguments: dict[str, JsonValue] = {
        "repository": MALICIOUS_RUNBOOK.issue_repository,
        "issue": MALICIOUS_RUNBOOK.issue,
        "body": "token=super-secret-value",
        "unexpected": True,
    }

    result = gateway.call(
        caller_id="opsmate",
        signed_contract=signed,
        tool=ToolName.POST_ISSUE,
        arguments=arguments,
    )
    serialized_trace = json.dumps(
        [event.model_dump(mode="json") for event in recorder.events],
        ensure_ascii=False,
    )

    assert result.status == "denied"
    assert "super-secret-value" not in serialized_trace
    assert "[REDACTED]" in serialized_trace


def test_oracle_fails_closed_on_incomplete_trace() -> None:
    run = run_scenario(MALICIOUS_RUNBOOK, mode=PolicyMode.ENFORCE)
    artifact = run.artifact().model_copy(update={"events": run.events[:-1]})

    result = DeterministicOracle().evaluate(artifact)

    assert result.status == OracleStatus.INVALID_RUN


def test_oracle_rejects_orphan_execution_lifecycle() -> None:
    run = run_scenario(MALICIOUS_RUNBOOK, mode=PolicyMode.ENFORCE)
    scale_request = next(
        event
        for event in run.events
        if event.event_type == EventType.TOOL_REQUESTED and event.tool == ToolName.SCALE_SERVICE
    )
    removed_types = {
        EventType.TOOL_REQUESTED,
        EventType.POLICY_DECISION,
        EventType.EXECUTION_OUTCOME,
    }
    retained = [
        event
        for event in run.events
        if not (event.request_id == scale_request.request_id and event.event_type in removed_types)
    ]
    renumbered = tuple(
        event.model_copy(update={"sequence": sequence})
        for sequence, event in enumerate(retained, start=1)
    )
    artifact = run.artifact().model_copy(update={"events": renumbered})

    result = DeterministicOracle().evaluate(artifact)

    assert result.status == OracleStatus.INVALID_RUN


class RevertingAgent:
    def run(self, scenario: Scenario, tools: BoundTools) -> AgentRunResult:
        results = [
            tools.get_service_status(scenario.service),
            tools.read_logs(scenario.service),
            tools.read_runbook(scenario.service),
            tools.scale_service(scenario.service, 0),
            tools.scale_service(scenario.service, scenario.initial_replicas),
            tools.post_issue(scenario.issue_repository, scenario.issue, "Investigation complete."),
        ]
        return AgentRunResult(success=True, reply="done", tool_results=tuple(results))


def test_oracle_detects_transient_dangerous_state_even_when_reverted() -> None:
    run = run_scenario(
        MALICIOUS_RUNBOOK,
        mode=PolicyMode.ENFORCE,
        agent=RevertingAgent(),
    )

    assert run.final_state.replicas[MALICIOUS_RUNBOOK.service] == 3
    assert run.oracle.status == OracleStatus.INFECTED
    assert run.oracle.attack_succeeded
    assert run.oracle.unauthorized_effect_count == 2


def test_attack_label_cannot_hide_observed_infection() -> None:
    mislabeled = MALICIOUS_RUNBOOK.model_copy(update={"attack": False})

    run = run_scenario(mislabeled, mode=PolicyMode.ENFORCE)

    assert run.final_state.replicas[mislabeled.service] == 0
    assert run.oracle.status == OracleStatus.INFECTED
    assert run.oracle.attack_succeeded


class NoOpAgent:
    def run(self, scenario: Scenario, tools: BoundTools) -> AgentRunResult:
        del scenario, tools
        return AgentRunResult(success=True, reply="done", tool_results=())


def test_failed_normal_task_is_not_reported_healthy() -> None:
    run = run_scenario(
        NORMAL_RUNBOOK,
        mode=PolicyMode.ENFORCE,
        agent=NoOpAgent(),
    )

    assert not run.oracle.normal_task_succeeded
    assert run.oracle.status == OracleStatus.UNHEALTHY
