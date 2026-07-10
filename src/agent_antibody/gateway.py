from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from agent_antibody.contracts import (
    DecisionEffect,
    EventType,
    GetServiceStatusArgs,
    JsonValue,
    PolicyMode,
    PostIssueArgs,
    ReadLogsArgs,
    ReadRunbookArgs,
    ScaleServiceArgs,
    SignedApproval,
    SignedTaskContract,
    SourceKind,
    ToolCallResult,
    ToolName,
)
from agent_antibody.policy import PolicyEngine
from agent_antibody.signing import ContractSigner
from agent_antibody.simulator import OpsSimulator
from agent_antibody.trace import TraceRecorder

type ArgumentModel = (
    type[GetServiceStatusArgs]
    | type[ReadLogsArgs]
    | type[ReadRunbookArgs]
    | type[ScaleServiceArgs]
    | type[PostIssueArgs]
)

_ARGUMENT_MODELS: dict[ToolName, ArgumentModel] = {
    ToolName.GET_SERVICE_STATUS: GetServiceStatusArgs,
    ToolName.READ_LOGS: ReadLogsArgs,
    ToolName.READ_RUNBOOK: ReadRunbookArgs,
    ToolName.SCALE_SERVICE: ScaleServiceArgs,
    ToolName.POST_ISSUE: PostIssueArgs,
}

_SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|password|token)\s*[=:]\s*\S+"),
    re.compile(r"\b(?:sk|AIza)[-_A-Za-z0-9]{12,}\b"),
)


def _redact_sensitive(text: str) -> tuple[str, int]:
    redacted = text
    count = 0
    for pattern in _SECRET_PATTERNS:
        redacted, replacements = pattern.subn("[REDACTED]", redacted)
        count += replacements
    return redacted, count


def _redact_json(value: JsonValue) -> tuple[JsonValue, int]:
    if isinstance(value, str):
        return _redact_sensitive(value)
    if isinstance(value, list):
        redacted_items: list[JsonValue] = []
        count = 0
        for item in value:
            redacted, replacements = _redact_json(item)
            redacted_items.append(redacted)
            count += replacements
        return redacted_items, count
    if isinstance(value, dict):
        redacted_object: dict[str, JsonValue] = {}
        count = 0
        for key, item in value.items():
            redacted, replacements = _redact_json(item)
            redacted_object[key] = redacted
            count += replacements
        return redacted_object, count
    return value, 0


def _json_arguments(model: BaseModel) -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], model.model_dump(mode="json"))


class PolicyGateway:
    def __init__(
        self,
        *,
        mode: PolicyMode,
        simulator: OpsSimulator,
        recorder: TraceRecorder,
        contract_signer: ContractSigner,
        policy: PolicyEngine,
    ) -> None:
        self.mode = mode
        self._simulator = simulator
        self._recorder = recorder
        self._contract_signer = contract_signer
        self._policy = policy
        self._usage: dict[ToolName, int] = {}
        self._observed_sources: dict[str, SourceKind] = {}
        self._consumed_approval_ids: set[str] = set()

    def call(
        self,
        *,
        caller_id: str,
        signed_contract: SignedTaskContract,
        tool: ToolName,
        arguments: dict[str, JsonValue],
        approval: SignedApproval | None = None,
    ) -> ToolCallResult:
        request_id = str(uuid4())
        contract = signed_contract.contract

        try:
            normalized, redaction_count = self._normalize_arguments(tool, arguments)
        except ValidationError:
            reasons = ("invalid_arguments",)
            safe_arguments, redaction_count = self._safe_trace_arguments(tool, arguments)
            self._record_requested(request_id, tool, safe_arguments, redaction_count)
            self._record_decision(
                request_id=request_id,
                tool=tool,
                effect=DecisionEffect.DENY,
                policy_allowed=False,
                reasons=reasons,
            )
            self._record_outcome(request_id, tool, "denied", reasons)
            return ToolCallResult(request_id=request_id, status="denied", reasons=reasons)

        self._record_requested(request_id, tool, normalized, redaction_count)

        auth_reasons: list[str] = []
        if not self._contract_signer.verify(signed_contract):
            auth_reasons.append("task_contract_signature_invalid")
        if caller_id != contract.caller_id:
            auth_reasons.append("caller_not_bound_to_task")
        if contract.issued_at > datetime.now(UTC):
            auth_reasons.append("task_contract_not_yet_valid")
        if contract.expires_at <= datetime.now(UTC):
            auth_reasons.append("task_contract_expired")
        if approval is not None and approval.grant.approval_id in self._consumed_approval_ids:
            auth_reasons.append("approval_replayed")

        evaluation = self._policy.evaluate(
            contract=contract,
            tool=tool,
            arguments=normalized,
            usage=self._usage,
            approval=approval,
            observed_source_kinds=tuple(self._observed_sources.values()),
        )
        reasons = tuple(dict.fromkeys((*auth_reasons, *evaluation.reasons)))
        authenticated = not auth_reasons
        policy_allowed = evaluation.allowed and authenticated

        if policy_allowed:
            effect = DecisionEffect.ALLOW
            effective_allowed = True
        elif authenticated and self.mode == PolicyMode.AUDIT:
            effect = DecisionEffect.AUDIT_ALLOW
            effective_allowed = True
        else:
            effect = DecisionEffect.DENY
            effective_allowed = False

        self._record_decision(
            request_id=request_id,
            tool=tool,
            effect=effect,
            policy_allowed=policy_allowed,
            reasons=reasons,
            requires_approval=evaluation.requires_approval,
            matched_rule_ids=evaluation.matched_rule_ids,
            untrusted_source_kinds=evaluation.untrusted_source_kinds,
        )

        if not effective_allowed:
            self._record_outcome(request_id, tool, "denied", reasons)
            return ToolCallResult(request_id=request_id, status="denied", reasons=reasons)

        if (
            evaluation.requires_approval
            and approval is not None
            and "approval_invalid" not in reasons
        ):
            self._consumed_approval_ids.add(approval.grant.approval_id)

        before = self._simulator.snapshot()
        try:
            output = self._simulator.execute(tool, normalized)
        except (KeyError, ValueError) as error:
            failure_reasons = (f"execution_failed:{type(error).__name__}",)
            self._record_outcome(request_id, tool, "failed", failure_reasons)
            return ToolCallResult(request_id=request_id, status="failed", reasons=failure_reasons)

        self._usage[tool] = self._usage.get(tool, 0) + 1
        after = self._simulator.snapshot()
        self._observe_source(output)
        safe_arguments_value, argument_redaction_count = _redact_json(normalized)
        safe_output, output_redaction_count = _redact_json(output)
        safe_arguments = cast(dict[str, JsonValue], safe_arguments_value)
        self._recorder.record(
            EventType.TOOL_EXECUTED,
            request_id=request_id,
            tool=tool,
            payload={
                "arguments": safe_arguments,
                "output": safe_output,
                "redaction_count": argument_redaction_count + output_redaction_count,
            },
        )
        if before != after:
            self._recorder.record(
                EventType.STATE_CHANGED,
                request_id=request_id,
                tool=tool,
                payload={
                    "before": before.model_dump(mode="json"),
                    "after": after.model_dump(mode="json"),
                },
            )
        self._record_outcome(request_id, tool, "executed", ())
        return ToolCallResult(request_id=request_id, status="executed", output=output)

    def _normalize_arguments(
        self,
        tool: ToolName,
        arguments: dict[str, JsonValue],
    ) -> tuple[dict[str, JsonValue], int]:
        parsed = _ARGUMENT_MODELS[tool].model_validate(arguments)
        normalized = _json_arguments(parsed)
        redaction_count = 0
        if tool == ToolName.POST_ISSUE:
            body = cast(str, normalized["body"])
            redacted, redaction_count = _redact_sensitive(body)
            normalized["body"] = redacted
        return normalized, redaction_count

    def _safe_trace_arguments(
        self,
        tool: ToolName,
        arguments: dict[str, JsonValue],
    ) -> tuple[dict[str, JsonValue], int]:
        """Redact sensitive text even when argument validation fails."""
        safe_arguments = dict(arguments)
        redacted, redaction_count = _redact_json(safe_arguments)
        return cast(dict[str, JsonValue], redacted), redaction_count

    def _record_requested(
        self,
        request_id: str,
        tool: ToolName,
        arguments: dict[str, JsonValue],
        redaction_count: int = 0,
    ) -> None:
        safe_arguments_value, extra_redactions = _redact_json(arguments)
        safe_arguments = cast(dict[str, JsonValue], safe_arguments_value)
        source_refs: list[JsonValue] = [source for source in sorted(self._observed_sources)]
        source_kinds: list[JsonValue] = [
            kind for kind in sorted({kind.value for kind in self._observed_sources.values()})
        ]
        payload: dict[str, JsonValue] = {
            "arguments": safe_arguments,
            "source_refs": source_refs,
            "source_kinds": source_kinds,
            "redaction_count": redaction_count + extra_redactions,
        }
        self._recorder.record(
            EventType.TOOL_REQUESTED,
            request_id=request_id,
            tool=tool,
            payload=payload,
        )

    def _record_decision(
        self,
        *,
        request_id: str,
        tool: ToolName,
        effect: DecisionEffect,
        policy_allowed: bool,
        reasons: tuple[str, ...],
        requires_approval: bool = False,
        matched_rule_ids: tuple[str, ...] = (),
        untrusted_source_kinds: tuple[SourceKind, ...] = (),
    ) -> None:
        self._recorder.record(
            EventType.POLICY_DECISION,
            request_id=request_id,
            tool=tool,
            payload={
                "mode": self.mode.value,
                "effect": effect.value,
                "policy_allowed": policy_allowed,
                "reasons": list(reasons),
                "requires_approval": requires_approval,
                "matched_rule_ids": list(matched_rule_ids),
                "untrusted_source_kinds": [kind.value for kind in untrusted_source_kinds],
            },
        )

    def _observe_source(self, output: JsonValue) -> None:
        if not isinstance(output, dict) or output.get("trust") != "untrusted":
            return
        source_id = output.get("source_id")
        raw_kind = output.get("source_kind")
        if not isinstance(source_id, str) or not isinstance(raw_kind, str):
            return
        try:
            source_kind = SourceKind(raw_kind)
        except ValueError:
            return
        self._observed_sources[source_id] = source_kind

    def _record_outcome(
        self,
        request_id: str,
        tool: ToolName,
        status: str,
        reasons: tuple[str, ...],
    ) -> None:
        self._recorder.record(
            EventType.EXECUTION_OUTCOME,
            request_id=request_id,
            tool=tool,
            payload={"status": status, "reasons": list(reasons)},
        )
