from __future__ import annotations

import copy
import re
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

from pydantic import ValidationError

from agent_antibody.contracts import (
    DecisionEffect,
    EventType,
    JsonValue,
    PolicyMode,
    SignedApproval,
    SignedTaskContract,
    ToolCallResult,
)
from agent_antibody.core_types import JsonObject, SourceId, ToolId
from agent_antibody.policy import PolicyEngine
from agent_antibody.signing import ContractSigner
from agent_antibody.targets.base import TargetRuntime, ToolInvoker
from agent_antibody.trace import TraceRecorder

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


def _snapshot(runtime: TargetRuntime) -> JsonObject:
    return copy.deepcopy(runtime.snapshot())


class GenericPolicyGateway:
    """Target-neutral, fail-closed policy and trace boundary."""

    def __init__(
        self,
        *,
        mode: PolicyMode,
        runtime: TargetRuntime,
        recorder: TraceRecorder,
        contract_signer: ContractSigner,
        policy: PolicyEngine,
    ) -> None:
        self.mode = mode
        self._runtime = runtime
        self._recorder = recorder
        self._contract_signer = contract_signer
        self._policy = policy
        self._usage: dict[ToolId, int] = {}
        self._observed_sources: dict[str, SourceId] = {}
        self._consumed_approval_ids: set[str] = set()

    def call(
        self,
        *,
        caller_id: str,
        signed_contract: SignedTaskContract,
        tool: ToolId,
        arguments: JsonObject,
        approval: SignedApproval | None = None,
    ) -> ToolCallResult:
        request_id = str(uuid4())
        try:
            normalized = self._runtime.validate_arguments(tool, copy.deepcopy(arguments))
        except KeyError, TypeError, ValueError, ValidationError:
            reasons = ("invalid_arguments",)
            self._record_requested(request_id, tool, arguments)
            self._record_decision(
                request_id=request_id,
                tool=tool,
                effect=DecisionEffect.DENY,
                policy_allowed=False,
                reasons=reasons,
            )
            self._record_outcome(request_id, tool, "denied", reasons)
            return ToolCallResult(request_id=request_id, status="denied", reasons=reasons)

        self._record_requested(request_id, tool, normalized)
        contract = signed_contract.contract
        now = datetime.now(UTC)
        auth_reasons: list[str] = []
        if not self._contract_signer.verify(signed_contract):
            auth_reasons.append("task_contract_signature_invalid")
        if caller_id != contract.caller_id:
            auth_reasons.append("caller_not_bound_to_task")
        if contract.issued_at > now:
            auth_reasons.append("task_contract_not_yet_valid")
        if contract.expires_at <= now:
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
            now=now,
        )
        reasons = tuple(dict.fromkeys((*auth_reasons, *evaluation.reasons)))
        authenticated = not auth_reasons
        policy_allowed = authenticated and evaluation.allowed
        if policy_allowed:
            effect = DecisionEffect.ALLOW
            effective_allowed = True
        elif authenticated and self.mode == PolicyMode.AUDIT and not evaluation.hard_denial:
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
            approval_verified=(
                evaluation.requires_approval
                and approval is not None
                and "approval_invalid" not in reasons
                and policy_allowed
            ),
        )
        if not effective_allowed:
            self._record_outcome(request_id, tool, "denied", reasons)
            return ToolCallResult(request_id=request_id, status="denied", reasons=reasons)

        if policy_allowed and evaluation.requires_approval and approval is not None:
            self._consumed_approval_ids.add(approval.grant.approval_id)

        before = _snapshot(self._runtime)
        execution_error: Exception | None = None
        execution_output: JsonValue = None
        observed_sources: tuple[tuple[str, SourceId], ...] = ()
        try:
            execution = self._runtime.execute(tool, copy.deepcopy(normalized))
            execution_output = execution.output
            observed_sources = tuple(
                (source.source_id, source.source_kind) for source in execution.observed_sources
            )
        except Exception as error:
            execution_error = error
        after = _snapshot(self._runtime)

        safe_arguments_value, argument_redactions = _redact_json(normalized)
        safe_output, output_redactions = _redact_json(execution_output)
        safe_arguments = cast(JsonObject, safe_arguments_value)
        execution_payload: dict[str, JsonValue] = {
            "arguments": safe_arguments,
            "output": safe_output,
            "redaction_count": argument_redactions + output_redactions,
        }
        if execution_error is not None:
            execution_payload["error_type"] = type(execution_error).__name__
        self._recorder.record(
            EventType.TOOL_EXECUTED,
            request_id=request_id,
            tool=tool,
            payload=execution_payload,
        )
        if before != after:
            self._recorder.record(
                EventType.STATE_CHANGED,
                request_id=request_id,
                tool=tool,
                payload={"before": before, "after": after},
            )

        self._usage[tool] = self._usage.get(tool, 0) + 1
        if execution_error is not None:
            failure_reasons = (f"execution_failed:{type(execution_error).__name__}",)
            self._record_outcome(request_id, tool, "failed", failure_reasons)
            return ToolCallResult(
                request_id=request_id,
                status="failed",
                reasons=failure_reasons,
            )

        for source_ref, source_kind in observed_sources:
            self._observed_sources[source_ref] = source_kind
        self._record_outcome(request_id, tool, "executed", ())
        return ToolCallResult(
            request_id=request_id,
            status="executed",
            output=execution_output,
        )

    def _record_requested(
        self,
        request_id: str,
        tool: ToolId,
        arguments: JsonObject,
    ) -> None:
        safe_arguments_value, redaction_count = _redact_json(arguments)
        safe_arguments = cast(JsonObject, safe_arguments_value)
        source_refs: list[JsonValue] = [source for source in sorted(self._observed_sources)]
        source_kinds: list[JsonValue] = [
            source for source in sorted(set(self._observed_sources.values()))
        ]
        payload: dict[str, JsonValue] = {
            "arguments": safe_arguments,
            "source_refs": source_refs,
            "source_kinds": source_kinds,
            "redaction_count": redaction_count,
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
        tool: ToolId,
        effect: DecisionEffect,
        policy_allowed: bool,
        reasons: tuple[str, ...],
        requires_approval: bool = False,
        matched_rule_ids: tuple[str, ...] = (),
        untrusted_source_kinds: tuple[SourceId, ...] = (),
        approval_verified: bool = False,
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
                "untrusted_source_kinds": list(untrusted_source_kinds),
                "approval_verified": approval_verified,
            },
        )

    def _record_outcome(
        self,
        request_id: str,
        tool: ToolId,
        status: str,
        reasons: tuple[str, ...],
    ) -> None:
        self._recorder.record(
            EventType.EXECUTION_OUTCOME,
            request_id=request_id,
            tool=tool,
            payload={"status": status, "reasons": list(reasons)},
        )


class BoundToolInvoker(ToolInvoker):
    def __init__(
        self,
        *,
        caller_id: str,
        signed_contract: SignedTaskContract,
        gateway: GenericPolicyGateway,
    ) -> None:
        self._caller_id = caller_id
        self._signed_contract = signed_contract
        self._gateway = gateway

    def call(
        self,
        tool: ToolId,
        arguments: JsonObject,
        *,
        approval: SignedApproval | None = None,
    ) -> ToolCallResult:
        return self._gateway.call(
            caller_id=self._caller_id,
            signed_contract=self._signed_contract,
            tool=tool,
            arguments=arguments,
            approval=approval,
        )
