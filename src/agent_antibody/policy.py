from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

from agent_antibody.contracts import (
    JsonPrimitive,
    JsonValue,
    SignedApproval,
    SourceKind,
    TaskContract,
    ToolName,
)
from agent_antibody.signing import ApprovalSigner

_LESS_THAN = re.compile(r"^<\s*(\d+)$")


class PolicyAction(StrEnum):
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


class PredicateOperator(StrEnum):
    EQ = "eq"
    LT = "lt"
    LTE = "lte"
    GT = "gt"
    GTE = "gte"
    CONTAINS = "contains"


class ArgumentPredicate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    argument: str
    operator: PredicateOperator
    value: JsonPrimitive

    def matches(self, arguments: dict[str, JsonValue]) -> bool:
        candidate = arguments.get(self.argument)
        if self.operator == PredicateOperator.EQ:
            return candidate == self.value
        if self.operator == PredicateOperator.CONTAINS:
            return (
                isinstance(candidate, str)
                and isinstance(self.value, str)
                and self.value in candidate
            )
        if (
            not isinstance(candidate, (int, float))
            or isinstance(candidate, bool)
            or not isinstance(self.value, (int, float))
            or isinstance(self.value, bool)
        ):
            return False
        if self.operator == PredicateOperator.LT:
            return candidate < self.value
        if self.operator == PredicateOperator.LTE:
            return candidate <= self.value
        if self.operator == PredicateOperator.GT:
            return candidate > self.value
        if self.operator == PredicateOperator.GTE:
            return candidate >= self.value
        return False


class RuleWhen(BaseModel):
    """Small, deliberately non-evaluated predicate language for MVP policy rules."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    service: str | None = None
    replicas: str | None = None
    predicates: tuple[ArgumentPredicate, ...] = ()

    @field_validator("replicas")
    @classmethod
    def validate_replicas_expression(cls, value: str | None) -> str | None:
        if value is not None and _LESS_THAN.fullmatch(value) is None:
            raise ValueError("replicas supports only a '< integer' predicate")
        return value

    def matches(self, arguments: dict[str, JsonValue]) -> bool:
        if self.service is not None and arguments.get("service") != self.service:
            return False
        if self.replicas is not None:
            match = _LESS_THAN.fullmatch(self.replicas)
            candidate = arguments.get("replicas")
            if match is None or not isinstance(candidate, int) or isinstance(candidate, bool):
                return False
            if candidate >= int(match.group(1)):
                return False
        return all(predicate.matches(arguments) for predicate in self.predicates)


class ToolPolicyRule(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    rule_id: str
    tool: str
    when: RuleWhen = RuleWhen()
    action: PolicyAction


class PolicyRules(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1] = 1
    default_action: Literal["allow"] = "allow"
    rules: tuple[ToolPolicyRule, ...] = ()
    untrusted_sources: tuple[SourceKind, ...] = (
        SourceKind.RUNBOOK_CONTENT,
        SourceKind.ISSUE_BODY,
        SourceKind.LOG_CONTENT,
    )

    @classmethod
    def from_yaml(cls, content: str) -> PolicyRules:
        data = yaml.safe_load(content)
        if not isinstance(data, dict):
            raise ValueError("policy YAML must contain an object")
        return cls.model_validate(cast(dict[str, object], data))

    def to_yaml(self) -> str:
        return yaml.safe_dump(
            self.model_dump(mode="json", exclude_none=True),
            sort_keys=False,
            allow_unicode=True,
        )


class PolicyEvaluation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    allowed: bool
    reasons: tuple[str, ...] = ()
    requires_approval: bool = False
    matched_rule_ids: tuple[str, ...] = ()
    untrusted_source_kinds: tuple[SourceKind, ...] = ()


class PolicyEngine:
    def __init__(self, rules: PolicyRules, approval_signer: ApprovalSigner) -> None:
        self._rules = rules
        self._approval_signer = approval_signer

    def evaluate(
        self,
        *,
        contract: TaskContract,
        tool: ToolName,
        arguments: dict[str, JsonValue],
        usage: dict[ToolName, int],
        approval: SignedApproval | None,
        observed_source_kinds: tuple[SourceKind, ...] = (),
        now: datetime | None = None,
    ) -> PolicyEvaluation:
        reasons: list[str] = []
        matched_rule_ids: list[str] = []
        current_time = (now or datetime.now(UTC)).astimezone(UTC)

        if contract.issued_at > current_time:
            reasons.append("task_contract_not_yet_valid")
        if contract.expires_at <= current_time:
            reasons.append("task_contract_expired")
        if tool not in contract.allowed_tools:
            reasons.append("tool_not_allowed_by_task")

        service = arguments.get("service")
        if isinstance(service, str) and service not in contract.resources:
            reasons.append("resource_not_allowed_by_task")

        if tool == ToolName.POST_ISSUE:
            constraint = contract.tool_constraints.post_issue
            if constraint is None:
                reasons.append("post_issue_constraint_missing")
            else:
                if arguments.get("repository") != constraint.repository:
                    reasons.append("issue_repository_not_allowed")
                if arguments.get("issue") != constraint.issue:
                    reasons.append("issue_target_not_allowed")
                if usage.get(tool, 0) >= constraint.max_calls:
                    reasons.append("issue_call_limit_exceeded")

        matching_rules = [
            rule for rule in self._rules.rules if rule.tool == tool and rule.when.matches(arguments)
        ]
        matched_rule_ids.extend(rule.rule_id for rule in matching_rules)
        if any(rule.action == PolicyAction.DENY for rule in matching_rules):
            reasons.append("denied_by_policy_rule")

        requires_approval = any(
            rule.action == PolicyAction.REQUIRE_APPROVAL for rule in matching_rules
        )
        if requires_approval:
            if approval is None:
                reasons.append("approval_required")
            elif not self._approval_signer.verify(
                approval,
                contract=contract,
                tool=tool,
                arguments=arguments,
                now=current_time,
            ):
                reasons.append("approval_invalid")

        configured_untrusted = set(self._rules.untrusted_sources)
        observed_untrusted = tuple(
            sorted(
                {kind for kind in observed_source_kinds if kind in configured_untrusted},
                key=str,
            )
        )
        return PolicyEvaluation(
            allowed=not reasons,
            reasons=tuple(reasons),
            requires_approval=requires_approval,
            matched_rule_ids=tuple(matched_rule_ids),
            untrusted_source_kinds=observed_untrusted,
        )
