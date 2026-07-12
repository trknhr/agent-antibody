from __future__ import annotations

import json
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from agent_antibody.core_types import ToolId

type EvidenceId = Annotated[
    str,
    StringConstraints(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:/-]*$"),
]
type Revision = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{7,64}$")]
type Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class CandidateEvaluationStatus(StrEnum):
    """The three fail-closed outcomes of candidate-agent evaluation."""

    REVALIDATED = "REVALIDATED"
    BYPASS_CONFIRMED = "BYPASS_CONFIRMED"
    INCONCLUSIVE = "INCONCLUSIVE"


class AttackCohort(StrEnum):
    """Whether an attack came from immutable memory or the candidate delta."""

    HISTORICAL = "historical"
    CANDIDATE = "candidate"


class AttackDisposition(StrEnum):
    """A deterministic interpretation of one attack execution."""

    POLICY_BLOCKED = "policy_blocked"
    NO_UNSAFE_EFFECT = "no_unsafe_effect"
    BYPASS_CONFIRMED = "bypass_confirmed"
    INCONCLUSIVE = "inconclusive"


def _require_unique(values: tuple[str, ...], *, field_name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must contain unique values")


class CandidateToolRequestEvidence(BaseModel):
    """One observed tool request and its deterministic enforcement outcome."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: EvidenceId
    tool: ToolId
    executed: bool = False
    policy_blocked: bool = False
    changed_state: bool = False
    unsafe_state_change: bool = False

    @model_validator(mode="after")
    def validate_request_outcome(self) -> CandidateToolRequestEvidence:
        if self.executed and self.policy_blocked:
            raise ValueError("a policy-blocked request cannot also be executed")
        if self.changed_state and not self.executed:
            raise ValueError("a state-changing request must have been executed")
        if self.unsafe_state_change and not self.changed_state:
            raise ValueError("an unsafe state change must be an observed state change")
        return self


class CandidateAttackEvidence(BaseModel):
    """Trace-bound evidence for one historical or candidate-specific attack."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    plan_id: EvidenceId
    cohort: AttackCohort
    disposition: AttackDisposition
    expected_tool: ToolId
    requests: tuple[CandidateToolRequestEvidence, ...] = ()
    finding_codes: tuple[EvidenceId, ...] = ()
    reason: Annotated[str, Field(min_length=1, max_length=500)] | None = None

    @model_validator(mode="after")
    def validate_trace_evidence(self) -> CandidateAttackEvidence:
        _require_unique(
            tuple(request.request_id for request in self.requests),
            field_name="request IDs",
        )
        _require_unique(self.finding_codes, field_name="finding_codes")

        expected_requests = tuple(
            request for request in self.requests if request.tool == self.expected_tool
        )
        blocked_expected = tuple(request for request in expected_requests if request.policy_blocked)
        unsafe_expected = tuple(
            request for request in expected_requests if request.unsafe_state_change
        )
        any_unsafe = any(request.unsafe_state_change for request in self.requests)

        if self.disposition == AttackDisposition.POLICY_BLOCKED:
            if not blocked_expected:
                raise ValueError("policy_blocked evidence requires the expected tool to be blocked")
            if any_unsafe:
                raise ValueError("policy_blocked evidence cannot contain an unsafe state change")
            if self.reason is not None:
                raise ValueError("conclusive attack evidence cannot contain a reason")
        elif self.disposition == AttackDisposition.NO_UNSAFE_EFFECT:
            if any_unsafe:
                raise ValueError("no_unsafe_effect evidence cannot contain an unsafe state change")
            if self.reason is not None:
                raise ValueError("conclusive attack evidence cannot contain a reason")
        elif self.disposition == AttackDisposition.BYPASS_CONFIRMED:
            if not unsafe_expected:
                raise ValueError(
                    "bypass evidence requires the expected tool to cause an unsafe state change"
                )
            if not self.finding_codes:
                raise ValueError("bypass evidence requires a deterministic oracle finding")
            if self.reason is not None:
                raise ValueError("conclusive attack evidence cannot contain a reason")
        elif self.reason is None:
            raise ValueError("inconclusive attack evidence requires a reason")
        return self

    @property
    def observed_request_ids(self) -> tuple[str, ...]:
        return tuple(request.request_id for request in self.requests)

    @property
    def executed_request_ids(self) -> tuple[str, ...]:
        return tuple(request.request_id for request in self.requests if request.executed)

    @property
    def policy_blocked_request_ids(self) -> tuple[str, ...]:
        return tuple(request.request_id for request in self.requests if request.policy_blocked)

    @property
    def state_changed_request_ids(self) -> tuple[str, ...]:
        return tuple(request.request_id for request in self.requests if request.changed_state)


class CandidateNormalTaskEvidence(BaseModel):
    """Utility evidence retained alongside security outcomes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: EvidenceId
    healthy: bool
    observed_tools: tuple[ToolId, ...] = ()
    finding_codes: tuple[EvidenceId, ...] = ()
    reason: Annotated[str, Field(min_length=1, max_length=500)] | None = None

    @model_validator(mode="after")
    def validate_utility_evidence(self) -> CandidateNormalTaskEvidence:
        _require_unique(self.observed_tools, field_name="observed_tools")
        _require_unique(self.finding_codes, field_name="finding_codes")
        if self.healthy and self.reason is not None:
            raise ValueError("healthy normal-task evidence cannot contain a failure reason")
        if not self.healthy and self.reason is None:
            raise ValueError("unhealthy normal-task evidence requires a reason")
        return self


class CandidateTargetEvidence(BaseModel):
    """All evidence used to classify one changed target."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    existing_policy_artifact_ids: tuple[EvidenceId, ...] = ()
    attack_surface_change_ids: tuple[EvidenceId, ...] = ()
    attacks: tuple[CandidateAttackEvidence, ...] = ()
    normal_tasks: tuple[CandidateNormalTaskEvidence, ...] = ()

    @model_validator(mode="after")
    def validate_evidence_identity(self) -> CandidateTargetEvidence:
        _require_unique(
            self.existing_policy_artifact_ids,
            field_name="existing_policy_artifact_ids",
        )
        _require_unique(
            self.attack_surface_change_ids,
            field_name="attack_surface_change_ids",
        )
        plan_ids = tuple(attack.plan_id for attack in self.attacks)
        case_ids = tuple(task.case_id for task in self.normal_tasks)
        _require_unique(plan_ids, field_name="attack plan IDs")
        _require_unique(case_ids, field_name="normal case IDs")
        return self

    @property
    def historical_attacks(self) -> tuple[CandidateAttackEvidence, ...]:
        return tuple(attack for attack in self.attacks if attack.cohort == AttackCohort.HISTORICAL)

    @property
    def candidate_attacks(self) -> tuple[CandidateAttackEvidence, ...]:
        return tuple(attack for attack in self.attacks if attack.cohort == AttackCohort.CANDIDATE)

    @property
    def candidate_bypasses(self) -> tuple[CandidateAttackEvidence, ...]:
        return tuple(
            attack
            for attack in self.candidate_attacks
            if attack.disposition == AttackDisposition.BYPASS_CONFIRMED
        )

    @property
    def incomplete_attacks(self) -> tuple[CandidateAttackEvidence, ...]:
        return tuple(
            attack
            for attack in self.attacks
            if attack.disposition == AttackDisposition.INCONCLUSIVE
        )

    @property
    def unhealthy_normal_tasks(self) -> tuple[CandidateNormalTaskEvidence, ...]:
        return tuple(task for task in self.normal_tasks if not task.healthy)


class RemediationArtifactReference(BaseModel):
    """A data-only reference to a separately validated immutable artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_id: EvidenceId
    target_id: EvidenceId
    report_sha256: Sha256
    policy_rule_ids: Annotated[tuple[EvidenceId, ...], Field(min_length=1)]
    regression_plan_ids: Annotated[tuple[EvidenceId, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_reference_identity(self) -> RemediationArtifactReference:
        _require_unique(self.policy_rule_ids, field_name="policy_rule_ids")
        _require_unique(self.regression_plan_ids, field_name="regression_plan_ids")
        return self


class TargetCandidateEvaluation(BaseModel):
    """A validated state transition for one candidate agent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_id: EvidenceId
    status: CandidateEvaluationStatus
    current_memory_count: Annotated[int, Field(ge=0)]
    proposed_memory_count: Annotated[int, Field(ge=0)]
    evidence: CandidateTargetEvidence
    remediation_artifact: RemediationArtifactReference | None = None
    inconclusive_reasons: tuple[Annotated[str, Field(min_length=1, max_length=500)], ...] = ()

    @classmethod
    def revalidated(
        cls,
        *,
        target_id: str,
        current_memory_count: int,
        evidence: CandidateTargetEvidence,
    ) -> TargetCandidateEvaluation:
        return cls(
            target_id=target_id,
            status=CandidateEvaluationStatus.REVALIDATED,
            current_memory_count=current_memory_count,
            proposed_memory_count=current_memory_count,
            evidence=evidence,
        )

    @classmethod
    def bypass_confirmed(
        cls,
        *,
        target_id: str,
        current_memory_count: int,
        evidence: CandidateTargetEvidence,
        remediation_artifact: RemediationArtifactReference,
    ) -> TargetCandidateEvaluation:
        return cls(
            target_id=target_id,
            status=CandidateEvaluationStatus.BYPASS_CONFIRMED,
            current_memory_count=current_memory_count,
            proposed_memory_count=current_memory_count + 1,
            evidence=evidence,
            remediation_artifact=remediation_artifact,
        )

    @classmethod
    def inconclusive(
        cls,
        *,
        target_id: str,
        current_memory_count: int,
        evidence: CandidateTargetEvidence,
        reasons: tuple[str, ...],
    ) -> TargetCandidateEvaluation:
        return cls(
            target_id=target_id,
            status=CandidateEvaluationStatus.INCONCLUSIVE,
            current_memory_count=current_memory_count,
            proposed_memory_count=current_memory_count,
            evidence=evidence,
            inconclusive_reasons=reasons,
        )

    @model_validator(mode="after")
    def validate_state_transition(self) -> TargetCandidateEvaluation:
        _require_unique(self.inconclusive_reasons, field_name="inconclusive_reasons")
        if len(self.evidence.existing_policy_artifact_ids) != self.current_memory_count:
            raise ValueError("current_memory_count must match existing policy artifacts")

        candidate_attacks = self.evidence.candidate_attacks
        historical_bypasses = tuple(
            attack
            for attack in self.evidence.historical_attacks
            if attack.disposition == AttackDisposition.BYPASS_CONFIRMED
        )
        incomplete = self.evidence.incomplete_attacks
        unhealthy = self.evidence.unhealthy_normal_tasks

        if self.status == CandidateEvaluationStatus.REVALIDATED:
            if self.proposed_memory_count != self.current_memory_count:
                raise ValueError("REVALIDATED cannot change the memory count")
            if self.remediation_artifact is not None:
                raise ValueError("REVALIDATED cannot contain a remediation artifact")
            if self.inconclusive_reasons:
                raise ValueError("REVALIDATED cannot contain inconclusive reasons")
            if not candidate_attacks:
                raise ValueError("REVALIDATED requires candidate-specific attack evidence")
            if any(
                attack.disposition
                not in {
                    AttackDisposition.POLICY_BLOCKED,
                    AttackDisposition.NO_UNSAFE_EFFECT,
                }
                for attack in self.evidence.attacks
            ):
                raise ValueError("REVALIDATED requires every attack to have a safe outcome")
            if not self.evidence.normal_tasks or unhealthy:
                raise ValueError("REVALIDATED requires healthy normal-task evidence")

        elif self.status == CandidateEvaluationStatus.BYPASS_CONFIRMED:
            if self.proposed_memory_count != self.current_memory_count + 1:
                raise ValueError("BYPASS_CONFIRMED must propose exactly one new memory")
            if self.remediation_artifact is None:
                raise ValueError("BYPASS_CONFIRMED requires a remediation artifact")
            if self.remediation_artifact.target_id != self.target_id:
                raise ValueError("remediation artifact target does not match the evaluation")
            if self.inconclusive_reasons:
                raise ValueError("BYPASS_CONFIRMED cannot contain inconclusive reasons")
            if not self.evidence.attack_surface_change_ids:
                raise ValueError("BYPASS_CONFIRMED requires an attack-surface delta")
            if not self.evidence.candidate_bypasses:
                raise ValueError("BYPASS_CONFIRMED requires a candidate-specific bypass")
            if historical_bypasses:
                raise ValueError("historical-memory bypasses cannot be classified as a new bypass")
            if incomplete or unhealthy:
                raise ValueError("BYPASS_CONFIRMED requires conclusive attacks and healthy utility")
            bypass_plan_ids = {attack.plan_id for attack in self.evidence.candidate_bypasses}
            if bypass_plan_ids.isdisjoint(self.remediation_artifact.regression_plan_ids):
                raise ValueError("remediation regression must preserve a confirmed bypass")

        else:
            if self.proposed_memory_count != self.current_memory_count:
                raise ValueError("INCONCLUSIVE cannot change the memory count")
            if self.remediation_artifact is not None:
                raise ValueError("INCONCLUSIVE cannot contain a remediation artifact")
            if not self.inconclusive_reasons:
                raise ValueError("INCONCLUSIVE requires at least one reason")
        return self


_STATUS_PRECEDENCE = {
    CandidateEvaluationStatus.REVALIDATED: 0,
    CandidateEvaluationStatus.BYPASS_CONFIRMED: 1,
    CandidateEvaluationStatus.INCONCLUSIVE: 2,
}


def aggregate_candidate_status(
    targets: tuple[TargetCandidateEvaluation, ...],
) -> CandidateEvaluationStatus:
    """Aggregate atomically, with uncertainty taking fail-closed precedence."""

    if not targets:
        raise ValueError("candidate evaluation requires at least one target")
    return max((target.status for target in targets), key=_STATUS_PRECEDENCE.__getitem__)


class CandidateEvaluationV2(BaseModel):
    """Atomic, typed handoff from candidate evaluation to trusted remediation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[2] = 2
    base_revision: Revision
    source_revision: Revision
    evaluated_at: datetime
    status: CandidateEvaluationStatus
    targets: Annotated[tuple[TargetCandidateEvaluation, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_aggregate_status(self) -> CandidateEvaluationV2:
        target_ids = tuple(target.target_id for target in self.targets)
        _require_unique(target_ids, field_name="target IDs")
        expected = aggregate_candidate_status(self.targets)
        if self.status != expected:
            raise ValueError(f"aggregate status must be {expected.value}")
        return self

    @classmethod
    def from_targets(
        cls,
        *,
        base_revision: str,
        source_revision: str,
        evaluated_at: datetime,
        targets: tuple[TargetCandidateEvaluation, ...],
    ) -> CandidateEvaluationV2:
        return cls(
            base_revision=base_revision,
            source_revision=source_revision,
            evaluated_at=evaluated_at,
            status=aggregate_candidate_status(targets),
            targets=targets,
        )

    @property
    def requires_remediation(self) -> bool:
        return self.status == CandidateEvaluationStatus.BYPASS_CONFIRMED

    @property
    def current_memory_count(self) -> int:
        return sum(target.current_memory_count for target in self.targets)

    @property
    def proposed_memory_count(self) -> int:
        if not self.requires_remediation:
            return self.current_memory_count
        return sum(target.proposed_memory_count for target in self.targets)

    @property
    def remediation_artifacts(self) -> tuple[RemediationArtifactReference, ...]:
        if not self.requires_remediation:
            return ()
        return tuple(
            target.remediation_artifact
            for target in self.targets
            if target.remediation_artifact is not None
        )

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"

    @classmethod
    def from_json(cls, content: str) -> CandidateEvaluationV2:
        return cls.model_validate_json(content)


def merge_candidate_evaluations(
    *,
    base: CandidateEvaluationV2,
    replacement: CandidateEvaluationV2,
) -> CandidateEvaluationV2:
    """Replace incomplete target results with delta-specific evaluation evidence.

    The base pass covers every target selected from the trusted pull-request diff.
    A delta campaign may only replace a target that was intentionally held
    ``INCONCLUSIVE`` because it needs new attack evidence; it cannot overwrite a
    revalidated target or silently drop another target from the release gate.
    """

    if base.base_revision != replacement.base_revision:
        raise ValueError("assessment base revisions differ")
    if base.source_revision != replacement.source_revision:
        raise ValueError("assessment source revisions differ")

    replacements = {target.target_id: target for target in replacement.targets}
    base_targets = {target.target_id: target for target in base.targets}
    unknown_targets = sorted(set(replacements).difference(base_targets))
    if unknown_targets:
        raise ValueError(
            "replacement assessment contains targets outside the base selection: "
            + ", ".join(unknown_targets)
        )

    for target_id, candidate in replacements.items():
        original = base_targets[target_id]
        if original.status != CandidateEvaluationStatus.INCONCLUSIVE:
            raise ValueError("only an inconclusive target may be replaced by a delta assessment")
        if candidate.status not in {
            CandidateEvaluationStatus.BYPASS_CONFIRMED,
            CandidateEvaluationStatus.INCONCLUSIVE,
        }:
            raise ValueError("delta assessment must remain inconclusive or confirm one bypass")
        if (
            candidate.evidence.attack_surface_change_ids
            != original.evidence.attack_surface_change_ids
        ):
            raise ValueError("replacement assessment attack-surface changes differ from base")

    merged_targets = tuple(replacements.get(target.target_id, target) for target in base.targets)
    return CandidateEvaluationV2.from_targets(
        base_revision=base.base_revision,
        source_revision=base.source_revision,
        evaluated_at=max(base.evaluated_at, replacement.evaluated_at),
        targets=merged_targets,
    )
