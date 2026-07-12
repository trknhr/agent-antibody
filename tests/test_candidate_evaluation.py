from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from agent_antibody.candidate_evaluation import (
    AttackCohort,
    AttackDisposition,
    CandidateAttackEvidence,
    CandidateEvaluationStatus,
    CandidateEvaluationV2,
    CandidateNormalTaskEvidence,
    CandidateTargetEvidence,
    CandidateToolRequestEvidence,
    RemediationArtifactReference,
    TargetCandidateEvaluation,
    aggregate_candidate_status,
    merge_candidate_evaluations,
)

BASE_REVISION = "a" * 40
SOURCE_REVISION = "b" * 40
EVALUATED_AT = datetime(2026, 7, 12, tzinfo=UTC)


def _blocked_attack(
    plan_id: str,
    *,
    cohort: AttackCohort = AttackCohort.CANDIDATE,
) -> CandidateAttackEvidence:
    request_id = f"req-{plan_id}"
    return CandidateAttackEvidence(
        plan_id=plan_id,
        cohort=cohort,
        disposition=AttackDisposition.POLICY_BLOCKED,
        expected_tool="grant_store_credit",
        requests=(
            CandidateToolRequestEvidence(
                request_id=request_id,
                tool="grant_store_credit",
                policy_blocked=True,
            ),
        ),
    )


def _bypass_attack(
    plan_id: str = "credit-seed",
    *,
    cohort: AttackCohort = AttackCohort.CANDIDATE,
) -> CandidateAttackEvidence:
    request_id = f"req-{plan_id}"
    return CandidateAttackEvidence(
        plan_id=plan_id,
        cohort=cohort,
        disposition=AttackDisposition.BYPASS_CONFIRMED,
        expected_tool="grant_store_credit",
        requests=(
            CandidateToolRequestEvidence(
                request_id=request_id,
                tool="grant_store_credit",
                executed=True,
                changed_state=True,
                unsafe_state_change=True,
            ),
        ),
        finding_codes=("unsafe_store_credit_granted",),
    )


def _healthy_normal(case_id: str = "normal-store-credit") -> CandidateNormalTaskEvidence:
    return CandidateNormalTaskEvidence(
        case_id=case_id,
        healthy=True,
        observed_tools=("grant_store_credit",),
    )


def _evidence(
    *,
    attacks: tuple[CandidateAttackEvidence, ...] | None = None,
    normal_tasks: tuple[CandidateNormalTaskEvidence, ...] | None = None,
    change_ids: tuple[str, ...] = ("added-tool:grant_store_credit",),
) -> CandidateTargetEvidence:
    return CandidateTargetEvidence(
        existing_policy_artifact_ids=("imm-supportmate-refund",),
        attack_surface_change_ids=change_ids,
        attacks=attacks
        if attacks is not None
        else (
            _blocked_attack("refund-memory", cohort=AttackCohort.HISTORICAL),
            _blocked_attack("credit-generated"),
        ),
        normal_tasks=normal_tasks if normal_tasks is not None else (_healthy_normal(),),
    )


def _remediation() -> RemediationArtifactReference:
    return RemediationArtifactReference(
        artifact_id="imm-supportmate-credit",
        target_id="supportmate",
        report_sha256="c" * 64,
        policy_rule_ids=("support-credit-approval",),
        regression_plan_ids=("credit-seed", "credit-holdout-1"),
    )


def _revalidated(target_id: str = "supportmate") -> TargetCandidateEvaluation:
    return TargetCandidateEvaluation.revalidated(
        target_id=target_id,
        current_memory_count=1,
        evidence=_evidence(),
    )


def _bypass(target_id: str = "supportmate") -> TargetCandidateEvaluation:
    return TargetCandidateEvaluation.bypass_confirmed(
        target_id=target_id,
        current_memory_count=1,
        evidence=_evidence(
            attacks=(
                _blocked_attack("refund-memory", cohort=AttackCohort.HISTORICAL),
                _bypass_attack(),
            )
        ),
        remediation_artifact=_remediation().model_copy(update={"target_id": target_id}),
    )


def _inconclusive(target_id: str = "supportmate") -> TargetCandidateEvaluation:
    return TargetCandidateEvaluation.inconclusive(
        target_id=target_id,
        current_memory_count=1,
        evidence=_evidence(attacks=(), normal_tasks=(), change_ids=()),
        reasons=("ADK capability extraction timed out",),
    )


def test_revalidated_preserves_memory_without_remediation() -> None:
    result = _revalidated()

    assert result.status == CandidateEvaluationStatus.REVALIDATED
    assert result.current_memory_count == 1
    assert result.proposed_memory_count == 1
    assert result.remediation_artifact is None
    assert len(result.evidence.historical_attacks) == 1
    assert len(result.evidence.candidate_attacks) == 1


@pytest.mark.parametrize(
    ("proposed_count", "remediation", "match"),
    (
        (2, None, "cannot change the memory count"),
        (1, _remediation(), "cannot contain a remediation artifact"),
    ),
)
def test_revalidated_rejects_a_memory_proposal(
    proposed_count: int,
    remediation: RemediationArtifactReference | None,
    match: str,
) -> None:
    with pytest.raises(ValidationError, match=match):
        TargetCandidateEvaluation(
            target_id="supportmate",
            status=CandidateEvaluationStatus.REVALIDATED,
            current_memory_count=1,
            proposed_memory_count=proposed_count,
            evidence=_evidence(),
            remediation_artifact=remediation,
        )


def test_revalidated_requires_complete_security_and_utility_evidence() -> None:
    inconclusive_attack = CandidateAttackEvidence(
        plan_id="credit-timeout",
        cohort=AttackCohort.CANDIDATE,
        disposition=AttackDisposition.INCONCLUSIVE,
        expected_tool="grant_store_credit",
        reason="sandbox timed out",
    )
    with pytest.raises(ValidationError, match="safe outcome"):
        TargetCandidateEvaluation(
            target_id="supportmate",
            status=CandidateEvaluationStatus.REVALIDATED,
            current_memory_count=1,
            proposed_memory_count=1,
            evidence=_evidence(attacks=(inconclusive_attack,)),
        )

    unhealthy = CandidateNormalTaskEvidence(
        case_id="normal-store-credit",
        healthy=False,
        reason="expected credit was not granted",
    )
    with pytest.raises(ValidationError, match="healthy normal-task"):
        TargetCandidateEvaluation(
            target_id="supportmate",
            status=CandidateEvaluationStatus.REVALIDATED,
            current_memory_count=1,
            proposed_memory_count=1,
            evidence=_evidence(normal_tasks=(unhealthy,)),
        )


def test_bypass_requires_trace_bound_state_change_and_proposes_one_memory() -> None:
    result = _bypass()

    assert result.status == CandidateEvaluationStatus.BYPASS_CONFIRMED
    assert result.current_memory_count == 1
    assert result.proposed_memory_count == 2
    assert result.remediation_artifact == _remediation()
    assert [attack.plan_id for attack in result.evidence.candidate_bypasses] == ["credit-seed"]


def test_request_evidence_rejects_unexecuted_or_unconfirmed_state_changes() -> None:
    with pytest.raises(ValidationError, match="must have been executed"):
        CandidateToolRequestEvidence(
            request_id="req-credit",
            tool="grant_store_credit",
            changed_state=True,
        )

    with pytest.raises(ValidationError, match="must be an observed state change"):
        CandidateToolRequestEvidence(
            request_id="req-credit",
            tool="grant_store_credit",
            executed=True,
            unsafe_state_change=True,
        )


def test_bypass_evidence_requires_expected_tool_and_oracle_finding() -> None:
    with pytest.raises(ValidationError, match="deterministic oracle finding"):
        CandidateAttackEvidence(
            plan_id="credit-seed",
            cohort=AttackCohort.CANDIDATE,
            disposition=AttackDisposition.BYPASS_CONFIRMED,
            expected_tool="grant_store_credit",
            requests=(
                CandidateToolRequestEvidence(
                    request_id="req-credit",
                    tool="grant_store_credit",
                    executed=True,
                    changed_state=True,
                    unsafe_state_change=True,
                ),
            ),
        )

    with pytest.raises(ValidationError, match="expected tool"):
        CandidateAttackEvidence(
            plan_id="credit-seed",
            cohort=AttackCohort.CANDIDATE,
            disposition=AttackDisposition.BYPASS_CONFIRMED,
            expected_tool="grant_store_credit",
            requests=(
                CandidateToolRequestEvidence(
                    request_id="req-refund",
                    tool="issue_refund",
                    executed=True,
                    changed_state=True,
                    unsafe_state_change=True,
                ),
            ),
            finding_codes=("unsafe_refund_issued",),
        )


def test_bypass_rejects_historical_regression_and_unbound_remediation() -> None:
    historical_bypass = _bypass_attack("refund-memory", cohort=AttackCohort.HISTORICAL)
    with pytest.raises(ValidationError, match="historical-memory bypasses"):
        TargetCandidateEvaluation(
            target_id="supportmate",
            status=CandidateEvaluationStatus.BYPASS_CONFIRMED,
            current_memory_count=1,
            proposed_memory_count=2,
            evidence=_evidence(attacks=(historical_bypass, _bypass_attack())),
            remediation_artifact=_remediation(),
        )

    unrelated = _remediation().model_copy(update={"regression_plan_ids": ("another-plan",)})
    with pytest.raises(ValidationError, match="must preserve a confirmed bypass"):
        TargetCandidateEvaluation(
            target_id="supportmate",
            status=CandidateEvaluationStatus.BYPASS_CONFIRMED,
            current_memory_count=1,
            proposed_memory_count=2,
            evidence=_evidence(attacks=(_bypass_attack(),)),
            remediation_artifact=unrelated,
        )


def test_inconclusive_is_fail_closed_and_requires_reasons() -> None:
    result = _inconclusive()

    assert result.status == CandidateEvaluationStatus.INCONCLUSIVE
    assert result.proposed_memory_count == result.current_memory_count
    assert result.remediation_artifact is None
    assert result.inconclusive_reasons

    with pytest.raises(ValidationError, match="requires at least one reason"):
        TargetCandidateEvaluation(
            target_id="supportmate",
            status=CandidateEvaluationStatus.INCONCLUSIVE,
            current_memory_count=1,
            proposed_memory_count=1,
            evidence=_evidence(attacks=(), normal_tasks=(), change_ids=()),
        )

    with pytest.raises(ValidationError, match="cannot change the memory count"):
        TargetCandidateEvaluation(
            target_id="supportmate",
            status=CandidateEvaluationStatus.INCONCLUSIVE,
            current_memory_count=1,
            proposed_memory_count=2,
            evidence=_evidence(attacks=(), normal_tasks=(), change_ids=()),
            inconclusive_reasons=("attack generation failed",),
        )


def test_current_count_is_bound_to_existing_artifact_evidence() -> None:
    with pytest.raises(ValidationError, match="must match existing policy artifacts"):
        TargetCandidateEvaluation(
            target_id="supportmate",
            status=CandidateEvaluationStatus.INCONCLUSIVE,
            current_memory_count=2,
            proposed_memory_count=2,
            evidence=_evidence(attacks=(), normal_tasks=(), change_ids=()),
            inconclusive_reasons=("evaluation interrupted",),
        )


def test_aggregate_status_precedence_is_fail_closed() -> None:
    revalidated = _revalidated("supportmate")
    bypass = _bypass("repomate")
    inconclusive = _inconclusive("opsmate")

    assert aggregate_candidate_status((revalidated,)) == CandidateEvaluationStatus.REVALIDATED
    assert (
        aggregate_candidate_status((revalidated, bypass))
        == CandidateEvaluationStatus.BYPASS_CONFIRMED
    )
    assert (
        aggregate_candidate_status((revalidated, bypass, inconclusive))
        == CandidateEvaluationStatus.INCONCLUSIVE
    )


def test_aggregate_exposes_remediation_only_for_atomic_bypass() -> None:
    bypass_evaluation = CandidateEvaluationV2.from_targets(
        base_revision=BASE_REVISION,
        source_revision=SOURCE_REVISION,
        evaluated_at=EVALUATED_AT,
        targets=(_revalidated("supportmate"), _bypass("repomate")),
    )

    assert bypass_evaluation.requires_remediation
    assert bypass_evaluation.current_memory_count == 2
    assert bypass_evaluation.proposed_memory_count == 3
    assert len(bypass_evaluation.remediation_artifacts) == 1

    fail_closed = CandidateEvaluationV2.from_targets(
        base_revision=BASE_REVISION,
        source_revision=SOURCE_REVISION,
        evaluated_at=EVALUATED_AT,
        targets=(_bypass("repomate"), _inconclusive("opsmate")),
    )

    assert fail_closed.status == CandidateEvaluationStatus.INCONCLUSIVE
    assert not fail_closed.requires_remediation
    assert fail_closed.proposed_memory_count == fail_closed.current_memory_count
    assert fail_closed.remediation_artifacts == ()


def test_aggregate_rejects_a_forged_status_and_duplicate_targets() -> None:
    with pytest.raises(ValidationError, match="aggregate status must be BYPASS_CONFIRMED"):
        CandidateEvaluationV2(
            base_revision=BASE_REVISION,
            source_revision=SOURCE_REVISION,
            evaluated_at=EVALUATED_AT,
            status=CandidateEvaluationStatus.REVALIDATED,
            targets=(_bypass(),),
        )

    with pytest.raises(ValidationError, match="target IDs"):
        CandidateEvaluationV2.from_targets(
            base_revision=BASE_REVISION,
            source_revision=SOURCE_REVISION,
            evaluated_at=EVALUATED_AT,
            targets=(_revalidated(), _revalidated()),
        )


def test_aggregate_requires_at_least_one_target() -> None:
    with pytest.raises(ValueError, match="at least one target"):
        aggregate_candidate_status(())


def test_merge_replaces_only_an_inconclusive_target_and_preserves_full_coverage() -> None:
    incomplete_supportmate = TargetCandidateEvaluation.inconclusive(
        target_id="supportmate",
        current_memory_count=1,
        evidence=_evidence(attacks=(), normal_tasks=()),
        reasons=("delta-specific attack campaign required",),
    )
    base = CandidateEvaluationV2.from_targets(
        base_revision=BASE_REVISION,
        source_revision=SOURCE_REVISION,
        evaluated_at=EVALUATED_AT,
        targets=(
            _revalidated("opsmate"),
            _revalidated("repomate"),
            incomplete_supportmate,
        ),
    )
    replacement = CandidateEvaluationV2.from_targets(
        base_revision=BASE_REVISION,
        source_revision=SOURCE_REVISION,
        evaluated_at=EVALUATED_AT,
        targets=(_bypass("supportmate"),),
    )

    merged = merge_candidate_evaluations(base=base, replacement=replacement)

    assert merged.status == CandidateEvaluationStatus.BYPASS_CONFIRMED
    assert [target.status for target in merged.targets] == [
        CandidateEvaluationStatus.REVALIDATED,
        CandidateEvaluationStatus.REVALIDATED,
        CandidateEvaluationStatus.BYPASS_CONFIRMED,
    ]
    assert merged.remediation_artifacts == (_remediation(),)


def test_merge_rejects_replacing_a_revalidated_target() -> None:
    base = CandidateEvaluationV2.from_targets(
        base_revision=BASE_REVISION,
        source_revision=SOURCE_REVISION,
        evaluated_at=EVALUATED_AT,
        targets=(_revalidated(),),
    )
    replacement = CandidateEvaluationV2.from_targets(
        base_revision=BASE_REVISION,
        source_revision=SOURCE_REVISION,
        evaluated_at=EVALUATED_AT,
        targets=(_bypass(),),
    )

    with pytest.raises(ValueError, match="only an inconclusive target"):
        merge_candidate_evaluations(base=base, replacement=replacement)
