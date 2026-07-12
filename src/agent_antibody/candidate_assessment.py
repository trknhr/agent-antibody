from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from agent_antibody.ai_models import AttackPlan
from agent_antibody.candidate_evaluation import (
    AttackDisposition,
    CandidateEvaluationV2,
    CandidateTargetEvidence,
    TargetCandidateEvaluation,
)
from agent_antibody.candidate_runner import collect_candidate_evidence
from agent_antibody.capabilities import AttackSurfaceChangeKind, AttackSurfaceDelta
from agent_antibody.immunity_artifacts import load_artifacts
from agent_antibody.targets.registry import get_target_adapter


def _change_id(delta: AttackSurfaceDelta, index: int) -> str:
    change = delta.changes[index]
    subject = change.tool_name or "agent-instruction"
    return f"{change.kind.value}:{subject}"


def _revalidation_attacks(
    *,
    source_revision: str,
    persisted_attacks: tuple[AttackPlan, ...],
) -> tuple[AttackPlan, ...]:
    prefix = source_revision[:12]
    return tuple(
        attack.model_copy(update={"plan_id": f"candidate-{prefix}-{attack.plan_id}"})
        for attack in persisted_attacks
    )


def assess_target_revalidation(
    *,
    memory_repository_root: Path,
    target_id: str,
    source_revision: str,
    capability_delta: AttackSurfaceDelta,
) -> TargetCandidateEvaluation:
    """Revalidate a candidate when existing campaigns cover every changed ADK tool.

    A genuinely new tool needs a delta-specific attack campaign and remediation builder.
    Until that evidence exists this function returns ``INCONCLUSIVE`` instead of
    silently treating an old campaign as coverage for the new surface.
    """

    adapter = get_target_adapter(target_id)
    artifacts = load_artifacts(repository_root=memory_repository_root, target_id=target_id)
    current_count = len(artifacts)
    base_evidence = CandidateTargetEvidence(
        existing_policy_artifact_ids=tuple(artifact.artifact_id for artifact in artifacts),
        attack_surface_change_ids=tuple(
            _change_id(capability_delta, index) for index in range(len(capability_delta.changes))
        ),
    )
    if not artifacts:
        return TargetCandidateEvaluation.inconclusive(
            target_id=target_id,
            current_memory_count=0,
            evidence=base_evidence,
            reasons=("target has no existing immunity memory",),
        )

    instruction_changed = any(
        change.kind == AttackSurfaceChangeKind.AGENT_INSTRUCTION_CHANGED
        for change in capability_delta.changes
    )
    if instruction_changed and not capability_delta.attack_required_tools:
        return TargetCandidateEvaluation.inconclusive(
            target_id=target_id,
            current_memory_count=current_count,
            evidence=base_evidence,
            reasons=(
                "agent instruction changed without an observed tool delta; "
                "an expanded attack campaign is required",
            ),
        )

    persisted_plans = tuple(
        persisted.to_plan() for artifact in artifacts for persisted in artifact.regression.attacks
    )
    persisted_tools = {plan.expected_tool for plan in persisted_plans}
    uncovered_tools = set(capability_delta.attack_required_tools).difference(persisted_tools)
    if uncovered_tools:
        return TargetCandidateEvaluation.inconclusive(
            target_id=target_id,
            current_memory_count=current_count,
            evidence=base_evidence,
            reasons=(
                "delta-specific attack campaign required for tools: "
                + ", ".join(sorted(uncovered_tools)),
            ),
        )

    candidate_attacks = _revalidation_attacks(
        source_revision=source_revision,
        persisted_attacks=persisted_plans,
    )
    try:
        evidence = collect_candidate_evidence(
            repository_root=memory_repository_root,
            adapter=adapter,
            candidate_attacks=candidate_attacks,
            attack_surface_change_ids=base_evidence.attack_surface_change_ids,
        )
    except (TypeError, ValueError) as error:
        return TargetCandidateEvaluation.inconclusive(
            target_id=target_id,
            current_memory_count=current_count,
            evidence=base_evidence,
            reasons=(f"candidate replay failed: {type(error).__name__}",),
        )

    reasons: list[str] = []
    if evidence.incomplete_attacks:
        reasons.append("one or more attacks produced inconclusive traces")
    if any(
        attack.disposition == AttackDisposition.BYPASS_CONFIRMED
        for attack in evidence.historical_attacks
    ):
        reasons.append("candidate broke an existing immunity regression")
    if evidence.candidate_bypasses:
        reasons.append("candidate bypass requires a verified delta remediation")
    if evidence.unhealthy_normal_tasks:
        reasons.append("candidate failed one or more normal utility tasks")
    if reasons:
        return TargetCandidateEvaluation.inconclusive(
            target_id=target_id,
            current_memory_count=current_count,
            evidence=evidence,
            reasons=tuple(reasons),
        )
    return TargetCandidateEvaluation.revalidated(
        target_id=target_id,
        current_memory_count=current_count,
        evidence=evidence,
    )


def assess_repository_revalidation(
    *,
    memory_repository_root: Path,
    base_revision: str,
    source_revision: str,
    capability_deltas: dict[str, AttackSurfaceDelta],
) -> CandidateEvaluationV2:
    if not capability_deltas:
        raise ValueError("candidate assessment requires at least one target delta")
    targets = tuple(
        assess_target_revalidation(
            memory_repository_root=memory_repository_root,
            target_id=target_id,
            source_revision=source_revision,
            capability_delta=delta,
        )
        for target_id, delta in sorted(capability_deltas.items())
    )
    return CandidateEvaluationV2.from_targets(
        base_revision=base_revision,
        source_revision=source_revision,
        evaluated_at=datetime.now(UTC),
        targets=targets,
    )
