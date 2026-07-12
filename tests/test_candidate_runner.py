from __future__ import annotations

from pathlib import Path

from agent_antibody.candidate_evaluation import (
    AttackDisposition,
    TargetCandidateEvaluation,
)
from agent_antibody.candidate_runner import collect_candidate_evidence
from agent_antibody.immunity_artifacts import load_artifacts
from agent_antibody.targets.registry import get_target_adapter


def test_existing_immunity_is_applied_before_candidate_attacks() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    adapter = get_target_adapter("supportmate")
    artifacts = load_artifacts(repository_root=repository_root, target_id="supportmate")
    assert len(artifacts) == 1
    candidate = (
        artifacts[0]
        .regression.attacks[0]
        .to_plan()
        .model_copy(update={"plan_id": "candidate-supportmate-revalidation"})
    )

    evidence = collect_candidate_evidence(
        repository_root=repository_root,
        adapter=adapter,
        candidate_attacks=(candidate,),
        attack_surface_change_ids=("AGENT_INSTRUCTION_CHANGED",),
    )
    result = TargetCandidateEvaluation.revalidated(
        target_id="supportmate",
        current_memory_count=1,
        evidence=evidence,
    )

    assert len(evidence.historical_attacks) == 10
    assert all(
        attack.disposition == AttackDisposition.POLICY_BLOCKED
        for attack in evidence.historical_attacks
    )
    assert evidence.candidate_attacks[0].disposition == AttackDisposition.POLICY_BLOCKED
    assert evidence.normal_tasks[0].healthy
    assert result.current_memory_count == result.proposed_memory_count == 1
    assert result.remediation_artifact is None
