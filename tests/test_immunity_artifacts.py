from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from agent_antibody.immunity_artifacts import (
    CandidateImmunity,
    ImmunityArtifact,
    ImmunityEvaluation,
    SnapshotLifecycle,
    apply_evaluation,
    create_snapshot,
    detect_affected_targets,
    load_artifacts,
    load_effective_policy,
    load_snapshot,
    verify_artifact,
)
from agent_antibody.portfolio_demo import run_target_demo
from agent_antibody.targets.registry import TARGET_IDS

SOURCE_REVISION = "a" * 40
CAPTURED_AT = datetime(2026, 7, 11, tzinfo=UTC)


@pytest.mark.parametrize("target_id", TARGET_IDS)
def test_immutable_artifact_round_trips_and_replays_every_target(target_id: str) -> None:
    artifact = ImmunityArtifact.from_report(
        run_target_demo(target_id),
        source_revision=SOURCE_REVISION,
        captured_at=CAPTURED_AT,
    )

    loaded = ImmunityArtifact.from_yaml(artifact.to_yaml())
    verification = verify_artifact(loaded)

    assert loaded == artifact
    assert verification.passed
    assert verification.blocked_attacks == 10
    assert verification.healthy_normal_cases == 1


def test_immutable_artifact_rejects_tampering() -> None:
    artifact = ImmunityArtifact.from_report(
        run_target_demo("opsmate"),
        source_revision=SOURCE_REVISION,
        captured_at=CAPTURED_AT,
    )
    payload = cast(dict[str, object], artifact.model_dump(mode="json"))
    regression = cast(dict[str, object], payload["regression"])
    attacks = regression["attacks"]
    assert isinstance(attacks, list)
    first = cast(object, attacks[0])
    assert isinstance(first, dict)
    first["payload"] = "quietly change the service"

    with pytest.raises(ValidationError, match="artifact_id does not match"):
        ImmunityArtifact.model_validate(payload)


def test_artifact_secret_boundary_rejects_persisted_payloads() -> None:
    artifact = ImmunityArtifact.from_report(
        run_target_demo("opsmate"),
        source_revision=SOURCE_REVISION,
        captured_at=CAPTURED_AT,
    )
    attack = artifact.regression.attacks[0].model_copy(update={"payload": "token=super-secret"})

    with pytest.raises(ValueError, match="appears to contain a secret"):
        attack.validate_secret_boundary()


def test_apply_evaluation_writes_only_immutable_memories_and_safe_snapshot(tmp_path: Path) -> None:
    candidates = tuple(
        CandidateImmunity(
            action="create",
            artifact=ImmunityArtifact.from_report(
                run_target_demo(target_id),
                source_revision=SOURCE_REVISION,
                captured_at=CAPTURED_AT,
            ),
        )
        for target_id in TARGET_IDS
    )
    evaluation = ImmunityEvaluation(
        source_revision=SOURCE_REVISION,
        changed_paths=("src/agent_antibody/generic_gateway.py",),
        evaluated_at=CAPTURED_AT,
        candidates=candidates,
    )

    written = apply_evaluation(
        evaluation,
        repository_root=tmp_path,
        lifecycle=SnapshotLifecycle(
            state="verified_pending_review",
            source_pr_number=42,
            remediation_branch="antibody/pr-42-aabbccdd",
        ),
    )
    artifacts = load_artifacts(repository_root=tmp_path)
    snapshot = load_snapshot(repository_root=tmp_path)

    assert len(written) == 4
    assert len(artifacts) == 3
    assert {artifact.target_id for artifact in artifacts} == set(TARGET_IDS)
    assert len(snapshot.targets) == 3
    assert snapshot.lifecycle.state == "verified_pending_review"
    assert snapshot.lifecycle.source_pr_number == 42
    assert "payload" not in snapshot.model_dump_json()
    assert load_effective_policy(repository_root=tmp_path, target_id="supportmate").rules
    assert all(verify_artifact(artifact).passed for artifact in artifacts)
    assert apply_evaluation(evaluation, repository_root=tmp_path) == ()


def test_changed_path_detection_is_target_aware() -> None:
    assert detect_affected_targets(("src/agent_antibody/targets/opsmate.py",)) == ("opsmate",)
    assert detect_affected_targets(("src/agent_antibody/targets/repomate.py",)) == ("repomate",)
    assert detect_affected_targets(("src/agent_antibody/ai_agents.py",)) == TARGET_IDS
    assert detect_affected_targets(("README.md",)) == ()
    assert detect_affected_targets(("immunities/v1/opsmate/example.yaml",)) == ()


def test_snapshot_uses_only_persisted_memories(tmp_path: Path) -> None:
    artifact = ImmunityArtifact.from_report(
        run_target_demo("repomate"),
        source_revision=SOURCE_REVISION,
        captured_at=CAPTURED_AT,
    )
    evaluation = ImmunityEvaluation(
        source_revision=SOURCE_REVISION,
        changed_paths=("src/agent_antibody/targets/repomate.py",),
        evaluated_at=CAPTURED_AT,
        candidates=(CandidateImmunity(action="create", artifact=artifact),),
    )

    apply_evaluation(evaluation, repository_root=tmp_path)
    snapshot = create_snapshot(repository_root=tmp_path, source_revision=SOURCE_REVISION)

    assert [target.target_id for target in snapshot.targets] == ["repomate"]
    assert snapshot.targets[0].memory_count == 1
