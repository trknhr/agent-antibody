from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

import agent_antibody.immunity_artifacts as immunity_artifacts
from agent_antibody.candidate_evaluation import (
    AttackCohort,
    AttackDisposition,
    CandidateAttackEvidence,
    CandidateEvaluationV2,
    CandidateNormalTaskEvidence,
    CandidateTargetEvidence,
    CandidateToolRequestEvidence,
    RemediationArtifactReference,
    TargetCandidateEvaluation,
)
from agent_antibody.enforcement import run_with_persisted_immunity
from agent_antibody.immunity_artifacts import (
    CandidateImmunity,
    ImmunityArtifact,
    ImmunityEvaluation,
    SnapshotLifecycle,
    apply_evaluation,
    audit_immutable_changes,
    create_snapshot,
    detect_affected_targets,
    load_artifacts,
    load_effective_policy,
    load_snapshot,
    snapshot_path,
    source_fingerprint,
    validate_evaluation_for_apply,
    verify_artifact,
    verify_snapshot,
)
from agent_antibody.manifests import RiskTag, TargetManifest, ToolManifest
from agent_antibody.portfolio_demo import run_target_demo
from agent_antibody.targets.base import TargetAdapter
from agent_antibody.targets.registry import TARGET_ADAPTERS, TARGET_IDS, get_target_adapter

SOURCE_REVISION = "a" * 40
CAPTURED_AT = datetime(2026, 7, 11, tzinfo=UTC)


class _ManifestOverrideAdapter:
    def __init__(self, delegate: TargetAdapter, manifest: TargetManifest) -> None:
        self._delegate = delegate
        self._manifest = manifest

    @property
    def manifest(self) -> TargetManifest:
        return self._manifest

    def __getattr__(self, name: str) -> object:
        return getattr(self._delegate, name)


def _with_supportmate_manifest(
    monkeypatch: pytest.MonkeyPatch,
    manifest: TargetManifest,
) -> None:
    adapter = get_target_adapter("supportmate")
    monkeypatch.setitem(
        TARGET_ADAPTERS,
        "supportmate",
        cast(TargetAdapter, _ManifestOverrideAdapter(adapter, manifest)),
    )


def _manifest_with_unrelated_tool(manifest: TargetManifest) -> TargetManifest:
    added = ToolManifest(
        name="grant_store_credit",
        description="Grant store credit to a customer account.",
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "customer_id": {"type": "string"},
                "amount_cents": {"type": "integer", "minimum": 1},
            },
            "required": ["customer_id", "amount_cents"],
        },
        risk_tags=(RiskTag.WRITE, RiskTag.FINANCIAL),
        mutates_state=True,
    )
    return manifest.model_copy(update={"tools": (*manifest.tools, added)})


def _as_legacy_v1(artifact: ImmunityArtifact) -> ImmunityArtifact:
    legacy = artifact.model_copy(
        update={
            "schema_version": 1,
            "manifest_contract": None,
        }
    )
    legacy = legacy.model_copy(update={"artifact_id": legacy.expected_artifact_id()})
    return ImmunityArtifact.model_validate(legacy.model_dump(mode="json"))


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


def test_artifact_allows_an_unrelated_tool_addition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = ImmunityArtifact.from_report(
        run_target_demo("supportmate"),
        source_revision=SOURCE_REVISION,
        captured_at=CAPTURED_AT,
    )
    original_manifest = get_target_adapter("supportmate").manifest
    _with_supportmate_manifest(
        monkeypatch,
        _manifest_with_unrelated_tool(original_manifest),
    )

    artifact.validate_for_runtime()
    _as_legacy_v1(artifact).validate_for_runtime()


def test_artifact_rejects_removing_a_referenced_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = ImmunityArtifact.from_report(
        run_target_demo("supportmate"),
        source_revision=SOURCE_REVISION,
        captured_at=CAPTURED_AT,
    )
    original_manifest = get_target_adapter("supportmate").manifest
    incompatible_manifest = original_manifest.model_copy(
        update={
            "tools": tuple(tool for tool in original_manifest.tools if tool.name != "issue_refund"),
            "attack_profile": None,
        }
    )
    _with_supportmate_manifest(monkeypatch, incompatible_manifest)

    with pytest.raises(ValueError, match="referenced tool input schema changed: issue_refund"):
        artifact.validate_for_runtime()


def test_artifact_rejects_a_referenced_tool_schema_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = ImmunityArtifact.from_report(
        run_target_demo("supportmate"),
        source_revision=SOURCE_REVISION,
        captured_at=CAPTURED_AT,
    )
    original_manifest = get_target_adapter("supportmate").manifest
    changed_tools = tuple(
        tool.model_copy(
            update={
                "input_schema": {
                    **tool.input_schema,
                    "properties": {
                        **cast(dict[str, object], tool.input_schema["properties"]),
                        "amount_cents": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 1_000,
                        },
                    },
                }
            }
        )
        if tool.name == "issue_refund"
        else tool
        for tool in original_manifest.tools
    )
    _with_supportmate_manifest(
        monkeypatch,
        original_manifest.model_copy(update={"tools": changed_tools}),
    )

    with pytest.raises(ValueError, match="referenced tool input schema changed: issue_refund"):
        artifact.validate_for_runtime()

    with pytest.raises(ValueError, match="arguments do not match the tool schema"):
        _as_legacy_v1(artifact).validate_for_runtime()


def test_manifest_contract_is_content_addressed() -> None:
    artifact = ImmunityArtifact.from_report(
        run_target_demo("supportmate"),
        source_revision=SOURCE_REVISION,
        captured_at=CAPTURED_AT,
    )
    payload = cast(dict[str, object], artifact.model_dump(mode="json"))
    manifest_contract = cast(dict[str, object], payload["manifest_contract"])
    tools = cast(list[dict[str, object]], manifest_contract["tools"])
    schema = cast(dict[str, object], tools[0]["input_schema"])
    schema["title"] = "tampered"

    with pytest.raises(ValidationError, match="artifact_id does not match"):
        ImmunityArtifact.model_validate(payload)


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

    with pytest.raises(ValidationError, match="report_sha256 does not match"):
        ImmunityArtifact.model_validate(payload)


def test_artifact_secret_boundary_rejects_persisted_payloads() -> None:
    artifact = ImmunityArtifact.from_report(
        run_target_demo("opsmate"),
        source_revision=SOURCE_REVISION,
        captured_at=CAPTURED_AT,
    )
    for payload in (
        "token=super-secret",
        '{"api_key":"super-secret"}',
        "Authorization: Bearer super-secret",
    ):
        attack = artifact.regression.attacks[0].model_copy(update={"payload": payload})

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
    assert verify_snapshot(repository_root=tmp_path) == snapshot
    assert load_effective_policy(repository_root=tmp_path, target_id="supportmate").rules
    assert all(verify_artifact(artifact).passed for artifact in artifacts)
    support_adapter = get_target_adapter("supportmate")
    support_artifact = next(
        artifact for artifact in artifacts if artifact.target_id == "supportmate"
    )
    support_case = support_adapter.materialize_attack(
        support_artifact.regression.attacks[0].to_plan()
    )
    protected = run_with_persisted_immunity(
        support_case,
        adapter=support_adapter,
        agent=support_adapter.create_harness_agent(support_case, protected=True),
        repository_root=tmp_path,
    )
    assert protected.oracle.status.value == "IMMUNE"
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


def test_trusted_apply_rejects_an_empty_or_unbound_candidate(tmp_path: Path) -> None:
    source = tmp_path / "src" / "agent_antibody"
    source.mkdir(parents=True)
    (source / "candidate.py").write_text("# candidate source\n", encoding="utf-8")
    fingerprint = source_fingerprint(repository_root=tmp_path, target_id="supportmate")
    artifact = ImmunityArtifact.from_report(
        run_target_demo("supportmate"),
        source_revision=SOURCE_REVISION,
        source_fingerprint=fingerprint,
        captured_at=CAPTURED_AT,
    )
    changed_paths = ("src/agent_antibody/targets/supportmate.py",)
    empty = ImmunityEvaluation(
        source_revision=SOURCE_REVISION,
        changed_paths=changed_paths,
        evaluated_at=CAPTURED_AT,
        candidates=(),
    )
    with pytest.raises(ValueError, match="target set"):
        validate_evaluation_for_apply(
            empty,
            repository_root=tmp_path,
            trusted_source_revision=SOURCE_REVISION,
            trusted_changed_paths=changed_paths,
        )

    unbound = ImmunityEvaluation(
        source_revision=SOURCE_REVISION,
        changed_paths=changed_paths,
        evaluated_at=CAPTURED_AT,
        candidates=(CandidateImmunity(action="create", artifact=artifact),),
    )
    with pytest.raises(ValueError, match="changed paths"):
        validate_evaluation_for_apply(
            unbound.model_copy(update={"changed_paths": ("README.md",)}),
            repository_root=tmp_path,
            trusted_source_revision=SOURCE_REVISION,
            trusted_changed_paths=changed_paths,
        )

    wrong_fingerprint_artifact = ImmunityArtifact.from_report(
        run_target_demo("supportmate"),
        source_revision=SOURCE_REVISION,
        source_fingerprint="b" * 64,
        captured_at=CAPTURED_AT,
    )
    with pytest.raises(ValueError, match="source fingerprint"):
        validate_evaluation_for_apply(
            unbound.model_copy(
                update={
                    "candidates": (
                        CandidateImmunity(
                            action="create",
                            artifact=wrong_fingerprint_artifact,
                        ),
                    )
                }
            ),
            repository_root=tmp_path,
            trusted_source_revision=SOURCE_REVISION,
            trusted_changed_paths=changed_paths,
        )


def test_trusted_apply_accepts_one_remediation_after_full_target_assessment(
    tmp_path: Path,
) -> None:
    source = tmp_path / "src" / "agent_antibody"
    source.mkdir(parents=True)
    (source / "candidate.py").write_text("# candidate source\n", encoding="utf-8")
    fingerprint = source_fingerprint(repository_root=tmp_path, target_id="supportmate")
    artifact = ImmunityArtifact.from_report(
        run_target_demo("supportmate"),
        source_revision=SOURCE_REVISION,
        source_fingerprint=fingerprint,
        captured_at=CAPTURED_AT,
    )
    evaluation = ImmunityEvaluation(
        source_revision=SOURCE_REVISION,
        changed_paths=("src/agent_antibody/ai_agents.py",),
        evaluated_at=CAPTURED_AT,
        candidates=(CandidateImmunity(action="create", artifact=artifact),),
    )

    def revalidated(target_id: str) -> TargetCandidateEvaluation:
        return TargetCandidateEvaluation.revalidated(
            target_id=target_id,
            current_memory_count=1,
            evidence=CandidateTargetEvidence(
                existing_policy_artifact_ids=(f"imm-{target_id}-existing",),
                attacks=(
                    CandidateAttackEvidence(
                        plan_id=f"{target_id}-existing-attack",
                        cohort=AttackCohort.CANDIDATE,
                        disposition=AttackDisposition.POLICY_BLOCKED,
                        expected_tool="existing_tool",
                        requests=(
                            CandidateToolRequestEvidence(
                                request_id=f"{target_id}-request",
                                tool="existing_tool",
                                policy_blocked=True,
                            ),
                        ),
                    ),
                ),
                normal_tasks=(
                    CandidateNormalTaskEvidence(
                        case_id=f"{target_id}-normal",
                        healthy=True,
                        observed_tools=("existing_tool",),
                    ),
                ),
            ),
        )

    reference = RemediationArtifactReference(
        artifact_id=artifact.artifact_id,
        target_id="supportmate",
        report_sha256=artifact.report_sha256,
        policy_rule_ids=tuple(rule.rule_id for rule in artifact.policy.rules),
        regression_plan_ids=tuple(plan.plan_id for plan in artifact.regression.attacks),
    )
    supportmate = TargetCandidateEvaluation.bypass_confirmed(
        target_id="supportmate",
        current_memory_count=1,
        evidence=CandidateTargetEvidence(
            existing_policy_artifact_ids=("imm-supportmate-existing",),
            attack_surface_change_ids=("ADDED_TOOL:grant_store_credit",),
            attacks=(
                CandidateAttackEvidence(
                    plan_id=artifact.regression.attacks[0].plan_id,
                    cohort=AttackCohort.CANDIDATE,
                    disposition=AttackDisposition.BYPASS_CONFIRMED,
                    expected_tool=artifact.regression.attacks[0].expected_tool,
                    requests=(
                        CandidateToolRequestEvidence(
                            request_id="supportmate-bypass-request",
                            tool=artifact.regression.attacks[0].expected_tool,
                            executed=True,
                            changed_state=True,
                            unsafe_state_change=True,
                        ),
                    ),
                    finding_codes=("unsafe_state_change",),
                ),
            ),
            normal_tasks=(
                CandidateNormalTaskEvidence(
                    case_id="supportmate-normal",
                    healthy=True,
                    observed_tools=(artifact.regression.attacks[0].expected_tool,),
                ),
            ),
        ),
        remediation_artifact=reference,
    )
    assessment = CandidateEvaluationV2.from_targets(
        base_revision=SOURCE_REVISION,
        source_revision=SOURCE_REVISION,
        evaluated_at=CAPTURED_AT,
        targets=(revalidated("opsmate"), revalidated("repomate"), supportmate),
    )

    validate_evaluation_for_apply(
        evaluation,
        repository_root=tmp_path,
        assessment=assessment,
        trusted_source_revision=SOURCE_REVISION,
        trusted_changed_paths=("src/agent_antibody/ai_agents.py",),
    )


def test_audit_skips_a_stale_base_snapshot_when_source_pr_changes_no_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = ImmunityArtifact.from_report(
        run_target_demo("repomate"),
        source_revision=SOURCE_REVISION,
        captured_at=CAPTURED_AT,
    )
    apply_evaluation(
        ImmunityEvaluation(
            source_revision=SOURCE_REVISION,
            changed_paths=("src/agent_antibody/targets/repomate.py",),
            evaluated_at=CAPTURED_AT,
            candidates=(CandidateImmunity(action="create", artifact=artifact),),
        ),
        repository_root=tmp_path,
    )
    for command in (
        ("git", "init"),
        ("git", "config", "user.email", "test@example.com"),
        ("git", "config", "user.name", "Agent Antibody Test"),
        ("git", "add", "."),
        ("git", "commit", "-m", "baseline immunity"),
    ):
        subprocess.run(command, cwd=tmp_path, check=True, capture_output=True)
    base_revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source = tmp_path / "src" / "agent_antibody" / "targets"
    source.mkdir(parents=True)
    (source / "repomate.py").write_text("# candidate tool delta\n", encoding="utf-8")
    subprocess.run(("git", "add", "."), cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ("git", "commit", "-m", "candidate adds a tool"),
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    def snapshot_must_not_be_verified(*, repository_root: Path) -> object:
        del repository_root
        raise AssertionError("source PR did not change immutable memory")

    monkeypatch.setattr(immunity_artifacts, "verify_snapshot", snapshot_must_not_be_verified)

    audit_immutable_changes(repository_root=tmp_path, base_revision=base_revision)


def test_snapshot_rejects_unallowlisted_nested_content(tmp_path: Path) -> None:
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
    raw = json.loads(snapshot_path(repository_root=tmp_path).read_text(encoding="utf-8"))
    raw["targets"][0]["latest_memory"]["memory_seed"]["payload"] = "must-not-publish"
    snapshot_path(repository_root=tmp_path).write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValidationError, match="payload"):
        verify_snapshot(repository_root=tmp_path)
