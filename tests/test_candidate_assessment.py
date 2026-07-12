from __future__ import annotations

from pathlib import Path
from shutil import copy2

from agent_antibody.adk_capabilities import capture_target_adk_capabilities
from agent_antibody.candidate_assessment import assess_target_revalidation
from agent_antibody.candidate_evaluation import CandidateEvaluationStatus
from agent_antibody.capabilities import diff_capability_snapshots
from agent_antibody.targets.registry import get_target_adapter

SOURCE_REVISION = "b" * 40
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_BASELINE_SUPPORT_MEMORY = "imm-supportmate-f39e99d38aea0305a38f.yaml"


def _memory_root(tmp_path: Path) -> Path:
    destination = tmp_path / "immunities" / "v1" / "supportmate"
    destination.mkdir(parents=True)
    copy2(
        _REPOSITORY_ROOT / "immunities" / "v1" / "supportmate" / _BASELINE_SUPPORT_MEMORY,
        destination / _BASELINE_SUPPORT_MEMORY,
    )
    return tmp_path


def test_unchanged_adk_surface_revalidates_without_new_memory(tmp_path: Path) -> None:
    root = _memory_root(tmp_path)
    snapshot = capture_target_adk_capabilities(get_target_adapter("supportmate"))
    result = assess_target_revalidation(
        memory_repository_root=root,
        target_id="supportmate",
        source_revision=SOURCE_REVISION,
        capability_delta=diff_capability_snapshots(snapshot, snapshot),
    )

    assert result.status == CandidateEvaluationStatus.REVALIDATED
    assert result.current_memory_count == result.proposed_memory_count == 1
    assert result.remediation_artifact is None


def test_new_adk_tool_cannot_be_declared_revalidated_by_an_old_campaign(
    tmp_path: Path,
) -> None:
    root = _memory_root(tmp_path)
    head = capture_target_adk_capabilities(get_target_adapter("supportmate"))
    base = head.model_copy(
        update={"tools": tuple(tool for tool in head.tools if tool.name != "grant_store_credit")}
    )
    result = assess_target_revalidation(
        memory_repository_root=root,
        target_id="supportmate",
        source_revision=SOURCE_REVISION,
        capability_delta=diff_capability_snapshots(base, head),
    )

    assert result.status == CandidateEvaluationStatus.INCONCLUSIVE
    assert result.current_memory_count == result.proposed_memory_count == 1
    assert result.inconclusive_reasons == (
        "delta-specific attack campaign required for tools: grant_store_credit",
    )


def test_instruction_only_change_fails_closed_until_an_expanded_campaign_exists(
    tmp_path: Path,
) -> None:
    root = _memory_root(tmp_path)
    base = capture_target_adk_capabilities(get_target_adapter("supportmate"))
    head = base.model_copy(update={"instruction_sha256": "c" * 64})

    result = assess_target_revalidation(
        memory_repository_root=root,
        target_id="supportmate",
        source_revision=SOURCE_REVISION,
        capability_delta=diff_capability_snapshots(base, head),
    )

    assert result.status == CandidateEvaluationStatus.INCONCLUSIVE
    assert result.current_memory_count == result.proposed_memory_count == 1
    assert result.inconclusive_reasons == (
        "agent instruction changed without an observed tool delta; an expanded attack campaign "
        "is required",
    )
