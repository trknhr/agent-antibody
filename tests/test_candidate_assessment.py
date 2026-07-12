from __future__ import annotations

from pathlib import Path

from agent_antibody.adk_capabilities import capture_target_adk_capabilities
from agent_antibody.candidate_assessment import assess_target_revalidation
from agent_antibody.candidate_evaluation import CandidateEvaluationStatus
from agent_antibody.capabilities import AdkToolCapability, diff_capability_snapshots
from agent_antibody.targets.registry import get_target_adapter

SOURCE_REVISION = "b" * 40


def test_unchanged_adk_surface_revalidates_without_new_memory() -> None:
    root = Path(__file__).resolve().parents[1]
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


def test_new_adk_tool_cannot_be_declared_revalidated_by_an_old_campaign() -> None:
    root = Path(__file__).resolve().parents[1]
    base = capture_target_adk_capabilities(get_target_adapter("supportmate"))
    head = base.model_copy(
        update={
            "tools": (
                *base.tools,
                AdkToolCapability(
                    name="grant_store_credit",
                    description="Grant store credit to a customer account.",
                    input_schema={
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "customer_id": {"type": "string"},
                            "amount_cents": {"type": "integer", "minimum": 1},
                            "reason": {"type": "string"},
                        },
                        "required": ["customer_id", "amount_cents", "reason"],
                    },
                ),
            )
        }
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
