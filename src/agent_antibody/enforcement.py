"""Runtime integration point for committed Agent Antibody policy memory."""

from __future__ import annotations

from pathlib import Path

from agent_antibody.contracts import PolicyMode
from agent_antibody.core_types import ExecutionCase
from agent_antibody.generic_runner import CaseRun, run_case
from agent_antibody.immunity_artifacts import load_effective_policy
from agent_antibody.targets.base import TargetAdapter, TargetAgent


def run_with_persisted_immunity(
    case: ExecutionCase,
    *,
    adapter: TargetAdapter,
    agent: TargetAgent,
    repository_root: Path,
) -> CaseRun:
    """Run a target task through the active policy reconstructed from merged memory.

    Target hosts call this instead of constructing an unguarded ``run_case``.  The
    policy is reconstructed solely from immutable ``immunities/v1`` artifacts,
    so merging an Antibody PR changes the policy enforced by the gateway without
    letting an agent or browser choose rules at runtime.
    """

    if case.target_id != adapter.manifest.target_id:
        raise ValueError("execution case target does not match the target adapter")
    return run_case(
        case,
        adapter=adapter,
        mode=PolicyMode.ENFORCE,
        rules=load_effective_policy(
            repository_root=repository_root,
            target_id=adapter.manifest.target_id,
        ),
        agent=agent,
    )
