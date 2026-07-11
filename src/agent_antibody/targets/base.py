from __future__ import annotations

from typing import Protocol

from agent_antibody.agent import AgentRunResult
from agent_antibody.ai_models import AttackPlan
from agent_antibody.contracts import SignedApproval, ToolCallResult
from agent_antibody.core_types import (
    ExecutionCase,
    JsonObject,
    ToolExecution,
    ToolId,
)
from agent_antibody.manifests import TargetManifest


class TargetRuntime(Protocol):
    """Trusted simulator or sandbox behind the policy gateway."""

    def snapshot(self) -> JsonObject: ...

    def validate_arguments(self, tool: ToolId, raw: JsonObject) -> JsonObject: ...

    def execute(self, tool: ToolId, arguments: JsonObject) -> ToolExecution: ...


class ToolInvoker(Protocol):
    """The only target-operation surface exposed to an evaluated agent."""

    def call(
        self,
        tool: ToolId,
        arguments: JsonObject,
        *,
        approval: SignedApproval | None = None,
    ) -> ToolCallResult: ...


class TargetAgent(Protocol):
    def run(self, case: ExecutionCase, tools: ToolInvoker) -> AgentRunResult: ...


class TargetAdapter(Protocol):
    """Binds portable target metadata to trusted target-specific runtime code."""

    @property
    def manifest(self) -> TargetManifest: ...

    def materialize_attack(self, plan: AttackPlan) -> ExecutionCase: ...

    def normal_cases(self) -> tuple[ExecutionCase, ...]: ...

    def create_replay_agent(self) -> TargetAgent:
        """Return the deterministic target behavior used for persisted regressions."""

        ...

    def create_harness_agent(
        self,
        case: ExecutionCase,
        *,
        protected: bool,
    ) -> TargetAgent:
        """Exercise the production ADK agent with a bounded adversarial model.

        The harness is deterministic and credential-free, but it deliberately
        drives the same target implementation that is used by the live Gemini
        path. ``protected`` lets a target's scripted model acknowledge a
        policy-denied mutation while preserving the normal task outcome.
        """

        ...

    def create_runtime(self, case: ExecutionCase) -> TargetRuntime: ...

    def create_agent(
        self,
        *,
        model: str,
        timeout_seconds: float,
        max_llm_calls: int,
    ) -> TargetAgent: ...
