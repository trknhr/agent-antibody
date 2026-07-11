from __future__ import annotations

import copy
import secrets
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from agent_antibody.agent import AgentRunResult
from agent_antibody.contracts import PolicyMode, TaskContract, TraceEvent
from agent_antibody.core_types import ExecutionCase, JsonObject
from agent_antibody.generic_gateway import BoundToolInvoker, GenericPolicyGateway
from agent_antibody.generic_oracle import (
    CaseArtifact,
    GenericDeterministicOracle,
    GenericOracleResult,
)
from agent_antibody.policy import PolicyEngine, PolicyRules
from agent_antibody.signing import ApprovalSigner, ContractSigner
from agent_antibody.targets.base import TargetAdapter, TargetAgent
from agent_antibody.trace import TraceRecorder


class CaseRun(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: PolicyMode
    case: ExecutionCase
    contract: TaskContract
    initial_state: JsonObject
    final_state: JsonObject
    events: tuple[TraceEvent, ...]
    agent_result: AgentRunResult
    oracle: GenericOracleResult

    def artifact(self) -> CaseArtifact:
        return CaseArtifact(
            case=self.case,
            contract=self.contract,
            initial_state=self.initial_state,
            final_state=self.final_state,
            events=self.events,
            agent_result=self.agent_result,
        )


def run_case(
    case: ExecutionCase,
    *,
    adapter: TargetAdapter,
    mode: PolicyMode,
    rules: PolicyRules | None = None,
    agent: TargetAgent | None = None,
    model: str | None = None,
    signing_secret: bytes | None = None,
    timeout_seconds: float = 60.0,
    max_llm_calls: int = 12,
) -> CaseRun:
    if adapter.manifest.target_id != case.target_id:
        raise ValueError("execution case target does not match its adapter")
    secret = signing_secret or secrets.token_bytes(32)
    contract_signer = ContractSigner(secret)
    approval_signer = ApprovalSigner(secret)
    issued_at = datetime.now(UTC)
    caller_id = f"agent:{case.target_id}"
    contract = TaskContract(
        task_id=f"task:{case.case_id}",
        caller_id=caller_id,
        objective=case.objective,
        resources=case.resources,
        allowed_tools=tuple(capability.tool for capability in case.capabilities),
        capabilities=case.capabilities,
        issued_at=issued_at,
        expires_at=issued_at + timedelta(minutes=10),
    )
    signed_contract = contract_signer.issue(contract)
    runtime = adapter.create_runtime(case)
    initial_state = copy.deepcopy(runtime.snapshot())
    recorder = TraceRecorder(
        run_id=str(uuid4()),
        target_id=case.target_id,
        case_id=case.case_id,
    )
    gateway = GenericPolicyGateway(
        mode=mode,
        runtime=runtime,
        recorder=recorder,
        contract_signer=contract_signer,
        policy=PolicyEngine(rules or PolicyRules(), approval_signer),
    )
    tools = BoundToolInvoker(
        caller_id=caller_id,
        signed_contract=signed_contract,
        gateway=gateway,
    )
    target_agent = agent
    if target_agent is None:
        if model is None:
            raise ValueError("run_case requires an agent or model")
        target_agent = adapter.create_agent(
            model=model,
            timeout_seconds=timeout_seconds,
            max_llm_calls=max_llm_calls,
        )
    try:
        agent_result = target_agent.run(case, tools)
    except Exception as error:
        agent_result = AgentRunResult(
            success=False,
            reply=f"agent_error:{type(error).__name__}",
            tool_results=(),
        )
    recorder.complete(success=agent_result.success)
    final_state = copy.deepcopy(runtime.snapshot())
    artifact = CaseArtifact(
        case=case,
        contract=contract,
        initial_state=initial_state,
        final_state=final_state,
        events=recorder.events,
        agent_result=agent_result,
    )
    oracle = GenericDeterministicOracle().evaluate(artifact)
    return CaseRun(
        mode=mode,
        case=case,
        contract=contract,
        initial_state=initial_state,
        final_state=final_state,
        events=recorder.events,
        agent_result=agent_result,
        oracle=oracle,
    )
