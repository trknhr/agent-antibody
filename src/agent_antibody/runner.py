from __future__ import annotations

import secrets
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from agent_antibody.agent import AgentAdapter, AgentRunResult, ReplayOpsMate
from agent_antibody.contracts import PolicyMode, TaskContract, TraceEvent
from agent_antibody.gateway import PolicyGateway
from agent_antibody.oracle import (
    DeterministicOracle,
    OracleResult,
    RunArtifact,
)
from agent_antibody.policy import PolicyEngine, PolicyRules
from agent_antibody.scenarios import Scenario
from agent_antibody.signing import ApprovalSigner, ContractSigner
from agent_antibody.simulator import SimulatorSnapshot
from agent_antibody.tools import BoundTools
from agent_antibody.trace import TraceRecorder


class ScenarioRun(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: PolicyMode
    scenario: Scenario
    contract: TaskContract
    initial_state: SimulatorSnapshot
    final_state: SimulatorSnapshot
    events: tuple[TraceEvent, ...]
    agent_result: AgentRunResult
    oracle: OracleResult

    def artifact(self) -> RunArtifact:
        return RunArtifact(
            scenario=self.scenario,
            contract=self.contract,
            initial_state=self.initial_state,
            final_state=self.final_state,
            events=self.events,
        )


def run_scenario(
    scenario: Scenario,
    *,
    mode: PolicyMode,
    rules: PolicyRules | None = None,
    agent: AgentAdapter | None = None,
    signing_secret: bytes | None = None,
) -> ScenarioRun:
    secret = signing_secret or secrets.token_bytes(32)
    contract_signer = ContractSigner(secret)
    approval_signer = ApprovalSigner(secret)
    signed_contract = scenario.issue_contract(contract_signer, caller_id="opsmate")
    simulator = scenario.simulator()
    initial_state = simulator.snapshot()
    recorder = TraceRecorder(run_id=str(uuid4()))
    policy = PolicyEngine(rules or PolicyRules(), approval_signer)
    gateway = PolicyGateway(
        mode=mode,
        simulator=simulator,
        recorder=recorder,
        contract_signer=contract_signer,
        policy=policy,
    )
    tools = BoundTools(
        caller_id="opsmate",
        signed_contract=signed_contract,
        gateway=gateway,
    )

    agent_result = (agent or ReplayOpsMate()).run(scenario, tools)
    recorder.complete(success=agent_result.success)
    final_state = simulator.snapshot()
    artifact = RunArtifact(
        scenario=scenario,
        contract=signed_contract.contract,
        initial_state=initial_state,
        final_state=final_state,
        events=recorder.events,
    )
    oracle = DeterministicOracle().evaluate(artifact)
    return ScenarioRun(
        mode=mode,
        scenario=scenario,
        contract=signed_contract.contract,
        initial_state=initial_state,
        final_state=final_state,
        events=recorder.events,
        agent_result=agent_result,
        oracle=oracle,
    )
