from __future__ import annotations

import json
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from agent_antibody.contracts import EventType, JsonPrimitive, JsonValue
from agent_antibody.manifests import TargetManifest
from agent_antibody.policy import ArgumentPredicate, PolicyAction
from agent_antibody.runner import ScenarioRun


class AttackArgument(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    value: JsonPrimitive


class AttackPlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    plan_id: str
    target_id: str
    surface_id: str
    payload: Annotated[str, Field(min_length=1, max_length=2_000)]
    expected_tool: str
    expected_arguments: tuple[AttackArgument, ...]
    rationale: Annotated[str, Field(min_length=1, max_length=1_000)]

    def validate_for(self, manifest: TargetManifest) -> None:
        if self.target_id != manifest.target_id:
            raise ValueError("attack target_id does not match the target manifest")
        manifest.surface(self.surface_id)
        tool = manifest.tool(self.expected_tool)
        if not tool.mutates_state:
            raise ValueError("attack expected_tool must mutate target state")
        properties = tool.input_schema.get("properties")
        if not isinstance(properties, dict):
            raise ValueError("target tool has no object properties schema")
        argument_names = {argument.name for argument in self.expected_arguments}
        if len(argument_names) != len(self.expected_arguments):
            raise ValueError("attack contains duplicate tool arguments")
        unknown_arguments = argument_names.difference(properties)
        if unknown_arguments:
            raise ValueError(f"attack contains unknown tool arguments: {unknown_arguments}")

    def arguments_dict(self) -> dict[str, JsonPrimitive]:
        return {argument.name: argument.value for argument in self.expected_arguments}


class AttackSuite(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    attacks: Annotated[tuple[AttackPlan, ...], Field(min_length=1, max_length=3)]

    def validate_for(self, manifest: TargetManifest) -> None:
        for attack in self.attacks:
            attack.validate_for(manifest)


class PolicyRuleProposal(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    rule_id: str
    tool: str
    predicates: Annotated[tuple[ArgumentPredicate, ...], Field(max_length=4)] = ()
    action: PolicyAction
    evidence_request_ids: tuple[str, ...]


class AntibodyProposal(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    target_id: str
    rules: Annotated[tuple[PolicyRuleProposal, ...], Field(min_length=1, max_length=3)]
    untrusted_sources: tuple[str, ...]
    rationale: Annotated[str, Field(min_length=1, max_length=1_000)]


class ObservedRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str
    tool: str
    arguments: dict[str, JsonValue]
    source_kinds: tuple[str, ...]
    policy_effect: str | None
    executed: bool
    changed_state: bool


class AttackEvidence(BaseModel):
    """Normalized evidence safe to send to the Antibody Agent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_id: str
    scenario_id: str
    oracle_status: str
    findings: tuple[str, ...]
    requests: tuple[ObservedRequest, ...]

    @classmethod
    def from_run(cls, target_id: str, run: ScenarioRun) -> AttackEvidence:
        decisions = {
            event.request_id: event
            for event in run.events
            if event.event_type == EventType.POLICY_DECISION and event.request_id is not None
        }
        executed_ids = {
            event.request_id
            for event in run.events
            if event.event_type == EventType.TOOL_EXECUTED and event.request_id is not None
        }
        changed_ids = {
            event.request_id
            for event in run.events
            if event.event_type == EventType.STATE_CHANGED and event.request_id is not None
        }
        requests: list[ObservedRequest] = []
        for event in run.events:
            if event.event_type != EventType.TOOL_REQUESTED or event.request_id is None:
                continue
            arguments = event.payload.get("arguments")
            source_kinds = event.payload.get("source_kinds")
            decision = decisions.get(event.request_id)
            requests.append(
                ObservedRequest(
                    request_id=event.request_id,
                    tool=event.tool.value if event.tool is not None else "unknown",
                    arguments=arguments if isinstance(arguments, dict) else {},
                    source_kinds=(
                        tuple(str(kind) for kind in source_kinds)
                        if isinstance(source_kinds, list)
                        else ()
                    ),
                    policy_effect=(
                        str(decision.payload.get("effect")) if decision is not None else None
                    ),
                    executed=event.request_id in executed_ids,
                    changed_state=event.request_id in changed_ids,
                )
            )
        return cls(
            target_id=target_id,
            scenario_id=run.scenario.scenario_id,
            oracle_status=run.oracle.status.value,
            findings=tuple(finding.code.value for finding in run.oracle.findings),
            requests=tuple(requests),
        )

    def prompt_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
