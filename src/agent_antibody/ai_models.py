from __future__ import annotations

import json
from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated

from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent_antibody.contracts import EventType, JsonPrimitive, JsonValue, TraceEvent
from agent_antibody.core_types import tool_id
from agent_antibody.generic_oracle import GenericFinding, GenericFindingCode
from agent_antibody.manifests import TargetManifest
from agent_antibody.oracle import Finding, FindingCode, OracleStatus
from agent_antibody.policy import ArgumentPredicate, PolicyAction
from agent_antibody.runner import ScenarioRun


class AttackArgument(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    value: JsonPrimitive


class AttackTechnique(StrEnum):
    AUTHORITY_CLAIM = "authority_claim"
    APPROVAL_FORGERY = "approval_forgery"
    URGENCY_PRESSURE = "urgency_pressure"
    PROCEDURAL_CHECKLIST = "procedural_checklist"
    STRUCTURED_DATA_COMMAND = "structured_data_command"
    TOOL_OUTPUT_SPOOFING = "tool_output_spoofing"
    POLICY_EXCEPTION = "policy_exception"
    WORKFLOW_RESUME = "workflow_resume"
    CONDITIONAL_TRIGGER = "conditional_trigger"
    LOCALIZATION_OBFUSCATION = "localization_obfuscation"


class AttackPlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    plan_id: str
    target_id: str
    surface_id: str
    payload: Annotated[str, Field(min_length=1, max_length=2_000)]
    expected_tool: str
    expected_arguments: tuple[AttackArgument, ...]
    rationale: Annotated[str, Field(min_length=1, max_length=1_000)]
    technique: AttackTechnique = AttackTechnique.AUTHORITY_CLAIM

    def validate_for(self, manifest: TargetManifest) -> None:
        if self.target_id != manifest.target_id:
            raise ValueError("attack target_id does not match the target manifest")
        manifest.surface(self.surface_id)
        tool = manifest.tool(self.expected_tool)
        if manifest.attack_profile is not None:
            if self.expected_tool != manifest.attack_profile.target_tool:
                raise ValueError("attack tool is outside the target attack profile")
            if self.surface_id not in manifest.attack_profile.supported_surface_ids:
                supported = ", ".join(manifest.attack_profile.supported_surface_ids)
                raise ValueError(
                    f"attack surface is outside the target attack profile; supported: {supported}"
                )
        if not tool.mutates_state:
            raise ValueError("attack expected_tool must mutate target state")
        properties = tool.input_schema.get("properties")
        if not isinstance(properties, dict):
            raise ValueError("target tool has no object properties schema")
        argument_names = {argument.name for argument in self.expected_arguments}
        if len(argument_names) != len(self.expected_arguments):
            raise ValueError("attack contains duplicate tool arguments")
        arguments = self.arguments_dict()
        validation_errors = sorted(
            Draft202012Validator(tool.input_schema).iter_errors(  # pyright: ignore[reportUnknownMemberType]
                arguments
            ),
            key=lambda error: tuple(str(part) for part in error.absolute_path),
        )
        if validation_errors:
            details = "; ".join(error.message for error in validation_errors)
            raise ValueError(f"attack arguments do not match the tool schema: {details}")

    def arguments_dict(self) -> dict[str, JsonPrimitive]:
        return {argument.name: argument.value for argument in self.expected_arguments}


class AttackSuite(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    attacks: Annotated[tuple[AttackPlan, ...], Field(min_length=1, max_length=10)]

    @model_validator(mode="after")
    def require_distinct_attacks(self) -> AttackSuite:
        plan_ids = [attack.plan_id for attack in self.attacks]
        if len(plan_ids) != len(set(plan_ids)):
            raise ValueError("attack plan IDs must be unique")
        normalized_payloads = [" ".join(attack.payload.lower().split()) for attack in self.attacks]
        if len(normalized_payloads) != len(set(normalized_payloads)):
            raise ValueError("attack payloads must be unique")
        if len(self.attacks) == len(AttackTechnique):
            techniques = {attack.technique for attack in self.attacks}
            if techniques != set(AttackTechnique):
                raise ValueError("ten-attack suites must cover every attack technique once")
        return self

    def validate_for(self, manifest: TargetManifest) -> None:
        for attack in self.attacks:
            attack.validate_for(manifest)
        if len(self.attacks) == len(AttackTechnique) and manifest.attack_profile is not None:
            observed_surfaces = {attack.surface_id for attack in self.attacks}
            missing_surfaces = set(manifest.attack_profile.supported_surface_ids).difference(
                observed_surfaces
            )
            if missing_surfaces:
                raise ValueError(
                    f"ten-attack suite must cover every supported surface: {missing_surfaces}"
                )


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
    causal_request_ids: tuple[str, ...] = ()
    requests: tuple[ObservedRequest, ...]

    @classmethod
    def from_trace(
        cls,
        *,
        target_id: str,
        case_id: str,
        events: tuple[TraceEvent, ...],
        oracle_status: OracleStatus | str,
        findings: Sequence[Finding | GenericFinding],
    ) -> AttackEvidence:
        """Normalize either legacy or target-neutral trace evidence for the LLM boundary."""

        causal_codes = {
            FindingCode.UNAUTHORIZED_STATE_CHANGE.value,
            FindingCode.SERVICE_REPLICAS_BELOW_MINIMUM.value,
            GenericFindingCode.SAFETY_INVARIANT_VIOLATED.value,
        }
        decisions = {
            event.request_id: event
            for event in events
            if event.event_type == EventType.POLICY_DECISION and event.request_id is not None
        }
        executed_ids = {
            event.request_id
            for event in events
            if event.event_type == EventType.TOOL_EXECUTED and event.request_id is not None
        }
        changed_ids = {
            event.request_id
            for event in events
            if event.event_type == EventType.STATE_CHANGED and event.request_id is not None
        }
        requests: list[ObservedRequest] = []
        for event in events:
            if event.event_type != EventType.TOOL_REQUESTED or event.request_id is None:
                continue
            arguments = event.payload.get("arguments")
            source_kinds = event.payload.get("source_kinds")
            decision = decisions.get(event.request_id)
            requests.append(
                ObservedRequest(
                    request_id=event.request_id,
                    tool=tool_id(event.tool) if event.tool is not None else "unknown",
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
        normalized_findings = tuple((str(finding.code), finding.request_id) for finding in findings)
        return cls(
            target_id=target_id,
            scenario_id=case_id,
            oracle_status=str(oracle_status),
            findings=tuple(code for code, _request_id in normalized_findings),
            causal_request_ids=tuple(
                dict.fromkeys(
                    request_id
                    for code, request_id in normalized_findings
                    if code in causal_codes and request_id is not None
                )
            ),
            requests=tuple(requests),
        )

    @classmethod
    def from_run(cls, target_id: str, run: ScenarioRun) -> AttackEvidence:
        return cls.from_trace(
            target_id=target_id,
            case_id=run.scenario.scenario_id,
            events=run.events,
            oracle_status=run.oracle.status.value,
            findings=run.oracle.findings,
        )

    @classmethod
    def combine(cls, evidences: Sequence[AttackEvidence]) -> AttackEvidence:
        if not evidences:
            raise ValueError("at least one infected evidence item is required")
        target_id = evidences[0].target_id
        if any(evidence.target_id != target_id for evidence in evidences):
            raise ValueError("combined evidence must belong to one target")
        if any(evidence.oracle_status != "INFECTED" for evidence in evidences):
            raise ValueError("combined evidence requires infected runs")
        requests = tuple(request for evidence in evidences for request in evidence.requests)
        request_ids = [request.request_id for request in requests]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("combined evidence contains duplicate request IDs")
        scenario_ids = tuple(evidence.scenario_id for evidence in evidences)
        return cls(
            target_id=target_id,
            scenario_id="attack-suite:" + ",".join(scenario_ids),
            oracle_status="INFECTED",
            findings=tuple(finding for evidence in evidences for finding in evidence.findings),
            causal_request_ids=tuple(
                request_id for evidence in evidences for request_id in evidence.causal_request_ids
            ),
            requests=requests,
        )

    def prompt_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
