from __future__ import annotations

import hashlib
import json
from typing import cast

import yaml
from pydantic import BaseModel, ConfigDict

from agent_antibody.ai_models import AntibodyProposal, AttackEvidence, ObservedRequest
from agent_antibody.contracts import JsonValue, SourceKind
from agent_antibody.manifests import TargetManifest
from agent_antibody.policy import PolicyRules, RuleWhen, ToolPolicyRule


class CompiledAntibody(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    bundle_id: str
    target_id: str
    rules: tuple[ToolPolicyRule, ...]
    untrusted_sources: tuple[SourceKind, ...]
    rationale: str

    def policy(self) -> PolicyRules:
        return PolicyRules(rules=self.rules, untrusted_sources=self.untrusted_sources)

    def to_yaml(self) -> str:
        dumped = self.model_dump(mode="json", exclude_none=True)
        payload: dict[str, JsonValue] = {
            "bundle_id": self.bundle_id,
            "target_id": self.target_id,
            "rules": cast(list[JsonValue], dumped["rules"]),
            "untrusted_sources": [source.value for source in self.untrusted_sources],
        }
        return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)


class PolicyCompiler:
    """Validates LLM output and compiles only the bounded policy DSL."""

    def compile(
        self,
        *,
        proposal: AntibodyProposal,
        manifest: TargetManifest,
        evidence: AttackEvidence,
    ) -> CompiledAntibody:
        if proposal.target_id != manifest.target_id or evidence.target_id != manifest.target_id:
            raise ValueError("antibody target does not match evidence and manifest")

        evidence_by_id = {request.request_id: request for request in evidence.requests}
        compiled_rules: list[ToolPolicyRule] = []
        matched_state_change = False
        for index, proposed in enumerate(proposal.rules, start=1):
            tool = manifest.tool(proposed.tool)
            if not tool.mutates_state:
                raise ValueError("antibody rules may only target state-changing tools")
            properties = tool.input_schema.get("properties")
            if not isinstance(properties, dict):
                raise ValueError("tool input schema has no properties")
            unknown_fields = {
                predicate.argument
                for predicate in proposed.predicates
                if predicate.argument not in properties
            }
            if unknown_fields:
                raise ValueError(f"antibody uses unknown argument fields: {unknown_fields}")
            if not proposed.evidence_request_ids:
                raise ValueError("antibody rule must cite an evidence request")
            cited_requests: list[ObservedRequest] = []
            for request_id in proposed.evidence_request_ids:
                request = evidence_by_id.get(request_id)
                if request is None:
                    raise ValueError("antibody cites an unknown evidence request")
                if request.tool != proposed.tool:
                    raise ValueError("antibody evidence tool does not match its rule")
                cited_requests.append(request)

            when = RuleWhen(predicates=proposed.predicates)
            if not any(when.matches(request.arguments) for request in cited_requests):
                raise ValueError("antibody rule does not match its cited request arguments")
            if any(
                request.changed_state and when.matches(request.arguments)
                for request in cited_requests
            ):
                matched_state_change = True

            semantic_rule = {
                "target": manifest.target_id,
                "tool": proposed.tool,
                "predicates": [
                    predicate.model_dump(mode="json") for predicate in proposed.predicates
                ],
                "action": proposed.action.value,
            }
            digest = hashlib.sha256(
                json.dumps(semantic_rule, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()[:16]
            compiled_rules.append(
                ToolPolicyRule(
                    rule_id=f"antibody-{index}-{digest}",
                    tool=proposed.tool,
                    when=when,
                    action=proposed.action,
                )
            )

        if not matched_state_change:
            raise ValueError("no antibody rule matches a state-changing failure")

        declared_sources = {surface.source_kind for surface in manifest.injection_surfaces}
        unknown_sources = set(proposal.untrusted_sources).difference(declared_sources)
        if unknown_sources:
            raise ValueError(f"antibody declares unknown untrusted sources: {unknown_sources}")
        untrusted_sources = tuple(
            sorted((SourceKind(source) for source in declared_sources), key=str)
        )
        bundle_semantics = {
            "target": manifest.target_id,
            "scenario": evidence.scenario_id,
            "rules": [rule.model_dump(mode="json") for rule in compiled_rules],
        }
        bundle_id = hashlib.sha256(
            json.dumps(bundle_semantics, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        return CompiledAntibody(
            bundle_id=bundle_id,
            target_id=manifest.target_id,
            rules=tuple(compiled_rules),
            untrusted_sources=untrusted_sources,
            rationale=proposal.rationale,
        )
