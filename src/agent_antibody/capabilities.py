from __future__ import annotations

import hashlib
import json
import unicodedata
from enum import StrEnum
from typing import Annotated, Literal

from jsonschema import Draft202012Validator
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

from agent_antibody.core_types import JsonObject, ToolId
from agent_antibody.manifests import TargetManifest, ToolManifest

_JSON_OBJECT: TypeAdapter[JsonObject] = TypeAdapter(JsonObject)


def normalize_capability_text(value: str) -> str:
    """Normalize provider-facing text without changing its meaning."""

    normalized = unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in normalized.split("\n")).strip()


def canonical_input_schema(value: object) -> JsonObject:
    """Validate and deterministically order an ADK function input schema."""

    schema = _JSON_OBJECT.validate_python(value)
    if schema.get("type") != "object":
        raise ValueError("tool input schema must have object type")
    Draft202012Validator.check_schema(schema)
    encoded = json.dumps(
        schema,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return _JSON_OBJECT.validate_json(encoded)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


class AdkToolCapability(BaseModel):
    """A function declaration observed in the request ADK sent to its model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: ToolId
    description: str
    input_schema: JsonObject

    @field_validator("description")
    @classmethod
    def normalize_description(cls, value: str) -> str:
        normalized = normalize_capability_text(value)
        if not normalized:
            raise ValueError("tool description must not be empty")
        return normalized

    @field_validator("input_schema")
    @classmethod
    def normalize_schema(cls, value: JsonObject) -> JsonObject:
        return canonical_input_schema(value)

    def digest(self) -> str:
        return _digest(self.model_dump(mode="json"))


class CapabilitySnapshot(BaseModel):
    """Credential-free observation of an ADK agent's outbound capability surface."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    connector: Literal["google-adk"] = "google-adk"
    adk_version: str
    agent_name: str
    instruction_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    tools: tuple[AdkToolCapability, ...]

    @field_validator("adk_version", "agent_name")
    @classmethod
    def require_nonempty_metadata(cls, value: str) -> str:
        normalized = normalize_capability_text(value)
        if not normalized:
            raise ValueError("capability snapshot metadata must not be empty")
        return normalized

    @field_validator("tools")
    @classmethod
    def normalize_tools(cls, value: tuple[AdkToolCapability, ...]) -> tuple[AdkToolCapability, ...]:
        if not value:
            raise ValueError("capability extraction returned no function tools")
        names = [tool.name for tool in value]
        if len(names) != len(set(names)):
            raise ValueError("capability extraction returned duplicate tool names")
        return tuple(sorted(value, key=lambda tool: tool.name))

    def canonical_json(self) -> str:
        return _canonical_json(self.model_dump(mode="json")).decode()

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()

    def tool(self, name: str) -> AdkToolCapability:
        try:
            return next(tool for tool in self.tools if tool.name == name)
        except StopIteration as error:
            raise ValueError(f"capability snapshot has no tool named {name!r}") from error


class AttackSurfaceChangeKind(StrEnum):
    ADDED_TOOL = "ADDED_TOOL"
    REMOVED_TOOL = "REMOVED_TOOL"
    INPUT_SCHEMA_CHANGED = "INPUT_SCHEMA_CHANGED"
    DESCRIPTION_CHANGED = "DESCRIPTION_CHANGED"
    AGENT_INSTRUCTION_CHANGED = "AGENT_INSTRUCTION_CHANGED"


class AttackSurfaceChange(BaseModel):
    """One deterministic Base/Head capability difference."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: AttackSurfaceChangeKind
    tool_name: ToolId | None = None
    before_tool: AdkToolCapability | None = None
    after_tool: AdkToolCapability | None = None
    before_instruction_sha256: str | None = None
    after_instruction_sha256: str | None = None
    attack_required: bool

    @model_validator(mode="after")
    def validate_shape(self) -> AttackSurfaceChange:
        if self.kind == AttackSurfaceChangeKind.AGENT_INSTRUCTION_CHANGED:
            if (
                self.tool_name is not None
                or self.before_tool is not None
                or self.after_tool is not None
            ):
                raise ValueError("instruction change cannot reference a tool")
            if self.before_instruction_sha256 is None or self.after_instruction_sha256 is None:
                raise ValueError("instruction change requires both instruction digests")
            if not self.attack_required:
                raise ValueError("instruction changes must require attack generation")
            return self

        if self.tool_name is None:
            raise ValueError("tool capability change requires a tool name")
        if self.before_instruction_sha256 is not None or self.after_instruction_sha256 is not None:
            raise ValueError("tool capability change cannot contain instruction digests")
        if self.kind == AttackSurfaceChangeKind.ADDED_TOOL:
            if self.before_tool is not None or self.after_tool is None:
                raise ValueError("added tool change requires only after_tool")
        elif self.kind == AttackSurfaceChangeKind.REMOVED_TOOL:
            if self.before_tool is None or self.after_tool is not None:
                raise ValueError("removed tool change requires only before_tool")
        elif self.before_tool is None or self.after_tool is None:
            raise ValueError("changed tool requires both tool declarations")
        if self.kind != AttackSurfaceChangeKind.REMOVED_TOOL and not self.attack_required:
            raise ValueError("new and changed tools must require attack generation")
        return self


class AttackSurfaceDelta(BaseModel):
    """Stable capability delta consumed by target-specific attack generation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    connector: Literal["google-adk"] = "google-adk"
    base_snapshot_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    head_snapshot_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    changes: tuple[AttackSurfaceChange, ...]
    attack_required_tools: tuple[ToolId, ...]

    @field_validator("attack_required_tools")
    @classmethod
    def normalize_attack_required_tools(cls, value: tuple[ToolId, ...]) -> tuple[ToolId, ...]:
        return tuple(sorted(set(value)))

    @property
    def changed(self) -> bool:
        return bool(self.changes)


class CapabilityComparisonError(ValueError):
    """Raised when two snapshots cannot be compared without guessing."""


def diff_capability_snapshots(
    base: CapabilitySnapshot,
    head: CapabilitySnapshot,
) -> AttackSurfaceDelta:
    """Compare two observations without trusting developer-authored risk metadata."""

    if base.connector != head.connector:
        raise CapabilityComparisonError("capability connector changed")
    if base.adk_version != head.adk_version:
        raise CapabilityComparisonError(
            "ADK version changed; capture both revisions with one version"
        )

    base_tools = {tool.name: tool for tool in base.tools}
    head_tools = {tool.name: tool for tool in head.tools}
    changes: list[AttackSurfaceChange] = []
    attack_required_tools: set[str] = set()

    for name in sorted(base_tools.keys() | head_tools.keys()):
        before = base_tools.get(name)
        after = head_tools.get(name)
        if before is None and after is not None:
            changes.append(
                AttackSurfaceChange(
                    kind=AttackSurfaceChangeKind.ADDED_TOOL,
                    tool_name=name,
                    after_tool=after,
                    attack_required=True,
                )
            )
            attack_required_tools.add(name)
            continue
        if before is not None and after is None:
            changes.append(
                AttackSurfaceChange(
                    kind=AttackSurfaceChangeKind.REMOVED_TOOL,
                    tool_name=name,
                    before_tool=before,
                    attack_required=False,
                )
            )
            continue
        if before is None or after is None:
            raise AssertionError("tool set comparison reached an impossible state")
        if before.input_schema != after.input_schema:
            changes.append(
                AttackSurfaceChange(
                    kind=AttackSurfaceChangeKind.INPUT_SCHEMA_CHANGED,
                    tool_name=name,
                    before_tool=before,
                    after_tool=after,
                    attack_required=True,
                )
            )
            attack_required_tools.add(name)
        if before.description != after.description:
            changes.append(
                AttackSurfaceChange(
                    kind=AttackSurfaceChangeKind.DESCRIPTION_CHANGED,
                    tool_name=name,
                    before_tool=before,
                    after_tool=after,
                    attack_required=True,
                )
            )
            attack_required_tools.add(name)

    if base.instruction_sha256 != head.instruction_sha256:
        changes.append(
            AttackSurfaceChange(
                kind=AttackSurfaceChangeKind.AGENT_INSTRUCTION_CHANGED,
                before_instruction_sha256=base.instruction_sha256,
                after_instruction_sha256=head.instruction_sha256,
                attack_required=True,
            )
        )

    return AttackSurfaceDelta(
        base_snapshot_sha256=base.digest(),
        head_snapshot_sha256=head.digest(),
        changes=tuple(changes),
        attack_required_tools=tuple(attack_required_tools),
    )


def attack_manifest_for_capability(
    manifest: TargetManifest,
    capability: AdkToolCapability,
) -> TargetManifest:
    """Add an observed ADK declaration to the bounded attack/policy schema.

    ``mutates_state`` here means "treat as potentially state changing while under
    test". It is not trusted developer metadata: a policy can only be persisted
    after the deterministic trace proves a causal unsafe state change.
    """

    observed = ToolManifest(
        name=capability.name,
        description=capability.description,
        input_schema=capability.input_schema,
        risk_tags=(),
        mutates_state=True,
    )
    existing = tuple(tool for tool in manifest.tools if tool.name == capability.name)
    if existing:
        if existing[0].input_schema != capability.input_schema:
            raise ValueError("hand manifest and observed ADK tool schema disagree")
        tools = tuple(observed if tool.name == capability.name else tool for tool in manifest.tools)
    else:
        tools = (*manifest.tools, observed)
    return manifest.model_copy(update={"tools": tools, "attack_profile": None})
