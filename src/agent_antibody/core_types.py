from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

type JsonPrimitive = str | int | float | bool | None
type JsonValue = JsonPrimitive | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]

type ToolId = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z][a-z0-9_.-]{0,63}$"),
]
type SourceId = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z][a-z0-9_.-]{0,63}$"),
]


def tool_id(value: str | StrEnum) -> str:
    """Return the stable string representation of a tool identifier."""

    return str(value)


def source_id(value: str | StrEnum) -> str:
    """Return the stable string representation of a provenance source identifier."""

    return str(value)


class ObservedSource(BaseModel):
    """Trusted provenance returned beside, rather than inside, agent-visible output."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    source_kind: SourceId


class ToolExecution(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    output: JsonValue = None
    observed_sources: tuple[ObservedSource, ...] = ()


class ArgumentExpectationPredicate(BaseModel):
    """A bounded, target-neutral matcher for a dangerous tool argument."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    argument: str
    operator: InvariantOperator
    value: JsonPrimitive

    def matches(self, arguments: JsonObject) -> bool:
        candidate = arguments.get(self.argument)
        if self.operator == InvariantOperator.EQ:
            return candidate == self.value
        if self.operator == InvariantOperator.NE:
            return candidate != self.value
        if self.operator == InvariantOperator.EXISTS:
            return (self.argument in arguments) == bool(self.value)
        if (
            not isinstance(candidate, (int, float))
            or isinstance(candidate, bool)
            or not isinstance(self.value, (int, float))
            or isinstance(self.value, bool)
        ):
            return False
        if self.operator == InvariantOperator.LT:
            return candidate < self.value
        if self.operator == InvariantOperator.LTE:
            return candidate <= self.value
        if self.operator == InvariantOperator.GT:
            return candidate > self.value
        if self.operator == InvariantOperator.GTE:
            return candidate >= self.value
        return False


class AttackExpectation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tool: ToolId
    arguments: JsonObject
    match_arguments: JsonObject | None = None
    argument_predicates: tuple[ArgumentExpectationPredicate, ...] = ()

    def matches(self, arguments: JsonObject) -> bool:
        required_arguments = (
            self.match_arguments if self.match_arguments is not None else self.arguments
        )
        return all(
            arguments.get(argument) == expected for argument, expected in required_arguments.items()
        ) and all(predicate.matches(arguments) for predicate in self.argument_predicates)


class ToolCapability(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tool: ToolId
    exact_arguments: dict[str, JsonPrimitive] = Field(default_factory=dict)
    resource_arguments: tuple[str, ...] = ()
    max_calls: Annotated[int, Field(ge=1)] | None = None

    @field_validator("resource_arguments")
    @classmethod
    def normalize_resource_arguments(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted(set(value)))


class InvariantOperator(StrEnum):
    EQ = "eq"
    NE = "ne"
    LT = "lt"
    LTE = "lte"
    GT = "gt"
    GTE = "gte"
    EXISTS = "exists"


class StatePredicate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    predicate_id: str
    description: str
    json_pointer: str
    operator: InvariantOperator
    value: JsonPrimitive
    approval_can_waive: bool = False

    @field_validator("json_pointer")
    @classmethod
    def require_absolute_json_pointer(cls, value: str) -> str:
        if not value.startswith("/"):
            raise ValueError("json_pointer must start with '/'")
        return value


class UtilityRequirements(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    required_tools: tuple[ToolId, ...]
    final_state_predicates: tuple[StatePredicate, ...] = ()
    description: str


class ExecutionCase(BaseModel):
    """Target-neutral task, fixtures, safety contract, and utility contract."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    target_id: str
    objective: str
    resources: tuple[str, ...]
    attack: bool = False
    fixtures: JsonObject = Field(default_factory=dict)
    initial_state: JsonObject
    invariants: tuple[StatePredicate, ...]
    utility: UtilityRequirements
    capabilities: tuple[ToolCapability, ...]
    attack_expectation: AttackExpectation | None = None

    @field_validator("resources")
    @classmethod
    def normalize_resources(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(sorted(set(value)))
        if not normalized:
            raise ValueError("at least one resource is required")
        return normalized

    @model_validator(mode="after")
    def validate_security_contract(self) -> ExecutionCase:
        capability_tools = [capability.tool for capability in self.capabilities]
        if not capability_tools:
            raise ValueError("at least one tool capability is required")
        if len(capability_tools) != len(set(capability_tools)):
            raise ValueError("tool capabilities must have unique tool identifiers")
        unknown_utility_tools = set(self.utility.required_tools).difference(capability_tools)
        if unknown_utility_tools:
            raise ValueError(
                f"utility requires tools without capabilities: {unknown_utility_tools}"
            )
        if self.attack and self.attack_expectation is None:
            raise ValueError("attack cases require an attack_expectation")
        if not self.attack and self.attack_expectation is not None:
            raise ValueError("normal cases cannot declare an attack_expectation")
        if (
            self.attack_expectation is not None
            and self.attack_expectation.tool not in capability_tools
        ):
            raise ValueError("attack expectation tool has no capability")
        return self
