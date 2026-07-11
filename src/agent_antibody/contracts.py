from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent_antibody.core_types import JsonPrimitive as JsonPrimitive
from agent_antibody.core_types import JsonValue, ToolCapability, ToolId


class ToolName(StrEnum):
    GET_SERVICE_STATUS = "get_service_status"
    READ_LOGS = "read_logs"
    READ_RUNBOOK = "read_runbook"
    POST_ISSUE = "post_issue"
    SCALE_SERVICE = "scale_service"


class SourceKind(StrEnum):
    RUNBOOK_CONTENT = "runbook_content"
    ISSUE_BODY = "issue_body"
    LOG_CONTENT = "log_content"


class PolicyMode(StrEnum):
    AUDIT = "audit"
    ENFORCE = "enforce"


class EventType(StrEnum):
    TOOL_REQUESTED = "tool.requested"
    POLICY_DECISION = "policy.decision"
    TOOL_EXECUTED = "tool.executed"
    EXECUTION_OUTCOME = "execution.outcome"
    STATE_CHANGED = "state.changed"
    RUN_COMPLETED = "run.completed"


class DecisionEffect(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    AUDIT_ALLOW = "audit_allow"


class PostIssueConstraint(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    repository: str
    issue: str
    max_calls: Annotated[int, Field(ge=1, le=10)] = 1
    body_policy: Literal["redacted_incident_summary"] = "redacted_incident_summary"


class ToolConstraints(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    post_issue: PostIssueConstraint | None = None


class TaskContract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    caller_id: str
    objective: str
    resources: tuple[str, ...]
    allowed_tools: tuple[ToolId, ...]
    capabilities: tuple[ToolCapability, ...] = ()
    tool_constraints: ToolConstraints = Field(default_factory=ToolConstraints)
    issued_at: datetime
    expires_at: datetime

    @field_validator("resources")
    @classmethod
    def normalize_resources(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(sorted(set(value)))
        if not normalized:
            raise ValueError("at least one resource is required")
        return normalized

    @field_validator("allowed_tools")
    @classmethod
    def normalize_tools(cls, value: tuple[ToolId, ...]) -> tuple[ToolId, ...]:
        normalized = tuple(sorted(set(value), key=str))
        if not normalized:
            raise ValueError("at least one allowed tool is required")
        return normalized

    @field_validator("issued_at", "expires_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def require_valid_window(self) -> TaskContract:
        if self.expires_at <= self.issued_at:
            raise ValueError("expires_at must be later than issued_at")
        capability_tools = [capability.tool for capability in self.capabilities]
        if len(capability_tools) != len(set(capability_tools)):
            raise ValueError("tool capabilities must have unique tool identifiers")
        if unknown_tools := set(capability_tools).difference(self.allowed_tools):
            raise ValueError(f"capabilities reference tools that are not allowed: {unknown_tools}")
        return self


class SignedTaskContract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    contract: TaskContract
    signature: str


class ApprovalGrant(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    approval_id: str
    contract_digest: str
    tool: ToolId
    arguments_digest: str
    max_uses: Literal[1] = 1
    expires_at: datetime

    @field_validator("expires_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("expires_at must be timezone-aware")
        return value.astimezone(UTC)


class SignedApproval(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    grant: ApprovalGrant
    signature: str


class GetServiceStatusArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    service: str


class ReadLogsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    service: str


class ReadRunbookArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    service: str


class ScaleServiceArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    service: str
    replicas: Annotated[int, Field(ge=0, le=100)]


class PostIssueArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repository: str
    issue: str
    body: Annotated[str, Field(min_length=1, max_length=10_000)]


type ToolArgsModel = (
    GetServiceStatusArgs | ReadLogsArgs | ReadRunbookArgs | ScaleServiceArgs | PostIssueArgs
)


class ToolCallResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str
    status: Literal["executed", "denied", "failed"]
    output: JsonValue = None
    reasons: tuple[str, ...] = ()


class TraceEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    target_id: str | None = None
    case_id: str | None = None
    sequence: Annotated[int, Field(ge=1)]
    event_type: EventType
    request_id: str | None = None
    tool: ToolId | None = None
    payload: dict[str, JsonValue] = Field(default_factory=dict)
    timestamp: datetime

    @field_validator("timestamp")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return value.astimezone(UTC)
