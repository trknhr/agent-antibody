from __future__ import annotations

from typing import cast

from pydantic import BaseModel, ConfigDict

from agent_antibody.contracts import JsonValue, ToolName


class IssueRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    repository: str
    issue: str
    body: str


class SimulatorSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    replicas: dict[str, int]
    issues: tuple[IssueRecord, ...]


class OpsSimulator:
    def __init__(
        self,
        *,
        replicas: dict[str, int],
        runbooks: dict[str, str],
        logs: dict[str, tuple[str, ...]],
    ) -> None:
        self._replicas = dict(replicas)
        self._runbooks = dict(runbooks)
        self._logs = dict(logs)
        self._issues: list[IssueRecord] = []

    def snapshot(self) -> SimulatorSnapshot:
        return SimulatorSnapshot(replicas=dict(self._replicas), issues=tuple(self._issues))

    def execute(self, tool: ToolName, arguments: dict[str, JsonValue]) -> JsonValue:
        if tool == ToolName.GET_SERVICE_STATUS:
            service = cast(str, arguments["service"])
            return {"service": service, "replicas": self._replicas[service], "status": "degraded"}
        if tool == ToolName.READ_LOGS:
            service = cast(str, arguments["service"])
            return {
                "service": service,
                "source_id": f"logs:{service}",
                "source_kind": "log_content",
                "trust": "untrusted",
                "lines": list(self._logs[service]),
            }
        if tool == ToolName.READ_RUNBOOK:
            service = cast(str, arguments["service"])
            return {
                "service": service,
                "source_id": f"runbook:{service}",
                "source_kind": "runbook_content",
                "trust": "untrusted",
                "content": self._runbooks[service],
            }
        if tool == ToolName.SCALE_SERVICE:
            service = cast(str, arguments["service"])
            replicas = cast(int, arguments["replicas"])
            before = self._replicas[service]
            self._replicas[service] = replicas
            return {"service": service, "before": before, "after": replicas}
        if tool == ToolName.POST_ISSUE:
            record = IssueRecord(
                repository=cast(str, arguments["repository"]),
                issue=cast(str, arguments["issue"]),
                body=cast(str, arguments["body"]),
            )
            self._issues.append(record)
            return {"repository": record.repository, "issue": record.issue, "created": True}
        raise ValueError(f"unsupported tool: {tool}")
