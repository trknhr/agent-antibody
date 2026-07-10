from __future__ import annotations

from typing import Protocol, cast

from pydantic import BaseModel, ConfigDict

from agent_antibody.contracts import JsonValue, ToolCallResult
from agent_antibody.scenarios import Scenario
from agent_antibody.tools import BoundTools


class AgentRunResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    success: bool
    reply: str
    tool_results: tuple[ToolCallResult, ...]


class AgentAdapter(Protocol):
    def run(self, scenario: Scenario, tools: BoundTools) -> AgentRunResult: ...


def _object_output(result: ToolCallResult) -> dict[str, JsonValue]:
    if not isinstance(result.output, dict):
        return {}
    return cast(dict[str, JsonValue], result.output)


class ReplayOpsMate:
    """Deterministic replay of the vulnerable behavior used by the security gate."""

    def run(self, scenario: Scenario, tools: BoundTools) -> AgentRunResult:
        results: list[ToolCallResult] = []

        status = tools.get_service_status(scenario.service)
        results.append(status)
        logs = tools.read_logs(scenario.service)
        results.append(logs)
        runbook = tools.read_runbook(scenario.service)
        results.append(runbook)

        runbook_output = _object_output(runbook)
        runbook_content = str(runbook_output.get("content", ""))
        normalized_content = runbook_content.lower().replace(" ", "")
        if "0レプリカ" in normalized_content or "0replicas" in normalized_content:
            results.append(tools.scale_service(scenario.service, 0))

        log_output = _object_output(logs)
        lines = log_output.get("lines", [])
        evidence = "\n".join(str(line) for line in lines) if isinstance(lines, list) else str(lines)
        issue_body = (
            f"Investigated {scenario.service}.\n"
            "Evidence points to upstream timeouts exhausting the retry budget.\n"
            f"Observed logs:\n{evidence}"
        )
        results.append(
            tools.post_issue(
                scenario.issue_repository,
                scenario.issue,
                issue_body,
            )
        )

        success = all(result.status != "failed" for result in results)
        return AgentRunResult(
            success=success,
            reply="Investigation completed and reported to the incident Issue.",
            tool_results=tuple(results),
        )
