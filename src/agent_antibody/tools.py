from __future__ import annotations

from agent_antibody.contracts import (
    JsonValue,
    SignedApproval,
    SignedTaskContract,
    ToolCallResult,
    ToolName,
)
from agent_antibody.gateway import PolicyGateway


class BoundTools:
    """The only operations surface exposed to an agent."""

    def __init__(
        self,
        *,
        caller_id: str,
        signed_contract: SignedTaskContract,
        gateway: PolicyGateway,
    ) -> None:
        self._caller_id = caller_id
        self._signed_contract = signed_contract
        self._gateway = gateway

    def get_service_status(self, service: str) -> ToolCallResult:
        return self._call(ToolName.GET_SERVICE_STATUS, {"service": service})

    def read_logs(self, service: str) -> ToolCallResult:
        return self._call(ToolName.READ_LOGS, {"service": service})

    def read_runbook(self, service: str) -> ToolCallResult:
        return self._call(ToolName.READ_RUNBOOK, {"service": service})

    def scale_service(
        self,
        service: str,
        replicas: int,
        *,
        approval: SignedApproval | None = None,
    ) -> ToolCallResult:
        return self._call(
            ToolName.SCALE_SERVICE,
            {"service": service, "replicas": replicas},
            approval=approval,
        )

    def post_issue(
        self,
        repository: str,
        issue: str,
        body: str,
    ) -> ToolCallResult:
        return self._call(
            ToolName.POST_ISSUE,
            {"repository": repository, "issue": issue, "body": body},
        )

    def _call(
        self,
        tool: ToolName,
        arguments: dict[str, JsonValue],
        *,
        approval: SignedApproval | None = None,
    ) -> ToolCallResult:
        return self._gateway.call(
            caller_id=self._caller_id,
            signed_contract=self._signed_contract,
            tool=tool,
            arguments=arguments,
            approval=approval,
        )
