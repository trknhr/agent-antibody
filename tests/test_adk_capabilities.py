from __future__ import annotations

import importlib.metadata
from collections.abc import AsyncGenerator

import pytest
from google.adk import Agent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool
from google.genai import types
from pydantic import PrivateAttr

from agent_antibody.adk_capabilities import (
    CapabilityExtractionError,
    capability_snapshot_from_request,
    capture_adk_capabilities,
    capture_target_adk_capabilities,
)
from agent_antibody.capabilities import (
    AdkToolCapability,
    AttackSurfaceChangeKind,
    CapabilityComparisonError,
    CapabilitySnapshot,
    diff_capability_snapshots,
)
from agent_antibody.core_types import JsonObject
from agent_antibody.targets.registry import get_target_adapter

_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64


class NeverCalledLlm(BaseLlm):
    _calls: int = PrivateAttr(default=0)

    @property
    def calls(self) -> int:
        return self._calls

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse]:
        del llm_request, stream
        self._calls += 1
        raise AssertionError("original ADK model must not be called during capability capture")
        yield  # pragma: no cover


def read_customer(customer_id: str) -> dict[str, str]:
    """Read a customer record.

    Args:
        customer_id: Stable customer identifier.
    """

    return {"customer_id": customer_id}


def issue_refund(customer_id: str, amount_cents: int) -> dict[str, int | str]:
    """Issue a refund to a customer.

    Args:
        customer_id: Stable customer identifier.
        amount_cents: Refund amount in cents.
    """

    return {"customer_id": customer_id, "amount_cents": amount_cents}


def _tool(
    name: str,
    *,
    description: str = "Tool description.",
    schema: JsonObject | None = None,
) -> AdkToolCapability:
    return AdkToolCapability(
        name=name,
        description=description,
        input_schema=(
            schema
            if schema is not None
            else {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            }
        ),
    )


def _snapshot(
    *tools: AdkToolCapability,
    instruction_sha256: str = _DIGEST_A,
    adk_version: str = "2.4.0",
) -> CapabilitySnapshot:
    return CapabilitySnapshot(
        adk_version=adk_version,
        agent_name="support_mate",
        instruction_sha256=instruction_sha256,
        tools=tools,
    )


def _request_with_declarations(
    declarations: list[types.FunctionDeclaration],
    *,
    registry: dict[str, BaseTool] | None = None,
    tool: types.Tool | None = None,
) -> LlmRequest:
    return LlmRequest(
        model="capture",
        config=types.GenerateContentConfig(
            system_instruction="Inspect capabilities.",
            tools=[tool or types.Tool(function_declarations=declarations)],
        ),
        tools_dict=registry or {},
    )


def test_capture_observes_actual_adk_request_without_calling_original_model() -> None:
    original_model = NeverCalledLlm(model="must-not-run")
    agent = Agent(
        name="support_mate",
        model=original_model,
        instruction="Handle support requests safely.",
        tools=[issue_refund, read_customer],
    )

    snapshot = capture_adk_capabilities(agent)

    assert original_model.calls == 0
    assert snapshot.connector == "google-adk"
    assert snapshot.adk_version == importlib.metadata.version("google-adk")
    assert snapshot.agent_name == "support_mate"
    assert len(snapshot.instruction_sha256) == 64
    assert tuple(tool.name for tool in snapshot.tools) == ("issue_refund", "read_customer")
    assert snapshot.tool("issue_refund").input_schema["required"] == [
        "customer_id",
        "amount_cents",
    ]
    assert snapshot.tool("issue_refund").description.startswith("Issue a refund")
    assert "risk" not in snapshot.model_dump_json()


def test_capture_uses_the_registered_supportmate_adk_surface() -> None:
    snapshot = capture_target_adk_capabilities(get_target_adapter("supportmate"))

    assert snapshot.agent_name == "supportmate"
    assert tuple(tool.name for tool in snapshot.tools) == (
        "get_order",
        "grant_store_credit",
        "issue_refund",
        "read_attachment_text",
        "read_refund_policy",
        "read_ticket",
        "reply_to_customer",
    )


def test_capture_canonicalizes_equivalent_instruction_and_declarations() -> None:
    first = Agent(
        name="support_mate",
        model="unused-model",
        instruction="  Handle safely.\r\nReturn a result.   ",
        tools=[read_customer],
    )
    second = Agent(
        name="support_mate",
        model="unused-model",
        instruction="Handle safely.\nReturn a result.",
        tools=[read_customer],
    )

    first_snapshot = capture_adk_capabilities(first)
    second_snapshot = capture_adk_capabilities(second)

    assert first_snapshot == second_snapshot
    assert first_snapshot.digest() == second_snapshot.digest()


def test_diff_detects_all_supported_changes_and_requires_attacks() -> None:
    base = _snapshot(
        _tool("changed", description="Before."),
        _tool("removed"),
        _tool("unchanged"),
    )
    head = _snapshot(
        _tool("added"),
        _tool(
            "changed",
            description="After.",
            schema={
                "required": ["value"],
                "properties": {"value": {"type": "integer"}},
                "type": "object",
            },
        ),
        _tool("unchanged"),
        instruction_sha256=_DIGEST_B,
    )

    delta = diff_capability_snapshots(base, head)

    assert tuple(change.kind for change in delta.changes) == (
        AttackSurfaceChangeKind.ADDED_TOOL,
        AttackSurfaceChangeKind.INPUT_SCHEMA_CHANGED,
        AttackSurfaceChangeKind.DESCRIPTION_CHANGED,
        AttackSurfaceChangeKind.REMOVED_TOOL,
        AttackSurfaceChangeKind.AGENT_INSTRUCTION_CHANGED,
    )
    assert delta.attack_required_tools == ("added", "changed")
    assert all(
        change.attack_required
        for change in delta.changes
        if change.kind != AttackSurfaceChangeKind.REMOVED_TOOL
    )
    assert not next(
        change for change in delta.changes if change.kind == AttackSurfaceChangeKind.REMOVED_TOOL
    ).attack_required


def test_canonical_schema_and_description_do_not_create_false_delta() -> None:
    base = _snapshot(
        _tool(
            "read_customer",
            description="  Read customer.  \r\n",
            schema={
                "type": "object",
                "properties": {"customer_id": {"type": "string", "title": "Customer"}},
                "required": ["customer_id"],
            },
        )
    )
    head = _snapshot(
        _tool(
            "read_customer",
            description="Read customer.",
            schema={
                "required": ["customer_id"],
                "properties": {"customer_id": {"title": "Customer", "type": "string"}},
                "type": "object",
            },
        )
    )

    delta = diff_capability_snapshots(base, head)

    assert not delta.changed
    assert delta.changes == ()
    assert delta.attack_required_tools == ()


def test_diff_fails_closed_when_adk_versions_differ() -> None:
    with pytest.raises(CapabilityComparisonError, match="ADK version changed"):
        diff_capability_snapshots(
            _snapshot(_tool("read_customer"), adk_version="2.4.0"),
            _snapshot(_tool("read_customer"), adk_version="2.5.0"),
        )


def test_extraction_rejects_duplicate_declarations() -> None:
    declaration = types.FunctionDeclaration(
        name="duplicate",
        description="Duplicate tool.",
        parameters_json_schema={"type": "object", "properties": {}},
    )

    with pytest.raises(CapabilityExtractionError, match="duplicate tool names"):
        capability_snapshot_from_request(
            _request_with_declarations([declaration, declaration]),
            agent_name="support_mate",
            adk_version="2.4.0",
        )


def test_extraction_rejects_empty_and_unsupported_tool_surfaces() -> None:
    empty_request = LlmRequest(
        model="capture",
        config=types.GenerateContentConfig(system_instruction="Inspect capabilities.", tools=[]),
        tools_dict={},
    )
    with pytest.raises(CapabilityExtractionError, match="no function tools"):
        capability_snapshot_from_request(
            empty_request,
            agent_name="support_mate",
            adk_version="2.4.0",
        )

    unsupported = types.Tool(google_search=types.GoogleSearch())
    with pytest.raises(CapabilityExtractionError, match="unsupported ADK tool declaration"):
        capability_snapshot_from_request(
            _request_with_declarations([], tool=unsupported),
            agent_name="support_mate",
            adk_version="2.4.0",
        )


def test_extraction_rejects_missing_schema_and_registry_mismatch() -> None:
    missing_schema = types.FunctionDeclaration(
        name="missing_schema",
        description="Missing input schema.",
    )
    with pytest.raises(CapabilityExtractionError, match="has no input schema"):
        capability_snapshot_from_request(
            _request_with_declarations([missing_schema]),
            agent_name="support_mate",
            adk_version="2.4.0",
        )

    declaration = types.FunctionDeclaration(
        name="declared_only",
        description="Declared but not registered.",
        parameters_json_schema={"type": "object", "properties": {}},
    )
    with pytest.raises(
        CapabilityExtractionError, match="registry and outbound declarations differ"
    ):
        capability_snapshot_from_request(
            _request_with_declarations([declaration]),
            agent_name="support_mate",
            adk_version="2.4.0",
        )


def test_extraction_accepts_matching_real_registry() -> None:
    declaration = types.FunctionDeclaration(
        name="read_customer",
        description="Read customer.",
        parameters_json_schema={
            "type": "object",
            "properties": {"customer_id": {"type": "string"}},
            "required": ["customer_id"],
        },
    )
    request = _request_with_declarations(
        [declaration],
        registry={"read_customer": FunctionTool(read_customer)},
    )

    snapshot = capability_snapshot_from_request(
        request,
        agent_name="support_mate",
        adk_version="2.4.0",
    )

    assert tuple(tool.name for tool in snapshot.tools) == ("read_customer",)
