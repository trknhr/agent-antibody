from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import logging
import warnings
from collections.abc import AsyncGenerator
from typing import Protocol, cast

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types
from pydantic import BaseModel, PrivateAttr, TypeAdapter

from agent_antibody.capabilities import (
    AdkToolCapability,
    CapabilitySnapshot,
    canonical_input_schema,
    normalize_capability_text,
)
from agent_antibody.contracts import SignedApproval, ToolCallResult
from agent_antibody.core_types import JsonObject, JsonValue, ToolId
from agent_antibody.targets.base import TargetAdapter

_JSON_VALUE: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)
_CAPTURE_MODEL_NAME = "agent-antibody-adk-capability-capture"
_ADK_SCHEMA_WARNING = (
    r"\[EXPERIMENTAL\] feature FeatureName\.JSON_SCHEMA_FOR_FUNC_DECL is enabled\."
)

# A credential-free capture response intentionally has no provider token usage.
logging.getLogger("google_adk.google.adk.telemetry._metrics").setLevel(logging.ERROR)


class CapabilityExtractionError(ValueError):
    """Raised when ADK's real outbound capability declaration cannot be proven."""


class _CaptureConfig(Protocol):
    @property
    def tools(self) -> object: ...

    @property
    def system_instruction(self) -> object: ...


class ToolCaptureLlm(BaseLlm):
    """Capture exactly one outbound ADK request and return text without calling tools."""

    _captured_request: LlmRequest | None = PrivateAttr(default=None)

    @property
    def captured_request(self) -> LlmRequest:
        if self._captured_request is None:
            raise CapabilityExtractionError("ADK agent did not issue an LLM request")
        return self._captured_request

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse]:
        del stream
        if self._captured_request is not None:
            raise CapabilityExtractionError("ADK agent issued more than one capture request")
        self._captured_request = llm_request
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text="Capability capture complete.")],
            )
        )


def _canonical_instruction_value(value: object) -> JsonValue:
    if isinstance(value, str):
        return {"kind": "text", "value": normalize_capability_text(value)}
    if isinstance(value, BaseModel):
        dumped: object = value.model_dump(mode="json", exclude_none=True)
        return _canonical_instruction_value(dumped)
    if isinstance(value, dict):
        items = cast(dict[object, object], value).items()
        return {
            str(key): _canonical_instruction_value(item)
            for key, item in sorted(items, key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        sequence = cast(list[object] | tuple[object, ...], value)
        return [_canonical_instruction_value(item) for item in sequence]
    if value is None or isinstance(value, (int, float, bool)):
        return value
    raise CapabilityExtractionError(
        f"unsupported ADK system instruction value: {type(value).__name__}"
    )


def _instruction_digest(value: object) -> str:
    if value is None:
        raise CapabilityExtractionError("ADK outbound request has no system instruction")
    canonical = _JSON_VALUE.validate_python(_canonical_instruction_value(value))
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _input_schema(declaration: types.FunctionDeclaration) -> object:
    if declaration.parameters is not None and declaration.parameters_json_schema is not None:
        raise CapabilityExtractionError(
            f"tool {declaration.name!r} declares both parameters schema representations"
        )
    if declaration.parameters_json_schema is not None:
        return declaration.parameters_json_schema
    if declaration.parameters is not None:
        return declaration.parameters.model_dump(mode="json", exclude_none=True)
    raise CapabilityExtractionError(f"tool {declaration.name!r} has no input schema")


def _reject_unsupported_tool_fields(tool: types.Tool) -> None:
    supported = {"function_declarations"}
    populated = set(tool.model_dump(mode="json", exclude_none=True)).difference(supported)
    if populated:
        fields = ", ".join(sorted(populated))
        raise CapabilityExtractionError(f"unsupported ADK tool declaration fields: {fields}")


def _reject_unsupported_function_fields(declaration: types.FunctionDeclaration) -> None:
    supported = {"name", "description", "parameters", "parameters_json_schema"}
    populated = set(declaration.model_dump(mode="json", exclude_none=True)).difference(supported)
    if populated:
        fields = ", ".join(sorted(populated))
        raise CapabilityExtractionError(
            f"unsupported fields on tool {declaration.name!r}: {fields}"
        )


def capability_snapshot_from_request(
    llm_request: LlmRequest,
    *,
    agent_name: str,
    adk_version: str | None = None,
) -> CapabilitySnapshot:
    """Extract the declarations ADK actually placed in a model request."""

    tools: list[AdkToolCapability] = []
    config = cast(_CaptureConfig, llm_request.config)
    raw_tools = config.tools
    if raw_tools is None:
        request_tools: list[object] = []
    elif isinstance(raw_tools, list):
        request_tools = cast(list[object], raw_tools)
    else:
        raise CapabilityExtractionError("ADK outbound tools collection is unsupported")
    for raw_tool in request_tools:
        if not isinstance(raw_tool, types.Tool):
            raise CapabilityExtractionError(
                f"unsupported outbound ADK tool value: {type(raw_tool).__name__}"
            )
        tool = raw_tool
        _reject_unsupported_tool_fields(tool)
        declarations = tool.function_declarations or []
        if not declarations:
            raise CapabilityExtractionError("ADK tool entry contains no function declarations")
        for declaration in declarations:
            _reject_unsupported_function_fields(declaration)
            raw_name = declaration.name
            raw_description = declaration.description
            if raw_name is None or raw_description is None:
                raise CapabilityExtractionError(
                    "ADK function declaration is missing name or description"
                )
            try:
                tools.append(
                    AdkToolCapability(
                        name=raw_name,
                        description=raw_description,
                        input_schema=canonical_input_schema(_input_schema(declaration)),
                    )
                )
            except ValueError as error:
                raise CapabilityExtractionError(
                    f"invalid ADK declaration for tool {raw_name!r}: {error}"
                ) from error

    names = [tool.name for tool in tools]
    if len(names) != len(set(names)):
        raise CapabilityExtractionError("ADK outbound request contains duplicate tool names")
    registry_names = set(llm_request.tools_dict)
    if registry_names != set(names):
        missing = sorted(registry_names.difference(names))
        undeclared = sorted(set(names).difference(registry_names))
        raise CapabilityExtractionError(
            "ADK tool registry and outbound declarations differ "
            f"(missing declarations={missing}, undeclared functions={undeclared})"
        )

    try:
        version = adk_version or importlib.metadata.version("google-adk")
        return CapabilitySnapshot(
            adk_version=version,
            agent_name=agent_name,
            instruction_sha256=_instruction_digest(config.system_instruction),
            tools=tuple(tools),
        )
    except ValueError as error:
        raise CapabilityExtractionError(f"invalid ADK capability snapshot: {error}") from error


async def capture_adk_capabilities_async(
    agent: LlmAgent,
    *,
    timeout_seconds: float = 10.0,
) -> CapabilitySnapshot:
    """Run one credential-free ADK turn and capture its provider-facing declarations."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if agent.sub_agents:
        raise CapabilityExtractionError("multi-agent ADK trees are not supported by this connector")

    capture_model = ToolCaptureLlm(model=_CAPTURE_MODEL_NAME)
    capture_agent = agent.model_copy(update={"model": capture_model, "parent_agent": None})
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=_ADK_SCHEMA_WARNING)
            async with asyncio.timeout(timeout_seconds):
                async with InMemoryRunner(
                    agent=capture_agent,
                    app_name="agent-antibody-capability-capture",
                ) as runner:
                    session = await runner.session_service.create_session(
                        app_name=runner.app_name,
                        user_id="agent-antibody",
                    )
                    async for _event in runner.run_async(
                        user_id="agent-antibody",
                        session_id=session.id,
                        new_message=types.Content(
                            role="user",
                            parts=[
                                types.Part.from_text(text="Describe the available capabilities.")
                            ],
                        ),
                        run_config=RunConfig(max_llm_calls=1),
                    ):
                        pass
    except TimeoutError as error:
        raise CapabilityExtractionError(
            f"ADK capability capture timed out after {timeout_seconds:g}s"
        ) from error

    return capability_snapshot_from_request(
        capture_model.captured_request,
        agent_name=agent.name,
    )


def capture_adk_capabilities(
    agent: LlmAgent,
    *,
    timeout_seconds: float = 10.0,
) -> CapabilitySnapshot:
    """Synchronous entry point for CI capability capture."""

    return asyncio.run(capture_adk_capabilities_async(agent, timeout_seconds=timeout_seconds))


class _RejectingToolInvoker:
    """Prove that capability capture never needs to execute a target operation."""

    def call(
        self,
        tool: ToolId,
        arguments: JsonObject,
        *,
        approval: SignedApproval | None = None,
    ) -> ToolCallResult:
        del tool, arguments, approval
        raise CapabilityExtractionError("capability capture attempted to execute a target tool")


def capture_target_adk_capabilities(
    adapter: TargetAdapter,
    *,
    timeout_seconds: float = 10.0,
) -> CapabilitySnapshot:
    """Capture the declarations emitted by a registered Agent Antibody target.

    The target's production ADK wrapper constructs the real ``Agent`` and sends one
    request to the credential-free capture model. Any unexpected tool execution
    fails closed through ``_RejectingToolInvoker``.
    """

    normal_cases = adapter.normal_cases()
    if not normal_cases:
        raise CapabilityExtractionError("target declares no case for ADK capability capture")
    capture_model = ToolCaptureLlm(model=_CAPTURE_MODEL_NAME)
    target_agent = adapter.create_agent(
        model=capture_model,
        timeout_seconds=timeout_seconds,
        max_llm_calls=1,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_ADK_SCHEMA_WARNING)
        result = target_agent.run(normal_cases[0], _RejectingToolInvoker())
    if result.tool_results:
        raise CapabilityExtractionError("capability capture unexpectedly produced tool results")
    return capability_snapshot_from_request(
        capture_model.captured_request,
        agent_name=adapter.manifest.target_id,
    )
