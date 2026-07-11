"""Deterministic model fixtures for the isolated Agent Antibody harness.

The fixture implements the same ADK ``BaseLlm`` protocol as Gemini, but has no
network access and never reads untrusted documents itself.  Target adapters use
it to drive their *production* ADK agent through a known adversarial tool-call
sequence during candidate and regression evaluation.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import PrivateAttr

from agent_antibody.core_types import JsonObject

HARNESS_ID = "adk-scripted-v1"

# ADK emits one warning per tool turn when an offline fixture intentionally has
# no provider token-usage record.  It is expected for this harness and would
# otherwise bury the policy result in CI logs.
logging.getLogger("google_adk.google.adk.telemetry._metrics").setLevel(logging.ERROR)


class ScriptedHarnessLlm(BaseLlm):
    """A bounded ADK model that returns a predeclared sequence of responses."""

    responses: list[LlmResponse]
    _next: int = PrivateAttr(default=0)

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse]:
        del llm_request, stream
        if self._next >= len(self.responses):
            raise RuntimeError("scripted harness model response list exhausted")
        response = self.responses[self._next]
        self._next += 1
        yield response


def tool_call(name: str, arguments: JsonObject) -> LlmResponse:
    """Create one ADK function-call response without serializing prompt content."""

    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part.from_function_call(name=name, args=dict(arguments))],
        )
    )


def final_text(value: str) -> LlmResponse:
    """Create the terminal model response for a scripted harness run."""

    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part.from_text(text=value)])
    )
