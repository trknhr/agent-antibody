from __future__ import annotations

from collections.abc import AsyncGenerator

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from pydantic import PrivateAttr


class ScriptedLlm(BaseLlm):
    responses: list[LlmResponse]
    _next: int = PrivateAttr(default=0)

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse]:
        del llm_request
        del stream
        if self._next >= len(self.responses):
            raise AssertionError("ScriptedLlm response list exhausted")
        response = self.responses[self._next]
        self._next += 1
        yield response
