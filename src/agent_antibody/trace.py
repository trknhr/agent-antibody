from __future__ import annotations

from datetime import UTC, datetime

from agent_antibody.contracts import EventType, JsonValue, TraceEvent
from agent_antibody.core_types import ToolId


class TraceRecorder:
    def __init__(
        self,
        run_id: str,
        *,
        target_id: str | None = None,
        case_id: str | None = None,
    ) -> None:
        self.run_id = run_id
        self.target_id = target_id
        self.case_id = case_id
        self._events: list[TraceEvent] = []

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        return tuple(self._events)

    def record(
        self,
        event_type: EventType,
        *,
        request_id: str | None = None,
        tool: ToolId | None = None,
        payload: dict[str, JsonValue] | None = None,
    ) -> TraceEvent:
        event = TraceEvent(
            run_id=self.run_id,
            target_id=self.target_id,
            case_id=self.case_id,
            sequence=len(self._events) + 1,
            event_type=event_type,
            request_id=request_id,
            tool=tool,
            payload=payload or {},
            timestamp=datetime.now(UTC),
        )
        self._events.append(event)
        return event

    def complete(self, *, success: bool) -> None:
        self.record(EventType.RUN_COMPLETED, payload={"success": success})
