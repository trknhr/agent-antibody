from __future__ import annotations

from datetime import UTC, datetime

from agent_antibody.contracts import EventType, JsonValue, ToolName, TraceEvent


class TraceRecorder:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._events: list[TraceEvent] = []

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        return tuple(self._events)

    def record(
        self,
        event_type: EventType,
        *,
        request_id: str | None = None,
        tool: ToolName | None = None,
        payload: dict[str, JsonValue] | None = None,
    ) -> TraceEvent:
        event = TraceEvent(
            run_id=self.run_id,
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
