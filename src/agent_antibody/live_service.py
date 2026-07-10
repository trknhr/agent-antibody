from __future__ import annotations

import os
import threading
import time

from pydantic import BaseModel, ConfigDict

from agent_antibody.live_pipeline import LivePipelineReport, LiveSecurityPipeline


class LiveDemoState(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool
    busy: bool
    cached: bool
    model: str


class LiveDemoBusyError(RuntimeError):
    pass


class LiveDemoService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cached_report: LivePipelineReport | None = None
        self._cached_at = 0.0

    @property
    def enabled(self) -> bool:
        return os.getenv("AGENT_ANTIBODY_LIVE_ENABLED", "false").lower() == "true"

    @property
    def model(self) -> str:
        return os.getenv("AGENT_ANTIBODY_MODEL", "gemini-3.5-flash")

    @property
    def cache_seconds(self) -> float:
        return float(os.getenv("AGENT_ANTIBODY_LIVE_CACHE_SECONDS", "600"))

    @property
    def attack_suite_attempts(self) -> int:
        return int(os.getenv("AGENT_ANTIBODY_ATTACK_SUITE_ATTEMPTS", "3"))

    def state(self) -> LiveDemoState:
        return LiveDemoState(
            enabled=self.enabled,
            busy=self._lock.locked(),
            cached=self._cache_valid(),
            model=self.model,
        )

    def run(self, *, cloud_trace_id: str | None = None) -> LivePipelineReport:
        if not self.enabled:
            raise PermissionError("live Gemini demo is disabled")
        if self._cache_valid() and self._cached_report is not None:
            return self._cached_report
        if not self._lock.acquire(blocking=False):
            raise LiveDemoBusyError("another live security evaluation is already running")
        try:
            if self._cache_valid() and self._cached_report is not None:
                return self._cached_report
            report = LiveSecurityPipeline(
                model=self.model,
                attack_suite_attempts=self.attack_suite_attempts,
            ).run(cloud_trace_id=cloud_trace_id)
            self._cached_report = report
            self._cached_at = time.monotonic()
            return report
        finally:
            self._lock.release()

    def _cache_valid(self) -> bool:
        return (
            self._cached_report is not None
            and time.monotonic() - self._cached_at < self.cache_seconds
        )


live_demo_service = LiveDemoService()
