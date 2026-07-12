from __future__ import annotations

import os
import threading
import time

from pydantic import BaseModel, ConfigDict

from agent_antibody.live_pipeline import ATTACK_CASE_COUNT, LivePipelineReport, LiveSecurityPipeline
from agent_antibody.targets.registry import TARGET_IDS, get_target_adapter


class LiveDemoState(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool
    configured: bool
    authentication_required: bool
    busy: bool
    cached: bool
    model: str
    attack_case_count: int
    targets: tuple[str, ...]
    cached_targets: tuple[str, ...]


class LiveDemoBusyError(RuntimeError):
    pass


class LiveDemoService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cached_reports: dict[str, LivePipelineReport] = {}
        self._cached_at: dict[str, float] = {}

    @property
    def enabled(self) -> bool:
        return os.getenv("AGENT_ANTIBODY_LIVE_ENABLED", "false").lower() == "true"

    @property
    def model(self) -> str:
        return os.getenv("AGENT_ANTIBODY_MODEL", "gemini-3.5-flash")

    @property
    def api_token(self) -> str | None:
        token = os.getenv("AGENT_ANTIBODY_LIVE_API_TOKEN", "").strip()
        return token or None

    @property
    def cache_seconds(self) -> float:
        return float(os.getenv("AGENT_ANTIBODY_LIVE_CACHE_SECONDS", "600"))

    @property
    def attack_suite_attempts(self) -> int:
        return int(os.getenv("AGENT_ANTIBODY_ATTACK_SUITE_ATTEMPTS", "2"))

    @property
    def suite_concurrency(self) -> int:
        return int(os.getenv("AGENT_ANTIBODY_SUITE_CONCURRENCY", "3"))

    @property
    def protected_replay_attempts(self) -> int:
        return int(os.getenv("AGENT_ANTIBODY_PROTECTED_REPLAY_ATTEMPTS", "2"))

    def state(self) -> LiveDemoState:
        cached_targets = tuple(
            target_id for target_id in TARGET_IDS if self._cache_valid(target_id)
        )
        return LiveDemoState(
            enabled=self.enabled,
            configured=self.enabled and self.api_token is not None,
            authentication_required=True,
            busy=self._lock.locked(),
            cached=bool(cached_targets),
            model=self.model,
            attack_case_count=ATTACK_CASE_COUNT,
            targets=TARGET_IDS,
            cached_targets=cached_targets,
        )

    def run(
        self,
        *,
        target_id: str = "opsmate",
        cloud_trace_id: str | None = None,
    ) -> LivePipelineReport:
        if not self.enabled:
            raise PermissionError("live Gemini demo is disabled")
        get_target_adapter(target_id)
        if self._cache_valid(target_id):
            return self._cached_reports[target_id]
        if not self._lock.acquire(blocking=False):
            raise LiveDemoBusyError("another live security evaluation is already running")
        try:
            if self._cache_valid(target_id):
                return self._cached_reports[target_id]
            report = LiveSecurityPipeline(
                model=self.model,
                target_id=target_id,
                attack_suite_attempts=self.attack_suite_attempts,
                suite_concurrency=self.suite_concurrency,
                protected_replay_attempts=self.protected_replay_attempts,
            ).run(cloud_trace_id=cloud_trace_id)
            self._cached_reports[target_id] = report
            self._cached_at[target_id] = time.monotonic()
            return report
        finally:
            self._lock.release()

    def _cache_valid(self, target_id: str) -> bool:
        return (
            target_id in self._cached_reports
            and time.monotonic() - self._cached_at.get(target_id, 0.0) < self.cache_seconds
        )


live_demo_service = LiveDemoService()
