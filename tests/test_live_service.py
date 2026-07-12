from __future__ import annotations

from pytest import MonkeyPatch

import agent_antibody.live_service as live_service_module
from agent_antibody.live_service import LiveDemoService
from agent_antibody.portfolio_demo import TargetDemoReport, run_target_demo


def test_failed_live_report_is_not_cached(monkeypatch: MonkeyPatch) -> None:
    accepted = run_target_demo("supportmate")
    rejected = accepted.model_copy(update={"acceptance_passed": False})
    reports = [rejected, accepted]
    calls = 0

    class FakePipeline:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def run(self, *, cloud_trace_id: str | None = None) -> TargetDemoReport:
            nonlocal calls
            del cloud_trace_id
            calls += 1
            return reports.pop(0)

    monkeypatch.setenv("AGENT_ANTIBODY_LIVE_ENABLED", "true")
    monkeypatch.setattr(live_service_module, "LiveSecurityPipeline", FakePipeline)
    service = LiveDemoService()

    first = service.run(target_id="supportmate")
    assert not first.acceptance_passed
    assert not service.state().cached

    second = service.run(target_id="supportmate")
    assert second.acceptance_passed
    assert service.state().cached

    assert service.run(target_id="supportmate") is second
    assert calls == 2
