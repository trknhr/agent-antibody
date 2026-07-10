from typing import cast

from fastapi.testclient import TestClient
from httpx import Response
from pytest import MonkeyPatch

from agent_antibody import api
from agent_antibody.api import app

client = TestClient(app)


def test_healthz() -> None:
    response = cast(
        Response,
        client.get("/healthz"),  # pyright: ignore[reportUnknownMemberType]
    )
    api_response = cast(
        Response,
        client.get("/api/healthz"),  # pyright: ignore[reportUnknownMemberType]
    )

    assert response.json() == {"status": "ok"}
    assert api_response.json() == {"status": "ok"}


def test_demo_endpoint() -> None:
    response = cast(
        Response,
        client.get("/api/demo"),  # pyright: ignore[reportUnknownMemberType]
    )

    assert response.status_code == 200
    payload = cast(dict[str, object], response.json())
    vulnerable = cast(dict[str, object], payload["vulnerable"])
    protected = cast(dict[str, object], payload["protected"])
    normal = cast(dict[str, object], payload["normal"])
    assert cast(dict[str, object], vulnerable["oracle"])["status"] == "INFECTED"
    assert cast(dict[str, object], protected["oracle"])["status"] == "IMMUNE"
    assert cast(dict[str, object], normal["oracle"])["status"] == "HEALTHY"


def test_index() -> None:
    response = cast(
        Response,
        client.get("/"),  # pyright: ignore[reportUnknownMemberType]
    )

    assert response.status_code == 200
    assert "Agent Antibody" in response.text


def test_live_endpoint_is_disabled_and_requires_json_client_header() -> None:
    missing_content_type = cast(
        Response,
        client.post("/api/live"),  # pyright: ignore[reportUnknownMemberType]
    )
    disabled = cast(
        Response,
        client.post(  # pyright: ignore[reportUnknownMemberType]
            "/api/live",
            json={},
            headers={"X-Agent-Antibody-Client": "live-demo"},
        ),
    )

    assert missing_content_type.status_code == 415
    assert disabled.status_code == 503


def test_live_endpoint_sanitizes_unexpected_runtime_errors(monkeypatch: MonkeyPatch) -> None:
    def fail_run(*, cloud_trace_id: str | None = None) -> object:
        raise RuntimeError("provider request failed with internal details")

    monkeypatch.setattr(api.live_demo_service, "run", fail_run)
    response = cast(
        Response,
        client.post(  # pyright: ignore[reportUnknownMemberType]
            "/api/live",
            json={},
            headers={"X-Agent-Antibody-Client": "live-demo"},
        ),
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "live Gemini evaluation failed"}
