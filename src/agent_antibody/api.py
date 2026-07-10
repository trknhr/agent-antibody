from __future__ import annotations

from importlib.resources import files

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse

from agent_antibody.demo import DemoReport, run_demo
from agent_antibody.live_pipeline import LivePipelineError, LivePipelineReport
from agent_antibody.live_service import LiveDemoBusyError, LiveDemoState, live_demo_service

app = FastAPI(title="Agent Antibody", version="0.1.0")


@app.get("/healthz")
@app.get("/api/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/demo")
def demo() -> DemoReport:
    return run_demo()


@app.get("/api/live/status")
def live_status() -> LiveDemoState:
    return live_demo_service.state()


@app.post("/api/live")
def live_demo(
    request: Request,
    x_agent_antibody_client: str | None = Header(default=None),
) -> LivePipelineReport:
    if request.headers.get("content-type", "").split(";", maxsplit=1)[0] != "application/json":
        raise HTTPException(status_code=415, detail="application/json is required")
    if x_agent_antibody_client != "live-demo":
        raise HTTPException(status_code=403, detail="live demo client header is required")
    trace_context = request.headers.get("x-cloud-trace-context")
    trace_id = trace_context.split("/", maxsplit=1)[0] if trace_context else None
    try:
        return live_demo_service.run(cloud_trace_id=trace_id)
    except PermissionError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except LiveDemoBusyError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except LivePipelineError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail="live Gemini evaluation failed") from error


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = files("agent_antibody").joinpath("static/index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)
