from __future__ import annotations

from pathlib import Path
from typing import cast

import yaml

_ROOT = Path(__file__).resolve().parents[1]


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    mapping = cast(dict[object, object], value)
    assert all(isinstance(key, str) for key in mapping)
    return cast(dict[str, object], mapping)


def _sequence(value: object) -> list[object]:
    assert isinstance(value, list)
    return cast(list[object], value)


def _workflow(name: str) -> dict[str, object]:
    parsed: object = yaml.load(
        (_ROOT / ".github" / "workflows" / name).read_text(),
        Loader=yaml.BaseLoader,
    )
    return _mapping(parsed)


def _step(job: dict[str, object], name: str) -> dict[str, object]:
    return next(
        step
        for raw_step in _sequence(job["steps"])
        if (step := _mapping(raw_step)).get("name") == name
    )


def test_remediation_isolates_vertex_identity_from_github_write_job() -> None:
    workflow = _workflow("antibody-remediate.yml")
    jobs = _mapping(workflow["jobs"])
    evaluator = _mapping(jobs["evaluate_delta"])
    writer = _mapping(jobs["create-immunity-pr"])

    assert workflow["permissions"] == {}
    assert _mapping(evaluator["permissions"]) == {
        "actions": "read",
        "contents": "read",
        "id-token": "write",
    }
    assert _mapping(writer["permissions"]) == {
        "actions": "read",
        "contents": "write",
        "pull-requests": "write",
    }
    assert writer["needs"] == "evaluate_delta"
    assert _mapping(evaluator["environment"])["name"] == "production"


def test_vertex_job_uses_workflow_pinned_evaluator_and_data_only_handoff() -> None:
    jobs = _mapping(_workflow("antibody-remediate.yml")["jobs"])
    evaluator = _mapping(jobs["evaluate_delta"])
    writer = _mapping(jobs["create-immunity-pr"])
    auth = _step(evaluator, "Authenticate the Vertex-only evaluator")
    run_delta = _step(evaluator, "Run bounded Gemini delta evaluation")
    upload = _step(evaluator, "Upload trusted evaluation handoff")
    merge = _step(writer, "Merge only the evaluator delta result")

    assert auth["uses"] == ("google-github-actions/auth@7c6bc770dae815cd3e89ee6cdf493a5fab2cc093")
    auth_inputs = _mapping(auth["with"])
    assert auth_inputs["workload_identity_provider"] == ("${{ vars.GCP_EVALUATOR_WIF_PROVIDER }}")
    assert auth_inputs["service_account"] == "${{ vars.GCP_EVALUATOR_SERVICE_ACCOUNT }}"

    delta_script = cast(str, run_delta["run"])
    assert "-u ACTIONS_ID_TOKEN_REQUEST_TOKEN" in delta_script
    assert "immunity evaluate-delta" in delta_script
    assert "candidate-artifact/delta-evaluation" in delta_script
    assert _mapping(upload["with"])["name"] == "antibody-trusted-evaluation"

    merge_script = cast(str, merge["run"])
    assert merge["working-directory"] == "control-plane"
    assert "immunity merge-assessments" in merge_script
    assert "delta-evaluation/remediation.json" in merge_script


def test_candidate_workflow_never_receives_oidc_or_write_permissions() -> None:
    candidate = _workflow("antibody-candidate.yml")

    assert _mapping(candidate["permissions"]) == {"contents": "read"}
    text = (_ROOT / ".github" / "workflows" / "antibody-candidate.yml").read_text()
    assert "id-token: write" not in text
    assert "contents: write" not in text
    assert "pull-requests: write" not in text


def test_deploy_gate_runs_two_bounded_uncached_campaigns() -> None:
    jobs = _mapping(_workflow("deploy.yml")["jobs"])
    deploy = _mapping(jobs["deploy"])
    gate = _step(deploy, "Live Gemini release gate")
    script = cast(str, gate["run"])

    assert deploy["timeout-minutes"] == "60"
    assert gate["timeout-minutes"] == "32"
    assert "for attempt in 1 2" in script
    assert "Live Gemini release gate failed after two bounded campaigns" in script
    assert ".suite_metrics.confirmed_blocked == 10" in script


def test_pull_request_ci_defers_only_the_snapshot_projection() -> None:
    jobs = _mapping(_workflow("ci.yml")["jobs"])
    regression = _step(_mapping(jobs["test"]), "Execute generated antibody regression")
    script = cast(str, regression["run"])

    assert 'if [ "$GITHUB_EVENT_NAME" = "pull_request" ]' in script
    assert "--allow-stale-snapshot" in script
    assert "agent-antibody immunity verify --repository-root ." in script
