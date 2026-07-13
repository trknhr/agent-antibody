from __future__ import annotations

import subprocess
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


def _steps(job: dict[str, object], name: str) -> tuple[dict[str, object], ...]:
    return tuple(
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


def test_writer_runs_candidate_apply_without_credentials_then_rechecks_out() -> None:
    jobs = _mapping(_workflow("antibody-remediate.yml")["jobs"])
    writer = _mapping(jobs["create-immunity-pr"])
    apply = _step(writer, "Apply and verify immunity in the uncredentialed candidate tree")
    stage = _step(writer, "Stage candidate-generated immutable memory")
    recheckout = _step(writer, "Re-check out candidate tree before GitHub write")
    restore = _step(writer, "Restore staged immutable memory only")
    create = _step(writer, "Create stacked immunity pull request")
    link = _step(writer, "Link the immutable snapshot to the generated pull request")

    assert apply["working-directory"] == "candidate"
    apply_env = _mapping(apply["env"])
    assert "GH_TOKEN" not in apply_env
    apply_script = cast(str, apply["run"])
    assert "uv run --locked agent-antibody immunity apply" in apply_script
    assert "-u GH_TOKEN" in apply_script
    assert "-u GITHUB_TOKEN" in apply_script
    assert "-u GITHUB_ENV" in apply_script
    assert (
        "Candidate execution modified source files outside immutable immunity storage."
        in apply_script
    )

    assert "prepared-immunity" in cast(str, stage["run"])
    assert recheckout["uses"] == ("actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0")
    assert _mapping(recheckout["with"])["clean"] == "true"
    assert "cp -R" in cast(str, restore["run"])
    assert "git diff --cached --name-status" in cast(str, create["run"])
    assert "jq --arg pr_url" in cast(str, link["run"])
    assert "agent-antibody immunity apply" not in cast(str, link["run"])


def test_only_the_allowlisted_pr_author_can_trigger_antibody_workflows() -> None:
    candidate_jobs = _mapping(_workflow("antibody-candidate.yml")["jobs"])
    candidate = _mapping(candidate_jobs["evaluate"])
    candidate_condition = cast(str, candidate["if"])

    assert "vars.AGENT_ANTIBODY_ALLOWED_PR_AUTHOR" in candidate_condition
    assert "github.event.pull_request.user.login" in candidate_condition
    assert (
        "github.event.pull_request.head.repo.full_name == github.repository" in candidate_condition
    )
    assert "github.actor" in candidate_condition

    remediation_jobs = _mapping(_workflow("antibody-remediate.yml")["jobs"])
    evaluator = _mapping(remediation_jobs["evaluate_delta"])
    writer = _mapping(remediation_jobs["create-immunity-pr"])
    sources = (
        *_steps(evaluator, "Resolve and bind the source pull request"),
        *_steps(writer, "Resolve and bind the source pull request"),
    )

    assert len(sources) == 2
    for source in sources:
        assert _mapping(source["env"])["ALLOWED_PR_AUTHOR"] == (
            "${{ vars.AGENT_ANTIBODY_ALLOWED_PR_AUTHOR }}"
        )
        script = cast(str, source["run"])
        assert "PR_AUTHOR=" in script
        assert '[ -z "$ALLOWED_PR_AUTHOR" ]' in script
        assert "Source pull request author is not authorized" in script


def test_candidate_revalidates_only_a_verified_stacked_memory_update() -> None:
    candidate_jobs = _mapping(_workflow("antibody-candidate.yml")["jobs"])
    candidate = _mapping(candidate_jobs["evaluate"])
    provenance = _step(candidate, "Validate merged stacked immunity provenance")
    provenance_env = _mapping(provenance["env"])
    provenance_script = cast(str, provenance["run"])
    capture = _step(candidate, "Capture Base and Head ADK capability surfaces")
    capture_env = _mapping(capture["env"])
    capture_script = cast(str, capture["run"])

    assert provenance_env["GH_TOKEN"] == "${{ github.token }}"
    assert "use_candidate_memory=false" in provenance_script
    assert "git -C candidate diff --name-only --no-renames" in provenance_script
    assert '.lifecycle.state == "verified_pending_review"' in provenance_script
    assert '.lifecycle.evaluation_status == "BYPASS_CONFIRMED"' in provenance_script
    assert ".merged == true" in provenance_script
    assert ".head.ref == $branch" in provenance_script
    assert ".base.ref == $base_ref" in provenance_script
    assert 'merge-base --is-ancestor "$SOURCE_REVISION" "$HEAD_SHA"' in provenance_script
    assert 'merge-base --is-ancestor "$MERGE_COMMIT_SHA" "$HEAD_SHA"' in provenance_script
    assert 'cmp -s "$REMEDIATION_FILES" "$CANDIDATE_FILES"' in provenance_script
    assert "contents/$path?ref=$REMEDIATION_HEAD_SHA" in provenance_script
    assert 'rev-parse "$HEAD_SHA:$path"' in provenance_script
    assert "use_candidate_memory=true" in provenance_script
    subprocess.run(
        ("bash", "-n"),
        input=provenance_script,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "GH_TOKEN" not in capture_env
    assert capture_env["USE_CANDIDATE_MEMORY"] == (
        "${{ steps.stacked_memory.outputs.use_candidate_memory }}"
    )
    assert "MEMORY_ROOT=base" in capture_script
    assert "immunity audit" in capture_script
    assert "immunity verify" in capture_script
    assert "MEMORY_ROOT=candidate" in capture_script
    assert '--memory-repository-root "$MEMORY_ROOT"' in capture_script


def test_candidate_workflow_never_receives_oidc_or_write_permissions() -> None:
    candidate = _workflow("antibody-candidate.yml")

    assert _mapping(candidate["permissions"]) == {
        "contents": "read",
        "pull-requests": "read",
    }
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
    source_deploy = _step(deploy, "Deploy source to Cloud Run")
    env_vars = cast(str, _mapping(source_deploy["with"])["env_vars"])
    assert "AGENT_ANTIBODY_VULNERABLE_REPLAY_ATTEMPTS=2" in env_vars
    assert "AGENT_ANTIBODY_NORMAL_REPLAY_ATTEMPTS=2" in env_vars


def test_pull_request_ci_defers_only_the_snapshot_projection() -> None:
    jobs = _mapping(_workflow("ci.yml")["jobs"])
    regression = _step(_mapping(jobs["test"]), "Execute generated antibody regression")
    env = _mapping(regression["env"])
    script = cast(str, regression["run"])

    assert env["BASE_SHA"] == "${{ github.event.pull_request.base.sha || '' }}"
    assert env["HEAD_SHA"] == "${{ github.event.pull_request.head.sha || '' }}"
    assert 'if [ "$GITHUB_EVENT_NAME" = "pull_request" ]' in script
    assert 'git diff --quiet "$BASE_SHA...$HEAD_SHA" -- immunities/v1' in script
    assert "--allow-stale-snapshot" in script
    assert "agent-antibody immunity verify --repository-root ." in script
