from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from agent_antibody.adk_capabilities import capture_target_adk_capabilities
from agent_antibody.candidate_assessment import assess_repository_revalidation
from agent_antibody.candidate_evaluation import (
    CandidateEvaluationStatus,
    CandidateEvaluationV2,
    CandidateTargetEvidence,
    TargetCandidateEvaluation,
    merge_candidate_evaluations,
)
from agent_antibody.capabilities import (
    AttackSurfaceDelta,
    CapabilitySnapshot,
    diff_capability_snapshots,
)
from agent_antibody.contracts import PolicyMode
from agent_antibody.core_types import tool_id
from agent_antibody.delta_pipeline import DeltaSecurityPipeline
from agent_antibody.demo import DemoReport, run_demo
from agent_antibody.generic_runner import CaseRun
from agent_antibody.immunity_artifacts import (
    SnapshotLifecycle,
    SnapshotMemoryProjection,
    apply_evaluation,
    audit_immutable_changes,
    detect_affected_targets,
    evaluate_repository,
    load_artifacts,
    validate_assessment_remediation_binding,
    verify_artifact,
    verify_snapshot,
)
from agent_antibody.portfolio_demo import run_portfolio_demo, run_target_demo
from agent_antibody.recipes import (
    AntibodyBundle,
    RegressionRecipeGenerator,
    RegressionResult,
    run_regression,
)
from agent_antibody.runner import ScenarioRun, run_scenario
from agent_antibody.scenarios import MALICIOUS_RUNBOOK
from agent_antibody.targets.registry import TARGET_IDS, get_target_adapter


def _run_row(label: str, run: ScenarioRun) -> str:
    service = run.scenario.service
    before = run.initial_state.replicas[service]
    after = run.final_state.replicas[service]
    return (
        f"{label:<22} {run.mode.value:<8} {before} -> {after}  "
        f"{run.oracle.status.value:<10} issue={len(run.final_state.issues)}"
    )


def _case_run_row(label: str, run: CaseRun) -> str:
    target_id = run.case.target_id
    if target_id == "opsmate":
        service = str(run.case.fixtures["service"])
        before_replicas = cast(dict[str, object], run.initial_state["replicas"])[service]
        after_replicas = cast(dict[str, object], run.final_state["replicas"])[service]
        detail = f"replicas={before_replicas}->{after_replicas}"
    elif target_id == "repomate":
        before_files = cast(dict[str, object], run.initial_state["files"])
        after_files = cast(dict[str, object], run.final_state["files"])
        detail = (
            f"README={'yes' if 'README.md' in before_files else 'no'}"
            f"->{'yes' if 'README.md' in after_files else 'no'}"
        )
    else:
        before_orders = cast(dict[str, object], run.initial_state["orders"])
        after_orders = cast(dict[str, object], run.final_state["orders"])
        order_id = str(run.case.fixtures["order_id"])
        before_order = cast(dict[str, object], before_orders[order_id])
        after_order = cast(dict[str, object], after_orders[order_id])
        detail = f"refunded={before_order['refunded_cents']}->{after_order['refunded_cents']}c"
    return (
        f"{label:<22} {run.mode.value:<8} {target_id:<12} {detail:<24} "
        f"{run.oracle.status.value:<10}"
    )


def _print_human(report: DemoReport, *, show_trace: bool, show_bundle: bool) -> None:
    print("Agent Antibody MVP")
    print(_run_row("malicious / vulnerable", report.vulnerable))
    print(_run_row("malicious / protected", report.protected))
    print(_run_row("normal / protected", report.normal))
    print()
    print(
        "Attack Success Rate  "
        f"{report.metrics.attack_success_before} -> {report.metrics.attack_success_after}"
    )
    print(f"Normal Task Success   {report.metrics.normal_task_success}")
    print("Status                INFECTED -> IMMUNE")

    if show_trace:
        print("\nProtected attack trace")
        for event in report.protected.events:
            tool = tool_id(event.tool) if event.tool else "-"
            print(f"{event.sequence:02d} {event.event_type.value:<20} {tool}")

    if show_bundle:
        print("\nGenerated antibody")
        print(report.antibody.to_yaml().rstrip())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-antibody")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="Run the deterministic OpsMate attack demo")
    demo.add_argument("--json", action="store_true", help="Print a JSON report")
    demo.add_argument("--trace", action="store_true", help="Print the protected execution trace")
    demo.add_argument(
        "--bundle",
        action="store_true",
        help="Print generated policy and regression data",
    )
    demo.add_argument(
        "--output",
        type=Path,
        help="Write the generated antibody YAML to this directory",
    )
    regression = subparsers.add_parser(
        "regression",
        help="Execute one or more generated antibody bundles",
    )
    regression.add_argument("bundles", nargs="+", type=Path)
    regression.add_argument("--json", action="store_true", help="Print JSON results")

    portfolio = subparsers.add_parser(
        "portfolio",
        help="Run deterministic security evaluations for the registered target agents",
    )
    portfolio.add_argument(
        "--target",
        choices=(*TARGET_IDS, "all"),
        default="all",
        help="Target agent to evaluate, or all targets (default: all)",
    )
    portfolio.add_argument("--json", action="store_true", help="Print JSON reports")

    probe = subparsers.add_parser(
        "probe",
        help="Run OpsMate through a live Gemini model against the local simulator",
    )
    probe.add_argument(
        "--model",
        help="Gemini model name; defaults to AGENT_ANTIBODY_MODEL",
    )
    probe.add_argument(
        "--vulnerable",
        action="store_true",
        help="Omit the generated antibody policy",
    )
    probe.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Overall ADK run timeout in seconds (default: 60)",
    )
    probe.add_argument(
        "--max-llm-calls",
        type=int,
        default=12,
        help="Maximum model calls in one ADK run (default: 12)",
    )
    probe.add_argument("--trace", action="store_true", help="Print requested tools and decisions")
    probe.add_argument("--json", action="store_true", help="Print a JSON result")

    live = subparsers.add_parser(
        "live",
        help="Run Gemini Attack Agent, target agent, and Antibody Agent end to end",
    )
    live.add_argument(
        "--model",
        help="Gemini model name; defaults to AGENT_ANTIBODY_MODEL",
    )
    live.add_argument(
        "--target",
        choices=TARGET_IDS,
        default="opsmate",
        help="Target agent to evaluate (default: opsmate)",
    )
    live.add_argument("--json", action="store_true", help="Print the complete JSON report")

    capabilities = subparsers.add_parser(
        "capabilities",
        help="Capture and compare the tools Google ADK actually exposes to its model",
    )
    capability_subparsers = capabilities.add_subparsers(
        dest="capability_command",
        required=True,
    )
    capture = capability_subparsers.add_parser(
        "capture",
        help="Capture one registered target's provider-facing ADK declarations",
    )
    capture.add_argument("--target", choices=TARGET_IDS, required=True)
    capture.add_argument("--output", type=Path, required=True)
    compare = capability_subparsers.add_parser(
        "diff",
        help="Compare two previously captured ADK capability snapshots",
    )
    compare.add_argument("--base", type=Path, required=True)
    compare.add_argument("--head", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)

    immunity = subparsers.add_parser(
        "immunity",
        help="Capture, verify, and apply immutable policy-and-regression memory",
    )
    immunity_subparsers = immunity.add_subparsers(dest="immunity_command", required=True)
    evaluate = immunity_subparsers.add_parser(
        "evaluate",
        help="Evaluate changed target agents and write a data-only candidate artifact",
    )
    evaluate.add_argument(
        "--repository-root",
        type=Path,
        default=Path("."),
        help="Repository root containing the candidate checkout (default: .)",
    )
    evaluate.add_argument(
        "--source-revision",
        required=True,
        help="Candidate git SHA used to bind immutable immunity memory",
    )
    evaluate.add_argument(
        "--changed-files",
        type=Path,
        help="Newline-delimited repository-relative paths; defaults to no selected targets",
    )
    evaluate.add_argument(
        "--target",
        choices=(*TARGET_IDS, "all", "auto"),
        default="auto",
        help="Override changed-path selection (default: auto)",
    )
    evaluate.add_argument("--output", type=Path, required=True)

    apply = immunity_subparsers.add_parser(
        "apply",
        help="Validate a candidate artifact and write only immutable immunity files",
    )
    apply.add_argument("--evaluation", type=Path, required=True)
    apply.add_argument(
        "--assessment",
        type=Path,
        help="Typed v2 assessment used to label active versus pending memory",
    )
    apply.add_argument("--repository-root", type=Path, default=Path("."))
    apply.add_argument(
        "--lifecycle-state",
        choices=("baseline", "verified_pending_review", "release_ready"),
        default="verified_pending_review",
    )
    apply.add_argument("--source-pr-number", type=int)
    apply.add_argument("--source-pr-url")
    apply.add_argument("--remediation-branch")
    apply.add_argument("--remediation-pr-url")
    apply.add_argument("--workflow-run-url")
    apply.add_argument(
        "--trusted-source-revision",
        help="Current PR head SHA independently resolved by the trusted writer",
    )
    apply.add_argument(
        "--trusted-changed-files",
        type=Path,
        help="Newline-delimited changed paths independently fetched by the trusted writer",
    )
    apply.add_argument(
        "--refresh-snapshot",
        action="store_true",
        help="Rewrite only the allowlisted snapshot after a generated PR URL is known",
    )

    verify = immunity_subparsers.add_parser(
        "verify",
        help="Replay all persisted immunity memory against the current target adapters",
    )
    verify.add_argument("--repository-root", type=Path, default=Path("."))
    verify.add_argument(
        "--target",
        choices=(*TARGET_IDS, "all"),
        default="all",
    )
    verify.add_argument("--json", action="store_true")

    audit = immunity_subparsers.add_parser(
        "audit",
        help="Reject immutable-memory deletion, modification, or untruthful dashboard changes",
    )
    audit.add_argument("--repository-root", type=Path, default=Path("."))
    audit.add_argument("--base-revision", required=True)
    audit.add_argument("--head-revision", default="HEAD")
    gate = immunity_subparsers.add_parser(
        "gate",
        help="Apply the fail-closed CI exit status from a typed candidate assessment",
    )
    gate.add_argument("--assessment", type=Path, required=True)
    assess = immunity_subparsers.add_parser(
        "assess",
        help="Replay active memory and classify capability deltas without creating memory",
    )
    assess.add_argument("--memory-repository-root", type=Path, default=Path("."))
    assess.add_argument("--base-revision", required=True)
    assess.add_argument("--source-revision", required=True)
    assess.add_argument("--capability-delta-dir", type=Path, required=True)
    assess.add_argument(
        "--target",
        choices=(*TARGET_IDS, "all"),
        action="append",
        help="Target to assess; repeat for multiple targets (default: all)",
    )
    assess.add_argument("--output", type=Path, required=True)
    merge_assessments = immunity_subparsers.add_parser(
        "merge-assessments",
        help="Replace inconclusive target results with a delta-specific assessment",
    )
    merge_assessments.add_argument("--base", type=Path, required=True)
    merge_assessments.add_argument("--replacement", type=Path, required=True)
    merge_assessments.add_argument("--output", type=Path, required=True)
    affected = immunity_subparsers.add_parser(
        "affected",
        help="Resolve changed repository paths to registered security targets",
    )
    affected.add_argument("--changed-files", type=Path, required=True)
    affected.add_argument("--output", type=Path, required=True)
    evaluate_delta = immunity_subparsers.add_parser(
        "evaluate-delta",
        help="Use Gemini to attack one changed ADK tool and build verified delta memory",
    )
    evaluate_delta.add_argument("--model", help="Defaults to AGENT_ANTIBODY_MODEL")
    evaluate_delta.add_argument("--target", choices=TARGET_IDS, required=True)
    evaluate_delta.add_argument("--memory-repository-root", type=Path, required=True)
    evaluate_delta.add_argument("--candidate-repository-root", type=Path, default=Path("."))
    evaluate_delta.add_argument("--base-revision", required=True)
    evaluate_delta.add_argument("--source-revision", required=True)
    evaluate_delta.add_argument("--changed-files", type=Path, required=True)
    evaluate_delta.add_argument("--capability-delta", type=Path, required=True)
    evaluate_delta.add_argument("--output-dir", type=Path, required=True)
    evaluate_delta.add_argument(
        "--live-target",
        action="store_true",
        help="Run the changed target through Gemini instead of the credential-free ADK harness",
    )
    evaluate_delta.add_argument("--target-timeout", type=float, default=45.0)
    evaluate_delta.add_argument("--target-max-llm-calls", type=int, default=12)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "capabilities":
        if args.capability_command == "capture":
            snapshot = capture_target_adk_capabilities(get_target_adapter(cast(str, args.target)))
            output = cast(Path, args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(snapshot.canonical_json() + "\n", encoding="utf-8")
            print(
                json.dumps(
                    {
                        "target_id": args.target,
                        "snapshot_sha256": snapshot.digest(),
                        "tools": [tool.name for tool in snapshot.tools],
                        "output": str(output),
                    },
                    ensure_ascii=False,
                )
            )
            return 0

        base = CapabilitySnapshot.model_validate_json(
            cast(Path, args.base).read_text(encoding="utf-8")
        )
        head = CapabilitySnapshot.model_validate_json(
            cast(Path, args.head).read_text(encoding="utf-8")
        )
        delta = diff_capability_snapshots(base, head)
        output = cast(Path, args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(delta.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "changed": delta.changed,
                    "attack_required_tools": list(delta.attack_required_tools),
                    "output": str(output),
                },
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "demo":
        report = run_demo()
        if args.json:
            print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
        else:
            _print_human(report, show_trace=args.trace, show_bundle=args.bundle)
        if args.output is not None:
            destination = report.antibody.write(args.output)
            print(f"Wrote antibody bundle: {destination}")
        return 0 if report.acceptance_passed else 1

    if args.command == "regression":
        results: list[tuple[Path, RegressionResult]] = []
        bundle_paths = cast(list[Path], args.bundles)
        for path in bundle_paths:
            bundle = AntibodyBundle.from_yaml(path.read_text(encoding="utf-8"))
            result = run_regression(bundle)
            results.append((path, result))
        if args.json:
            print(
                json.dumps(
                    [
                        {"path": str(path), **result.model_dump(mode="json")}
                        for path, result in results
                    ],
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            for path, result in results:
                status = "PASS" if result.passed else "FAIL"
                reasons = ",".join(result.reasons) if result.reasons else "-"
                print(f"{status:<4} {path} reasons={reasons}")
        return 0 if all(result.passed for _path, result in results) else 1

    if args.command == "portfolio":
        reports = (
            run_portfolio_demo()
            if args.target == "all"
            else (run_target_demo(cast(str, args.target)),)
        )
        if args.json:
            print(
                json.dumps(
                    [report.model_dump(mode="json") for report in reports],
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print("Agent Antibody target portfolio")
            for report in reports:
                print(f"\n{report.target.name}")
                print(_case_run_row("ADK harness / vulnerable", report.vulnerable))
                print(_case_run_row("ADK harness / protected", report.protected))
                print(_case_run_row("ADK harness / normal", report.normal))
                print(f"Memory seed        {report.selected_attack.plan_id} (9 holdouts)")
                metrics = report.suite_metrics
                print(
                    "Attack suite       "
                    f"{metrics.success_before}/{metrics.total} -> "
                    f"{metrics.success_after}/{metrics.total} "
                    f"(confirmed blocks={metrics.confirmed_blocked})"
                )
                print(report.antibody.to_yaml().rstrip())
        return 0 if all(report.acceptance_passed for report in reports) else 1

    if args.command == "probe":
        model = cast(str | None, args.model) or os.getenv("AGENT_ANTIBODY_MODEL")
        if not model:
            parser.error("probe requires --model or AGENT_ANTIBODY_MODEL")
        from agent_antibody.adk_adapter import AdkOpsMate

        rules = None
        if not args.vulnerable:
            baseline = run_scenario(MALICIOUS_RUNBOOK, mode=PolicyMode.ENFORCE)
            rules = RegressionRecipeGenerator().generate(baseline).policy()
        run = run_scenario(
            MALICIOUS_RUNBOOK,
            mode=PolicyMode.ENFORCE,
            rules=rules,
            agent=AdkOpsMate(
                model=model,
                timeout_seconds=cast(float, args.timeout),
                max_llm_calls=cast(int, args.max_llm_calls),
                trust_runbook_actions=True,
            ),
        )
        if args.json:
            print(json.dumps(run.model_dump(mode="json"), ensure_ascii=False, indent=2))
        else:
            label = "live / vulnerable" if args.vulnerable else "live / protected"
            print(_run_row(label, run))
            print(f"Agent reply: {run.agent_result.reply or '<empty>'}")
            if args.trace:
                for event in run.events:
                    if event.event_type.value not in {"tool.requested", "policy.decision"}:
                        continue
                    effect = event.payload.get("effect", "-")
                    tool = tool_id(event.tool) if event.tool else "-"
                    print(f"{event.sequence:02d} {event.event_type.value:<16} {tool:<20} {effect}")
        return 0 if run.oracle.status.value != "INVALID_RUN" else 1

    if args.command == "live":
        model = cast(str | None, args.model) or os.getenv("AGENT_ANTIBODY_MODEL")
        if not model:
            parser.error("live requires --model or AGENT_ANTIBODY_MODEL")
        from agent_antibody.live_pipeline import LiveSecurityPipeline

        report = LiveSecurityPipeline(model=model, target_id=cast(str, args.target)).run()
        if args.json:
            print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
        else:
            print("Agent Antibody live Gemini pipeline")
            print(f"Memory seed        {report.selected_attack.plan_id} (9 holdouts)")
            print(_case_run_row("live / vulnerable", report.vulnerable))
            print(_case_run_row("live / protected", report.protected))
            print(_case_run_row("live / normal", report.normal))
            metrics = report.suite_metrics
            print(
                "Attack suite       "
                f"{metrics.success_before}/{metrics.total} -> "
                f"{metrics.success_after}/{metrics.total} "
                f"(confirmed blocks={metrics.confirmed_blocked})"
            )
            print(f"Antibody Agent     {report.antibody.bundle_id}")
            print(report.antibody.to_yaml().rstrip())
        return 0 if report.acceptance_passed else 1

    if args.command == "immunity":
        immunity_command = cast(str, args.immunity_command)
        if immunity_command == "evaluate-delta":
            model = cast(str | None, args.model) or os.getenv("AGENT_ANTIBODY_MODEL")
            if not model:
                parser.error("evaluate-delta requires --model or AGENT_ANTIBODY_MODEL")
            target_id = cast(str, args.target)
            base_revision = cast(str, args.base_revision)
            source_revision = cast(str, args.source_revision)
            memory_root = cast(Path, args.memory_repository_root)
            changed_paths = tuple(
                line.strip()
                for line in cast(Path, args.changed_files).read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            delta = AttackSurfaceDelta.model_validate_json(
                cast(Path, args.capability_delta).read_text(encoding="utf-8")
            )
            output_dir = cast(Path, args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            try:
                output = DeltaSecurityPipeline(
                    model=model,
                    adapter=get_target_adapter(target_id),
                    memory_repository_root=memory_root,
                    candidate_repository_root=cast(Path, args.candidate_repository_root),
                    target_model=model if cast(bool, args.live_target) else None,
                    target_timeout_seconds=cast(float, args.target_timeout),
                    target_max_llm_calls=cast(int, args.target_max_llm_calls),
                ).run(
                    base_revision=base_revision,
                    source_revision=source_revision,
                    changed_paths=changed_paths,
                    delta=delta,
                )
            except Exception as error:
                artifacts = load_artifacts(
                    repository_root=memory_root,
                    target_id=target_id,
                )
                evidence = CandidateTargetEvidence(
                    existing_policy_artifact_ids=tuple(
                        artifact.artifact_id for artifact in artifacts
                    ),
                    attack_surface_change_ids=tuple(
                        f"{change.kind.value}:{change.tool_name or 'agent-instruction'}"
                        for change in delta.changes
                    ),
                )
                target = TargetCandidateEvaluation.inconclusive(
                    target_id=target_id,
                    current_memory_count=len(artifacts),
                    evidence=evidence,
                    reasons=(
                        f"delta evaluation failed: {type(error).__name__}: {str(error)[:1_000]}",
                    ),
                )
                assessment = CandidateEvaluationV2.from_targets(
                    base_revision=base_revision,
                    source_revision=source_revision,
                    evaluated_at=datetime.now(UTC),
                    targets=(target,),
                )
                (output_dir / "assessment.json").write_text(assessment.to_json(), encoding="utf-8")
                print(
                    json.dumps(
                        {
                            "status": assessment.status.value,
                            "reason": target.inconclusive_reasons[0],
                        }
                    )
                )
                return 0
            (output_dir / "assessment.json").write_text(
                output.assessment.to_json(), encoding="utf-8"
            )
            (output_dir / "remediation.json").write_text(
                output.remediation.to_json(), encoding="utf-8"
            )
            print(
                json.dumps(
                    {
                        "status": output.assessment.status.value,
                        "artifact_ids": [
                            reference.artifact_id
                            for reference in output.assessment.remediation_artifacts
                        ],
                    },
                    ensure_ascii=False,
                )
            )
            return 0

        if immunity_command == "affected":
            changed_paths = tuple(
                line.strip()
                for line in cast(Path, args.changed_files).read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            targets = detect_affected_targets(changed_paths)
            output = cast(Path, args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(list(targets), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps({"targets": list(targets), "output": str(output)}))
            return 0

        if immunity_command == "assess":
            requested_targets = cast(list[str] | None, args.target)
            selected_targets = (
                TARGET_IDS
                if not requested_targets or "all" in requested_targets
                else tuple(target_id for target_id in TARGET_IDS if target_id in requested_targets)
            )
            delta_directory = cast(Path, args.capability_delta_dir)
            capability_deltas = {
                target_id: AttackSurfaceDelta.model_validate_json(
                    (delta_directory / f"{target_id}.json").read_text(encoding="utf-8")
                )
                for target_id in selected_targets
            }
            assessment = assess_repository_revalidation(
                memory_repository_root=cast(Path, args.memory_repository_root),
                base_revision=cast(str, args.base_revision),
                source_revision=cast(str, args.source_revision),
                capability_deltas=capability_deltas,
            )
            output = cast(Path, args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(assessment.to_json(), encoding="utf-8")
            print(
                json.dumps(
                    {
                        "status": assessment.status.value,
                        "targets": [
                            {
                                "target_id": target.target_id,
                                "status": target.status.value,
                                "current_memory_count": target.current_memory_count,
                                "proposed_memory_count": target.proposed_memory_count,
                            }
                            for target in assessment.targets
                        ],
                        "output": str(output),
                    },
                    ensure_ascii=False,
                )
            )
            return 0

        if immunity_command == "gate":
            assessment = CandidateEvaluationV2.from_json(
                cast(Path, args.assessment).read_text(encoding="utf-8")
            )
            print(
                json.dumps(
                    {
                        "status": assessment.status.value,
                        "current_memory_count": assessment.current_memory_count,
                        "proposed_memory_count": assessment.proposed_memory_count,
                        "requires_remediation": assessment.requires_remediation,
                    },
                    ensure_ascii=False,
                )
            )
            if assessment.status == CandidateEvaluationStatus.REVALIDATED:
                return 0
            if assessment.status == CandidateEvaluationStatus.BYPASS_CONFIRMED:
                return 2
            return 1

        if immunity_command == "evaluate":
            changed_files_path = cast(Path | None, args.changed_files)
            changed_paths = (
                tuple(
                    line.strip()
                    for line in changed_files_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
                if changed_files_path is not None
                else ()
            )
            selected = cast(str, args.target)
            target_ids = (
                TARGET_IDS if selected == "all" else (selected,) if selected != "auto" else None
            )
            evaluation = evaluate_repository(
                repository_root=cast(Path, args.repository_root),
                source_revision=cast(str, args.source_revision),
                changed_paths=changed_paths,
                target_ids=target_ids,
            )
            output = cast(Path, args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(evaluation.to_json(), encoding="utf-8")
            print(
                json.dumps(
                    {
                        "output": str(output),
                        "source_revision": evaluation.source_revision,
                        "targets": [
                            {
                                "target_id": candidate.artifact.target_id,
                                "artifact_id": candidate.artifact.artifact_id,
                                "action": candidate.action,
                            }
                            for candidate in evaluation.candidates
                        ],
                        "requires_remediation": evaluation.requires_remediation,
                    },
                    ensure_ascii=False,
                )
            )
            return 0

        if immunity_command == "apply":
            from agent_antibody.immunity_artifacts import ImmunityEvaluation

            evaluation = ImmunityEvaluation.from_json(
                cast(Path, args.evaluation).read_text(encoding="utf-8")
            )
            assessment_path = cast(Path | None, args.assessment)
            assessment = (
                CandidateEvaluationV2.from_json(assessment_path.read_text(encoding="utf-8"))
                if assessment_path is not None
                else None
            )
            if assessment is not None:
                validate_assessment_remediation_binding(assessment, evaluation)
            lifecycle = SnapshotLifecycle(
                state=cast(
                    Literal["baseline", "verified_pending_review", "release_ready"],
                    args.lifecycle_state,
                ),
                source_pr_number=cast(int | None, args.source_pr_number),
                source_pr_url=cast(str | None, args.source_pr_url),
                remediation_branch=cast(str | None, args.remediation_branch),
                remediation_pr_url=cast(str | None, args.remediation_pr_url),
                workflow_run_url=cast(str | None, args.workflow_run_url),
                evaluation_status=(
                    assessment.status.value if assessment is not None else "RELEASE_READY"
                ),
                memory_projections=(
                    tuple(
                        SnapshotMemoryProjection(
                            target_id=target.target_id,
                            active=target.current_memory_count,
                            pending=(target.proposed_memory_count - target.current_memory_count),
                        )
                        for target in assessment.targets
                    )
                    if assessment is not None
                    else ()
                ),
            )
            trusted_changed_files = cast(Path | None, args.trusted_changed_files)
            trusted_changed_paths = (
                tuple(
                    line.strip()
                    for line in trusted_changed_files.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
                if trusted_changed_files is not None
                else None
            )
            written = apply_evaluation(
                evaluation,
                repository_root=cast(Path, args.repository_root),
                lifecycle=lifecycle,
                assessment=assessment,
                refresh_snapshot=cast(bool, args.refresh_snapshot),
                trusted_source_revision=cast(str | None, args.trusted_source_revision),
                trusted_changed_paths=trusted_changed_paths,
            )
            print(json.dumps({"written": [str(path) for path in written]}, ensure_ascii=False))
            return 0

        if immunity_command == "merge-assessments":
            base = CandidateEvaluationV2.from_json(
                cast(Path, args.base).read_text(encoding="utf-8")
            )
            replacement = CandidateEvaluationV2.from_json(
                cast(Path, args.replacement).read_text(encoding="utf-8")
            )
            merged = merge_candidate_evaluations(base=base, replacement=replacement)
            output = cast(Path, args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(merged.to_json(), encoding="utf-8")
            print(
                json.dumps(
                    {
                        "status": merged.status.value,
                        "targets": [
                            {"target_id": target.target_id, "status": target.status.value}
                            for target in merged.targets
                        ],
                        "output": str(output),
                    },
                    ensure_ascii=False,
                )
            )
            return 0

        if immunity_command == "verify":
            selected = cast(str, args.target)
            artifacts = load_artifacts(
                repository_root=cast(Path, args.repository_root),
                target_id=None if selected == "all" else selected,
            )
            expected_targets = TARGET_IDS if selected == "all" else (selected,)
            present_targets = {artifact.target_id for artifact in artifacts}
            missing_targets = tuple(
                target_id for target_id in expected_targets if target_id not in present_targets
            )
            verification_results = tuple(verify_artifact(artifact) for artifact in artifacts)
            snapshot_error: str | None = None
            if selected == "all":
                try:
                    verify_snapshot(repository_root=cast(Path, args.repository_root))
                except (FileNotFoundError, ValueError) as error:
                    snapshot_error = str(error)
            if args.json:
                print(
                    json.dumps(
                        {
                            "artifacts": [
                                result.model_dump(mode="json") for result in verification_results
                            ],
                            "missing_targets": list(missing_targets),
                            "snapshot_error": snapshot_error,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            else:
                for result in verification_results:
                    status = "PASS" if result.passed else "FAIL"
                    reasons = ",".join(result.reasons) if result.reasons else "-"
                    print(
                        f"{status:<4} {result.target_id:<12} {result.artifact_id} "
                        f"blocks={result.blocked_attacks}/{result.attack_count} reasons={reasons}"
                    )
                if missing_targets:
                    print(f"FAIL missing immunity memory for: {','.join(missing_targets)}")
                if snapshot_error is not None:
                    print(f"FAIL snapshot={snapshot_error}")
            return (
                0
                if (
                    verification_results
                    and not missing_targets
                    and snapshot_error is None
                    and all(result.passed for result in verification_results)
                )
                else 1
            )

        if immunity_command == "audit":
            audit_immutable_changes(
                repository_root=cast(Path, args.repository_root),
                base_revision=cast(str, args.base_revision),
                head_revision=cast(str, args.head_revision),
            )
            print("PASS immutable immunity diff")
            return 0

    return 2
