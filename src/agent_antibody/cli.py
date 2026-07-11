from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from agent_antibody.contracts import PolicyMode
from agent_antibody.core_types import tool_id
from agent_antibody.demo import DemoReport, run_demo
from agent_antibody.generic_runner import CaseRun
from agent_antibody.portfolio_demo import run_portfolio_demo, run_target_demo
from agent_antibody.recipes import (
    AntibodyBundle,
    RegressionRecipeGenerator,
    RegressionResult,
    run_regression,
)
from agent_antibody.runner import ScenarioRun, run_scenario
from agent_antibody.scenarios import MALICIOUS_RUNBOOK
from agent_antibody.targets.registry import TARGET_IDS


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
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
                print(_case_run_row("deterministic / vulnerable", report.vulnerable))
                print(_case_run_row("deterministic / protected", report.protected))
                print(_case_run_row("deterministic / normal", report.normal))
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

    return 2
