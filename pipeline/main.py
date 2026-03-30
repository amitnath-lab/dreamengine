#!/usr/bin/env python3
"""
DreamEngine NEXUS Pipeline — CLI entry point.

Usage:
  # Start a new run
  python -m pipeline --brief "Build a SaaS task manager with auth and billing"

  # Start with a brief file
  python -m pipeline --brief-file project-brief.txt

  # Run only specific phases
  python -m pipeline --brief "..." --phases discover,strategize

  # Resume a halted/crashed run
  python -m pipeline --resume run-a1b2c3d4

  # Check run status
  python -m pipeline --status run-a1b2c3d4

  # List all runs
  python -m pipeline --list-runs

  # Diagnostics
  python -m pipeline --list-agents
  python -m pipeline --list-models
  python -m pipeline --check-bedrock
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pipeline.agents import PHASE_AGENTS, load_all_agents
from pipeline.config import DEFAULT_MODELS, PipelineConfig
from pipeline.graph import DEFAULT_DB_PATH, build_durable_graph, get_checkpointer
from pipeline.models import set_trace_path
from pipeline.state import PipelineState, make_initial_state, new_run_id

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Diagnostic commands
# ---------------------------------------------------------------------------

def check_bedrock(config: PipelineConfig) -> None:
    import boto3
    print(f"Checking AWS Bedrock in {config.aws_bedrock_region} ...")
    try:
        client = boto3.client("bedrock", region_name=config.aws_bedrock_region)
        resp = client.list_foundation_models()
        available = {m["modelId"] for m in resp.get("modelSummaries", [])}
        print(f"  OK: {len(available)} foundation models available\n")
        for spec in config.models:
            if spec.provider != "bedrock":
                continue
            found = spec.name in available
            icon = "OK" if found else "MISSING"
            extra = "" if found else "  <- enable in Bedrock console"
            print(f"  [{icon:7s}] {spec.name:30s} ({spec.tier.value}){extra}")
    except Exception as e:
        print(f"  FAILED: {e}")
        print(f"  Configure AWS credentials: aws configure")
        print(f"  Ensure Bedrock access is enabled in {config.aws_bedrock_region}")
        sys.exit(1)


def list_agents(config: PipelineConfig) -> None:
    registry = load_all_agents(config.agent_specs_root)
    print(f"\n{len(registry)} agents loaded from {config.agent_specs_root}\n")
    for phase, names in PHASE_AGENTS.items():
        print(f"  {phase.upper()}")
        for name in names:
            agent = registry.get(name)
            if agent:
                print(f"    {agent.emoji} {agent.name:35s} [{agent.division}]")
            else:
                print(f"    ? {name:35s} [NOT FOUND]")
        print()


def list_models() -> None:
    print("\nModel tiers (cheapest first):\n")
    for spec in DEFAULT_MODELS:
        print(f"  {spec.tier.value:16s}  {spec.name:30s}  "
              f"({spec.provider})  strengths={list(spec.strengths)}")
    print()


# ---------------------------------------------------------------------------
# Run management
# ---------------------------------------------------------------------------

def list_runs(db_path: str) -> None:
    """List all pipeline runs from the checkpoint database."""
    if not Path(db_path).exists():
        print("No runs found (database does not exist yet).")
        return

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
        ).fetchall()
        if not rows:
            print("No runs found.")
            return
        print(f"\n{len(rows)} run(s) found:\n")
        for (thread_id,) in rows:
            # Get latest checkpoint info
            latest = conn.execute(
                "SELECT checkpoint FROM checkpoints WHERE thread_id = ? "
                "ORDER BY checkpoint_id DESC LIMIT 1",
                (thread_id,),
            ).fetchone()
            status = "unknown"
            phase = "?"
            if latest and latest[0]:
                try:
                    data = json.loads(latest[0])
                    channel = data.get("channel_values", {})
                    phase = channel.get("current_phase", "?")
                    if channel.get("should_halt"):
                        status = "HALTED"
                    elif phase == "operate":
                        status = "COMPLETED"
                    else:
                        status = "IN PROGRESS"
                except Exception:
                    pass
            print(f"  {thread_id}  [{status:12s}]  phase: {phase}")
    finally:
        conn.close()
    print()


def show_status(run_id: str, db_path: str) -> None:
    """Show detailed status for a specific run."""
    if not Path(db_path).exists():
        print(f"No database found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT checkpoint FROM checkpoints WHERE thread_id = ? "
            "ORDER BY checkpoint_id DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        if not row:
            print(f"Run '{run_id}' not found.")
            return

        data = json.loads(row[0])
        channel = data.get("channel_values", {})

        print(f"\n  Run: {run_id}")
        print(f"  Current phase: {channel.get('current_phase', '?')}")
        print(f"  Completed phases: {channel.get('completed_phases', [])}")
        print(f"  Halted: {channel.get('should_halt', False)}")
        if channel.get("halt_reason"):
            print(f"  Halt reason: {channel['halt_reason']}")

        features = channel.get("features", [])
        if features:
            print(f"\n  Features ({len(features)}):")
            for f in features:
                print(f"    [{f.get('status', '?'):12s}] {f.get('id', '?'):20s} {f.get('title', '')}")

        results = channel.get("results", [])
        if results:
            # Tier usage
            tier_counts: dict[str, int] = {}
            model_counts: dict[str, int] = {}
            for r in results:
                tier_counts[r.get("model_tier", "?")] = tier_counts.get(r.get("model_tier", "?"), 0) + 1
                model_counts[r.get("model_used", "?")] = model_counts.get(r.get("model_used", "?"), 0) + 1

            successes = sum(1 for r in results if r.get("status") == "success")
            print(f"\n  Results: {successes}/{len(results)} succeeded")
            print(f"  Model usage:")
            for model, count in sorted(model_counts.items(), key=lambda x: -x[1]):
                print(f"    {model:30s}  {count} tasks")

        verdicts = channel.get("gate_verdicts", [])
        if verdicts:
            print(f"\n  Quality gates:")
            for v in verdicts:
                icon = "PASS" if v.get("passed") else "FAIL"
                print(f"    [{icon}] {v.get('phase', '?')}/{v.get('feature_id', '?')}: {v.get('reason', '')}")

        messages = channel.get("messages", [])
        if messages:
            print(f"\n  Recent log ({min(len(messages), 20)} of {len(messages)}):")
            for m in messages[-20:]:
                print(f"    {m}")
    finally:
        conn.close()
    print()


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

def run_pipeline(
    brief: str,
    config: PipelineConfig,
    run_id: str | None = None,
    db_path: str = DEFAULT_DB_PATH,
) -> dict:
    """Start a new pipeline run with durable checkpointing."""
    rid = run_id or new_run_id()
    initial = make_initial_state(brief, config.agent_specs_root, run_id=rid)

    graph, checkpointer = build_durable_graph(config, db_path)

    thread_config = {"configurable": {"thread_id": rid}}

    trace = str(Path(db_path).parent / f"trace-{rid}.jsonl")
    set_trace_path(trace)

    print(f"\n  Run ID: {rid}")
    print(f"  Database: {db_path}")
    print(f"  Trace:    {trace}")
    print(f"  Resume with: python -m pipeline --resume {rid}\n")

    final = graph.invoke(initial, config=thread_config)
    return final


def resume_pipeline(
    run_id: str,
    config: PipelineConfig,
    db_path: str = DEFAULT_DB_PATH,
) -> dict:
    """Resume a previously halted or crashed run.

    Loads the last checkpoint, strips results and verdicts for any failed
    feature+phase combos, then starts a fresh invocation so those features
    are re-executed while all passing work is reused.
    """
    graph, checkpointer = build_durable_graph(config, db_path)
    thread_config = {"configurable": {"thread_id": run_id}}

    # Read the saved state
    snapshot = graph.get_state(thread_config)
    if not snapshot or not snapshot.values:
        print(f"  No checkpoint found for run '{run_id}'.")
        sys.exit(1)

    state = dict(snapshot.values)

    # Identify failed gate combos (phase, feature_id)
    failed_combos: set[tuple[str, str]] = {
        (v["phase"], v["feature_id"])
        for v in state.get("gate_verdicts", [])
        if not v.get("passed")
    }

    if not failed_combos:
        print(f"  No failed gates in run '{run_id}' — nothing to re-run.")
        sys.exit(0)

    failed_phases = {phase for phase, _ in failed_combos}
    print(f"\n  Resuming run: {run_id}")
    print(f"  Database: {db_path}")
    print(f"  Re-running: {', '.join(f'{p}/{fid}' for p, fid in sorted(failed_combos))}\n")

    # Strip results and gate verdicts for failed combos
    clean_results = [
        r for r in state.get("results", [])
        if (r.get("phase"), r.get("feature_id")) not in failed_combos
    ]
    clean_verdicts = [
        v for v in state.get("gate_verdicts", [])
        if (v.get("phase"), v.get("feature_id")) not in failed_combos
    ]
    # Remove failed phases from completed_phases so their gates re-evaluate
    clean_phases = [p for p in state.get("completed_phases", []) if p not in failed_phases]

    retry_feature_ids = [fid for _, fid in failed_combos]

    # Use a new thread_id so operator.add reducers start truly fresh.
    # Invoking on the old thread would *accumulate* results/messages/verdicts
    # on top of the existing checkpoint rather than replacing them.
    continuation_id = new_run_id()
    new_thread_config = {"configurable": {"thread_id": continuation_id}}

    clean_state = {
        **state,
        "run_id": continuation_id,
        "results": clean_results,
        "gate_verdicts": clean_verdicts,
        "completed_phases": clean_phases,
        "should_halt": False,
        "halt_reason": "",
        "messages": [],
        "retry_features": retry_feature_ids,
    }

    trace = str(Path(db_path).parent / f"trace-{continuation_id}.jsonl")
    set_trace_path(trace)

    print(f"  Continuation ID: {continuation_id}")
    print(f"  Trace:    {trace}")
    print(f"  Resume next time with: python -m pipeline --resume {continuation_id}\n")

    final = graph.invoke(clean_state, config=new_thread_config)
    return final


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(state: dict) -> None:
    print("\n" + "=" * 70)
    print("  NEXUS PIPELINE — RUN REPORT")
    print("=" * 70)

    for msg in state.get("messages", []):
        print(f"  {msg}")

    # Deduplicate: keep the last verdict per (phase, feature_id) — resumes append new ones
    seen_verdicts: dict[tuple[str, str], dict] = {}
    for v in state.get("gate_verdicts", []):
        key = (v.get("phase", ""), v.get("feature_id", ""))
        seen_verdicts[key] = v
    print("\n  QUALITY GATES:")
    for v in seen_verdicts.values():
        icon = "PASS" if v["passed"] else "FAIL"
        fid = v.get("feature_id", "?")
        print(f"    [{icon}] {v['phase']:12s} / {fid:20s}  {v['reason']}")

    results = state.get("results", [])
    tier_counts: dict[str, int] = {}
    model_counts: dict[str, int] = {}
    for r in results:
        tier_counts[r["model_tier"]] = tier_counts.get(r["model_tier"], 0) + 1
        model_counts[r["model_used"]] = model_counts.get(r["model_used"], 0) + 1

    print("\n  MODEL USAGE:")
    for tier, count in sorted(tier_counts.items()):
        print(f"    {tier:16s}  {count} tasks")
    print()
    for model, count in sorted(model_counts.items(), key=lambda x: -x[1]):
        print(f"    {model:30s}  {count} tasks")

    successes = sum(1 for r in results if r["status"] == "success")
    print(f"\n  RESULTS: {successes}/{len(results)} tasks succeeded")

    features = state.get("features", [])
    if features:
        print(f"\n  FEATURES ({len(features)}):")
        for f in features:
            print(f"    {f.get('id', '?'):20s}  {f.get('title', '')}")

    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DreamEngine NEXUS Pipeline — durable, parallel AI dev workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python -m pipeline --brief "Build a task manager with auth and billing"
  python -m pipeline --resume run-a1b2c3d4
  python -m pipeline --status run-a1b2c3d4
  python -m pipeline --list-runs
  python -m pipeline --check-bedrock
""",
    )

    # Run modes
    run_group = parser.add_argument_group("Run")
    run_group.add_argument("--brief", type=str, help="Project brief (inline)")
    run_group.add_argument("--brief-file", type=str, help="Path to brief file")
    run_group.add_argument("--resume", type=str, metavar="RUN_ID", help="Resume a halted/crashed run")

    # Inspection
    inspect_group = parser.add_argument_group("Inspect")
    inspect_group.add_argument("--status", type=str, metavar="RUN_ID", help="Show run status")
    inspect_group.add_argument("--list-runs", action="store_true", help="List all runs")
    inspect_group.add_argument("--list-agents", action="store_true", help="List agents by phase")
    inspect_group.add_argument("--list-models", action="store_true", help="List model tiers")
    inspect_group.add_argument("--check-bedrock", action="store_true", help="Check AWS Bedrock status")

    # Config
    cfg_group = parser.add_argument_group("Config")
    cfg_group.add_argument("--bedrock-region", type=str, help="AWS Bedrock region")
    cfg_group.add_argument("--db-path", type=str, default=DEFAULT_DB_PATH, help="Checkpoint DB path")
    cfg_group.add_argument("--verbose", action="store_true", help="Debug logging")
    cfg_group.add_argument("--output", type=str, help="Save final state to JSON file")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    config = PipelineConfig(agent_specs_root=REPO_ROOT)
    if args.bedrock_region:
        config.aws_bedrock_region = args.bedrock_region

    # --- Diagnostic commands ---
    if args.check_bedrock:
        check_bedrock(config)
        return
    if args.list_agents:
        list_agents(config)
        return
    if args.list_models:
        list_models()
        return
    if args.list_runs:
        list_runs(args.db_path)
        return
    if args.status:
        show_status(args.status, args.db_path)
        return

    # --- Resume ---
    if args.resume:
        if not config.anthropic_api_key:
            print("ANTHROPIC_API_KEY not set.")
            sys.exit(1)
        final = resume_pipeline(args.resume, config, db_path=args.db_path)
        print_report(final)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(final, f, indent=2, default=str)
        return

    # --- New run ---
    brief = args.brief
    if args.brief_file:
        brief = Path(args.brief_file).read_text(encoding="utf-8")
    if not brief:
        parser.error("Provide --brief, --brief-file, or --resume")

    if not config.anthropic_api_key:
        print("ANTHROPIC_API_KEY not set. Export it: export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    print("\n  DreamEngine NEXUS Pipeline")
    print(f"  Repo:    {REPO_ROOT}")
    print(f"  Bedrock: {config.aws_bedrock_region}")
    print(f"  DB:      {args.db_path}")

    final = run_pipeline(brief, config, db_path=args.db_path)
    print_report(final)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(final, f, indent=2, default=str)
        print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    main()
