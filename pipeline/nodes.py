"""
LangGraph node functions for the NEXUS pipeline.

Supports:
  - Parallel feature execution via asyncio within each phase
  - Per-feature quality gates
  - Retry with escalation
  - All state mutations are returned as dicts (LangGraph reducer pattern)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from .agents import AgentSpec, agents_for_phase, load_all_agents
from .config import PipelineConfig
from .models import call_model, call_model_async
from .state import AgentResult, Feature, PipelineState
from .supervisor import (
    decompose_features,
    plan_phase,
    quality_gate,
    select_model_for_task,
)

log = logging.getLogger(__name__)

_registry_cache: dict[str, Any] = {}


def _get_registry(repo_root: str):
    if repo_root not in _registry_cache:
        _registry_cache[repo_root] = load_all_agents(repo_root)
    return _registry_cache[repo_root]


# ===================================================================
# Node: decompose  — supervisor breaks brief into parallel features
# ===================================================================

def make_decompose_node(config: PipelineConfig):
    def decompose(state: PipelineState) -> dict[str, Any]:
        log.info("\n" + "=" * 60)
        log.info("  SUPERVISOR: Decomposing project into features")
        log.info("=" * 60)

        features = decompose_features(state["project_brief"], config)
        feature_msgs = [
            f"  Feature [{f['id']}] p{f['priority']}: {f['title']}"
            + (f" (depends: {f['depends_on']})" if f.get("depends_on") else "")
            for f in features
        ]
        log.info(f"  Decomposed into {len(features)} features")
        for m in feature_msgs:
            log.info(m)

        return {
            "features": features,
            "messages": [
                f">> Supervisor decomposed project into {len(features)} features:",
                *feature_msgs,
            ],
        }
    decompose.__name__ = "decompose"
    return decompose


# ===================================================================
# Node: phase executor — runs one phase across all features in parallel
# ===================================================================

async def _execute_feature_phase(
    state: PipelineState,
    phase: str,
    feature: Feature,
    config: PipelineConfig,
) -> tuple[list[AgentResult], list[str]]:
    """Execute a single phase for a single feature. Returns (results, messages)."""
    registry = _get_registry(state["repo_root"])
    available = agents_for_phase(phase, registry)

    if not available:
        return [], [f"  [{feature['id']}] No agents for phase {phase} - skipped"]

    # Supervisor plans the work
    plan = plan_phase(state, phase, feature, available, config)
    results: list[AgentResult] = []
    messages: list[str] = [
        f"  [{feature['id']}] Phase {phase}: {len(plan)} tasks planned"
    ]

    # Group tasks by parallel_group for concurrent execution
    groups: dict[int, list[dict]] = {}
    for task_def in plan:
        g = task_def.get("parallel_group", 0)
        groups.setdefault(g, []).append(task_def)

    for group_id in sorted(groups.keys()):
        group_tasks = groups[group_id]
        coros = []
        task_meta = []

        for task_def in group_tasks:
            agent_name = task_def.get("agent", "Unknown")
            task_desc = task_def.get("task", "")
            rec_tier = task_def.get("recommended_tier", "free_fast")

            agent_spec = registry.get(agent_name)
            if not agent_spec:
                messages.append(f"    [{feature['id']}] Agent '{agent_name}' not found - skipped")
                continue

            model_spec = select_model_for_task(agent_name, rec_tier, config)

            system_prompt = (
                f"You are the {agent_spec.name} agent.\n"
                f"Division: {agent_spec.division}\n"
                f"Vibe: {agent_spec.vibe}\n\n"
                f"{agent_spec.system_prompt}\n\n"
                f"---\n"
                f"Phase: **{phase.upper()}** | Feature: **{feature['title']}**\n"
                f"Be concise and produce actionable output."
            )
            user_prompt = (
                f"PROJECT: {state['project_brief'][:300]}\n\n"
                f"FEATURE: {feature['title']}\n{feature['description']}\n\n"
                f"YOUR TASK:\n{task_desc}\n\n"
                f"Produce your deliverable. Be specific and concise."
            )

            coros.append(call_model_async(model_spec, system_prompt, user_prompt, config))
            task_meta.append((agent_name, model_spec, task_desc))

        if not coros:
            continue

        # Execute group in parallel
        outputs = await asyncio.gather(*coros, return_exceptions=True)

        for (agent_name, model_spec, task_desc), output in zip(task_meta, outputs):
            if isinstance(output, Exception):
                status = "failed"
                output_text = f"ERROR: {output}"
                log.warning(f"    [{feature['id']}] {agent_name} failed: {output}")
            else:
                status = "success"
                output_text = output

            result = AgentResult(
                agent_name=agent_name,
                model_used=model_spec.name,
                model_tier=model_spec.tier.value,
                phase=phase,
                feature_id=feature["id"],
                task_type=task_desc[:80],
                output=output_text,
                status=status,
                attempt=1,
                timestamp=time.time(),
            )
            results.append(result)

            icon = "[OK]" if status == "success" else "[FAIL]"
            messages.append(
                f"    {icon} {agent_name} -> {model_spec.name} ({model_spec.tier.value})"
            )

    return results, messages


def make_phase_node(phase: str, config: PipelineConfig):
    """
    Create a LangGraph node that executes *phase* across all features in parallel.
    Features whose dependencies haven't completed yet are deferred.
    """
    def phase_node(state: PipelineState) -> dict[str, Any]:
        features = state.get("features", [])
        if not features:
            return {"messages": [f"Phase {phase}: no features to process"], "current_phase": phase}

        log.info(f"\n{'='*60}")
        log.info(f"  PHASE: {phase.upper()} ({len(features)} features)")
        log.info(f"{'='*60}")

        # Determine which features are ready (dependencies satisfied)
        done_features = set()
        for v in state.get("gate_verdicts", []):
            if v.get("passed"):
                done_features.add(v.get("feature_id", ""))

        ready_features = []
        deferred_features = []
        for f in features:
            if f.get("status") == "done":
                continue
            deps = f.get("depends_on", [])
            if all(d in done_features or d == f["id"] for d in deps):
                ready_features.append(f)
            else:
                deferred_features.append(f)

        if not ready_features:
            ready_features = [f for f in features if f.get("status") != "done"]

        messages = [f">> Phase {phase}: {len(ready_features)} features ready, {len(deferred_features)} deferred"]

        # Run all ready features in parallel using asyncio
        async def run_all():
            tasks = [
                _execute_feature_phase(state, phase, f, config)
                for f in ready_features
            ]
            return await asyncio.gather(*tasks)

        loop_results = asyncio.run(run_all())

        all_results: list[AgentResult] = []
        for feature_results, feature_messages in loop_results:
            all_results.extend(feature_results)
            messages.extend(feature_messages)

        return {
            "results": all_results,
            "messages": messages,
            "current_phase": phase,
        }

    phase_node.__name__ = f"phase_{phase}"
    return phase_node


# ===================================================================
# Node: quality gate — supervisor reviews all features for a phase
# ===================================================================

def make_gate_node(phase: str, config: PipelineConfig):
    """Quality gate: reviews each feature's phase outputs."""
    def gate_node(state: PipelineState) -> dict[str, Any]:
        features = state.get("features", [])
        verdicts = []
        messages = []
        any_failed = False

        for feature in features:
            if feature.get("status") == "done":
                continue

            # Check if this feature has results for this phase
            phase_results = [
                r for r in state.get("results", [])
                if r.get("phase") == phase and r.get("feature_id") == feature["id"]
            ]
            if not phase_results:
                continue

            verdict = quality_gate(state, phase, feature, config)
            verdicts.append(verdict)

            icon = "[PASS]" if verdict["passed"] else "[FAIL]"
            msg = f"  {icon} Gate [{phase}/{feature['id']}]: {verdict['reason']}"
            messages.append(msg)
            log.info(msg)

            if not verdict["passed"]:
                any_failed = True

        return {
            "gate_verdicts": verdicts,
            "messages": messages,
            "completed_phases": [phase] if not any_failed else [],
            "should_halt": any_failed,
            "halt_reason": f"Quality gate failed for phase: {phase}" if any_failed else "",
        }

    gate_node.__name__ = f"gate_{phase}"
    return gate_node
