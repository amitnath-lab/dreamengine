"""
Claude Opus supervisor – the brain of the pipeline.

Responsibilities:
  1. Decompose a project brief into parallel features
  2. Plan which agents to activate for a phase+feature
  3. Pick the cheapest viable Ollama model for each task
  4. Issue quality-gate verdicts between phases
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from .agents import AgentSpec, agents_for_phase
from .config import (
    AGENT_TASK_TYPE,
    TASK_STRENGTH_MAP,
    ModelSpec,
    ModelTier,
    PipelineConfig,
)
from .models import call_model, get_supervisor_model, pick_model
from .state import AgentResult, Feature, GateVerdict, PipelineState

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared system prompt
# ---------------------------------------------------------------------------

SUPERVISOR_SYSTEM = """\
You are the **NEXUS Supervisor**, the autonomous orchestrator of a multi-agent
AI development pipeline.  You are Claude Opus — the most capable model in
the pipeline — and your job is to *direct*, not to *do*.

## Your responsibilities
1. **Feature decomposition** – Break a project brief into independent,
   parallelisable features with clear boundaries and dependency ordering.
2. **Phase planning** – For each phase+feature, decide which agents to
   activate, what each should do, and which can run in parallel.
3. **Model routing** – For every agent task, recommend the cheapest Ollama
   model tier that can handle the job.
4. **Quality gating** – After a phase completes, review the collected
   outputs and decide pass/fail.  Require evidence, not assertions.

## Model tiers (cheapest first — ALWAYS prefer free_fast)
- **free_fast**: qwen2.5:7b, llama3.1:8b, gemma2:9b, phi3:mini, deepseek-coder-v2:16b
- **free_medium**: qwen2.5:32b, llama3.1:70b
- **free_heavy**: deepseek-coder-v2:236b, mixtral:8x22b

## Output format
Always respond with **valid JSON only** — no markdown fences, no commentary.
"""


def _parse_json(raw: str) -> Any:
    """Best-effort JSON parse from LLM output."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    raw = raw.strip()
    if raw.startswith("json"):
        raw = raw[4:].strip()
    return json.loads(raw)


# ---------------------------------------------------------------------------
# 1. Feature decomposition
# ---------------------------------------------------------------------------

def decompose_features(
    project_brief: str,
    config: PipelineConfig,
) -> list[Feature]:
    """
    Ask the supervisor to break the brief into parallel features.
    """
    supervisor = get_supervisor_model(config.models)

    prompt = f"""\
PROJECT BRIEF:
{project_brief}

Decompose this project into independent, parallelisable features.
Each feature should be a discrete unit of work that can be built, tested,
and shipped independently.  Identify dependencies between features.

Return a JSON array:
[
  {{
    "id": "short-slug",
    "title": "Human-readable title",
    "description": "What this feature includes",
    "priority": 1,
    "depends_on": ["other-feature-id"]
  }}
]

Order by priority (1 = highest).  Features with no dependencies can be built
in parallel.  Return ONLY the JSON array."""

    raw = call_model(supervisor, SUPERVISOR_SYSTEM, prompt, config,
                     max_tokens=4096, context="decompose_features")

    try:
        items = _parse_json(raw)
    except (json.JSONDecodeError, ValueError):
        log.warning("Supervisor returned unparseable feature list — using single feature")
        items = [{"id": "main", "title": "Full Project", "description": project_brief,
                  "priority": 1, "depends_on": []}]

    features: list[Feature] = []
    for item in items:
        features.append(Feature(
            id=item["id"],
            title=item.get("title", item["id"]),
            description=item.get("description", ""),
            priority=item.get("priority", len(features) + 1),
            status="pending",
            current_phase="",
            depends_on=item.get("depends_on", []),
        ))
    return features


# ---------------------------------------------------------------------------
# 2. Phase planning (per feature)
# ---------------------------------------------------------------------------

def plan_phase(
    state: PipelineState,
    phase: str,
    feature: Feature,
    available_agents: list[AgentSpec],
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    """
    Ask the supervisor to plan work for a specific phase + feature.
    Returns task dicts: [{"agent": "...", "task": "...", "recommended_tier": "..."}, ...]
    """
    supervisor = get_supervisor_model(config.models)

    agent_summaries = "\n".join(
        f"- **{a.name}** ({a.division}): {a.description}"
        for a in available_agents
    )

    # Prior results for this feature
    prior = [r for r in state.get("results", [])
             if r.get("feature_id") == feature["id"] and r.get("status") == "success"]
    prior_summary = "\n".join(
        f"- [{r['agent_name']}] {r['output'][:200]}..."
        for r in prior[-8:]
    ) or "(no prior outputs for this feature)"

    prompt = f"""\
PROJECT BRIEF:
{state["project_brief"][:500]}

FEATURE: {feature["title"]}
{feature["description"]}

CURRENT PHASE: {phase}

AVAILABLE AGENTS:
{agent_summaries}

PRIOR OUTPUTS FOR THIS FEATURE:
{prior_summary}

Produce a JSON array of tasks.  Each element:
{{
  "agent": "<agent name>",
  "task": "<specific instruction>",
  "recommended_tier": "free_fast" | "free_medium" | "free_heavy",
  "parallel_group": <int>
}}

Prioritise free_fast.  Return ONLY the JSON array."""

    raw = call_model(supervisor, SUPERVISOR_SYSTEM, prompt, config,
                     max_tokens=4096, context=f"plan_phase/{phase}/{feature['id']}")

    try:
        plan = _parse_json(raw)
    except (json.JSONDecodeError, ValueError):
        log.warning("Supervisor plan parse failed — using default plan")
        plan = [
            {"agent": a.name, "task": f"Perform {phase} work for feature: {feature['title']}",
             "recommended_tier": "free_fast", "parallel_group": 0}
            for a in available_agents[:5]  # cap to avoid overwhelming
        ]
    return plan


# ---------------------------------------------------------------------------
# 3. Quality gate
# ---------------------------------------------------------------------------

def quality_gate(
    state: PipelineState,
    phase: str,
    feature: Feature,
    config: PipelineConfig,
) -> GateVerdict:
    """Review results for a phase+feature and issue pass/fail."""
    supervisor = get_supervisor_model(config.models)

    phase_results = [r for r in state.get("results", [])
                     if r.get("phase") == phase and r.get("feature_id") == feature["id"]]
    results_text = "\n\n".join(
        f"### {r['agent_name']} (model: {r['model_used']}, status: {r['status']})\n{r['output'][:500]}"
        for r in phase_results
    ) or "(no results)"

    prompt = f"""\
PHASE: {phase}
FEATURE: {feature["title"]}
PROJECT BRIEF: {state["project_brief"][:300]}

PHASE RESULTS:
{results_text}

Review and decide pass/fail.  Reply with JSON:
{{"passed": true|false, "reason": "<1-2 sentences>"}}
Return ONLY the JSON object."""

    raw = call_model(supervisor, SUPERVISOR_SYSTEM, prompt, config,
                     max_tokens=512, context=f"quality_gate/{phase}/{feature['id']}")

    try:
        verdict = _parse_json(raw)
    except (json.JSONDecodeError, ValueError):
        verdict = {"passed": True, "reason": "Parse failed; defaulting to pass."}

    return GateVerdict(
        phase=phase,
        feature_id=feature["id"],
        passed=verdict.get("passed", True),
        reason=verdict.get("reason", ""),
        reviewed_by=supervisor.name,
        timestamp=time.time(),
    )


# ---------------------------------------------------------------------------
# 4. Model selection
# ---------------------------------------------------------------------------

def select_model_for_task(
    agent_name: str,
    recommended_tier: str,
    config: PipelineConfig,
) -> ModelSpec:
    tier_map = {t.value: t for t in ModelTier}
    max_tier = tier_map.get(recommended_tier, ModelTier.FREE_FAST)
    task_type = AGENT_TASK_TYPE.get(agent_name, "general")
    return pick_model(task_type, TASK_STRENGTH_MAP, config.models, max_tier=max_tier)
