"""
LangGraph state for the NEXUS pipeline.

Supports parallel feature tracks, durable checkpointing, and long-lived runs.
"""

from __future__ import annotations

import operator
import time
import uuid
from typing import Annotated, Any, Literal, TypedDict


# ---------------------------------------------------------------------------
# Atomic result from a single agent invocation
# ---------------------------------------------------------------------------

class AgentResult(TypedDict, total=False):
    agent_name: str
    model_used: str
    model_tier: str
    phase: str
    feature_id: str          # which feature this result belongs to
    task_type: str
    output: str
    status: Literal["success", "retry", "failed"]
    attempt: int
    timestamp: float


# ---------------------------------------------------------------------------
# A feature decomposed by the supervisor
# ---------------------------------------------------------------------------

class Feature(TypedDict, total=False):
    id: str                  # unique slug, e.g. "auth-system"
    title: str
    description: str
    priority: int            # lower = higher priority
    status: Literal["pending", "in_progress", "done", "failed"]
    current_phase: str
    depends_on: list[str]    # feature ids this depends on


# ---------------------------------------------------------------------------
# Quality-gate verdict
# ---------------------------------------------------------------------------

class GateVerdict(TypedDict, total=False):
    phase: str
    feature_id: str
    passed: bool
    reason: str
    reviewed_by: str
    timestamp: float


# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------

class PipelineState(TypedDict, total=False):
    # --- Identity ---
    run_id: str                                           # persistent run identifier
    run_started: float                                    # epoch timestamp

    # --- Inputs ---
    project_brief: str
    repo_root: str

    # --- Feature decomposition ---
    features: list[Feature]                               # set by supervisor decompose step

    # --- Accumulating outputs (reducer = list concat) ---
    results: Annotated[list[AgentResult], operator.add]
    gate_verdicts: Annotated[list[GateVerdict], operator.add]
    messages: Annotated[list[str], operator.add]

    # --- Phase tracking ---
    current_phase: str
    completed_phases: Annotated[list[str], operator.add]

    # --- Control ---
    retry_count: int
    should_halt: bool
    halt_reason: str

    # --- Resume escalation ---
    # Feature IDs whose last gate failed; phase nodes use a higher model tier for these
    retry_features: list[str]


def new_run_id() -> str:
    """Generate a short, human-readable run ID."""
    return f"run-{uuid.uuid4().hex[:8]}"


def make_initial_state(
    project_brief: str,
    repo_root: str,
    run_id: str | None = None,
) -> PipelineState:
    return PipelineState(
        run_id=run_id or new_run_id(),
        run_started=time.time(),
        project_brief=project_brief,
        repo_root=repo_root,
        features=[],
        results=[],
        gate_verdicts=[],
        messages=[],
        current_phase="",
        completed_phases=[],
        retry_count=0,
        should_halt=False,
        halt_reason="",
    )
