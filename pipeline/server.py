"""
DreamEngine NEXUS Pipeline — Web UI server.

Serves a dashboard for visualizing pipeline runs, with human-in-the-loop
approval between phases.

Usage:
    python -m pipeline.web
    # or
    uvicorn pipeline.server:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from langgraph.checkpoint.sqlite import SqliteSaver

from .agents import list_phases
from .config import PipelineConfig
from .graph import DEFAULT_DB_PATH, build_durable_graph
from .models import set_trace_path
from .state import PipelineState, make_initial_state, new_run_id

log = logging.getLogger(__name__)

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="NEXUS Pipeline Dashboard")

# ---------------------------------------------------------------------------
# In-memory tracking of active runs and continuation chains
# ---------------------------------------------------------------------------

# Maps original_run_id -> latest continuation thread_id
_run_chains: dict[str, str] = {}
# Active background threads
_active_threads: dict[str, threading.Thread] = {}
# Lock for thread-safe access
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Shared read-only checkpointer for reading state
# ---------------------------------------------------------------------------

def _get_reader_saver(db_path: str = DEFAULT_DB_PATH) -> SqliteSaver | None:
    """Get a SqliteSaver for reading checkpoint state."""
    if not Path(db_path).exists():
        return None
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return SqliteSaver(conn)


# ---------------------------------------------------------------------------
# Helpers — read state from checkpoint DB
# ---------------------------------------------------------------------------

def _read_checkpoint_state(run_id: str, db_path: str = DEFAULT_DB_PATH) -> dict | None:
    """Read the latest checkpoint state for a run using LangGraph's deserializer."""
    saver = _get_reader_saver(db_path)
    if not saver:
        return None
    try:
        config = {"configurable": {"thread_id": run_id}}
        tup = saver.get_tuple(config)
        if not tup:
            return None
        return tup.checkpoint.get("channel_values", {})
    except Exception as e:
        log.warning(f"Failed to read checkpoint for {run_id}: {e}")
        return None


def _get_effective_run_id(run_id: str) -> str:
    """Resolve to the latest continuation thread_id for a logical run."""
    with _lock:
        return _run_chains.get(run_id, run_id)


def _get_all_runs(db_path: str = DEFAULT_DB_PATH) -> list[dict]:
    """List all runs from the checkpoint database."""
    if not Path(db_path).exists():
        return []
    # Get thread IDs via raw SQL (fast), then read state via SqliteSaver
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        rows = conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
        ).fetchall()
    finally:
        conn.close()

    runs = []
    for (thread_id,) in rows:
        state = _read_checkpoint_state(thread_id, db_path)
        status = "unknown"
        phase = "?"
        brief_preview = ""
        if state:
            phase = state.get("current_phase", "?")
            brief_preview = state.get("project_brief", "")[:100]
            halt_reason = state.get("halt_reason", "")
            if halt_reason.startswith("AWAITING_APPROVAL:"):
                status = "AWAITING_APPROVAL"
            elif halt_reason.startswith("GATE_FAILED:"):
                status = "GATE_FAILED"
            elif state.get("should_halt"):
                status = "HALTED"
            elif phase == "operate" and not state.get("should_halt"):
                status = "COMPLETED"
            else:
                status = "IN_PROGRESS"
        runs.append({
            "run_id": thread_id,
            "status": status,
            "current_phase": phase,
            "brief_preview": brief_preview,
        })
    return runs


def _state_to_response(state: dict) -> dict:
    """Convert pipeline state to a JSON-serializable API response."""
    phases = list_phases()
    completed = state.get("completed_phases", [])
    current = state.get("current_phase", "")
    halt_reason = state.get("halt_reason", "")

    # Determine phase statuses
    phase_statuses = {}
    for p in phases:
        if p in completed:
            phase_statuses[p] = "completed"
        elif p == current and halt_reason.startswith("AWAITING_APPROVAL:"):
            phase_statuses[p] = "awaiting_approval"
        elif p == current and halt_reason.startswith("GATE_FAILED:"):
            phase_statuses[p] = "failed"
        elif p == current:
            phase_statuses[p] = "active"
        else:
            phase_statuses[p] = "pending"

    # Group results by phase then feature
    results_grouped: dict[str, dict[str, list]] = {}
    for r in state.get("results", []):
        p = r.get("phase", "?")
        fid = r.get("feature_id", "?")
        results_grouped.setdefault(p, {}).setdefault(fid, []).append(r)

    # Group verdicts by phase
    verdicts_grouped: dict[str, list] = {}
    for v in state.get("gate_verdicts", []):
        p = v.get("phase", "?")
        verdicts_grouped.setdefault(p, []).append(v)

    return {
        "run_id": state.get("run_id", ""),
        "project_brief": state.get("project_brief", ""),
        "phases": phases,
        "phase_statuses": phase_statuses,
        "current_phase": current,
        "completed_phases": completed,
        "features": state.get("features", []),
        "results_grouped": results_grouped,
        "verdicts_grouped": verdicts_grouped,
        "messages": state.get("messages", [])[-50:],
        "should_halt": state.get("should_halt", False),
        "halt_reason": halt_reason,
        "total_results": len(state.get("results", [])),
        "total_verdicts": len(state.get("gate_verdicts", [])),
    }


# ---------------------------------------------------------------------------
# Pipeline execution in background threads
# ---------------------------------------------------------------------------

def _run_pipeline_thread(
    brief: str,
    run_id: str,
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    """Run the pipeline in a background thread with pause_for_approval enabled."""
    try:
        config = PipelineConfig(
            agent_specs_root=REPO_ROOT,
            pause_for_approval=True,
        )
        initial = make_initial_state(brief, config.agent_specs_root, run_id=run_id)
        graph, checkpointer = build_durable_graph(config, db_path)
        thread_config = {"configurable": {"thread_id": run_id}}

        trace = str(Path(db_path).parent / f"trace-{run_id}.jsonl")
        set_trace_path(trace)

        log.info(f"Pipeline started: {run_id}")
        graph.invoke(initial, config=thread_config)
        log.info(f"Pipeline halted/completed: {run_id}")
    except Exception as e:
        log.error(f"Pipeline error ({run_id}): {e}", exc_info=True)
    finally:
        with _lock:
            _active_threads.pop(run_id, None)


def _resume_pipeline_thread(
    original_run_id: str,
    current_thread_id: str,
    user_feedback: str | None = None,
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    """Resume pipeline from a halted state in a background thread."""
    try:
        config = PipelineConfig(
            agent_specs_root=REPO_ROOT,
            pause_for_approval=True,
        )
        graph, checkpointer = build_durable_graph(config, db_path)
        thread_config = {"configurable": {"thread_id": current_thread_id}}

        snapshot = graph.get_state(thread_config)
        if not snapshot or not snapshot.values:
            log.error(f"No checkpoint found for {current_thread_id}")
            return

        state = dict(snapshot.values)
        halt_reason = state.get("halt_reason", "")

        if halt_reason.startswith("GATE_FAILED:"):
            # Quality gate failed — strip failed results/verdicts and retry
            failed_combos: set[tuple[str, str]] = {
                (v["phase"], v["feature_id"])
                for v in state.get("gate_verdicts", [])
                if not v.get("passed")
            }
            failed_phases = {phase for phase, _ in failed_combos}
            clean_results = [
                r for r in state.get("results", [])
                if (r.get("phase"), r.get("feature_id")) not in failed_combos
            ]
            clean_verdicts = [
                v for v in state.get("gate_verdicts", [])
                if (v.get("phase"), v.get("feature_id")) not in failed_combos
            ]
            clean_phases = [p for p in state.get("completed_phases", []) if p not in failed_phases]
            retry_feature_ids = [fid for _, fid in failed_combos]
        else:
            # Approval halt — keep everything, just clear halt flags
            clean_results = state.get("results", [])
            clean_verdicts = state.get("gate_verdicts", [])
            clean_phases = state.get("completed_phases", [])
            retry_feature_ids = []

        continuation_id = new_run_id()
        new_thread_config = {"configurable": {"thread_id": continuation_id}}

        # Inject user feedback if provided
        extra_messages = []
        if user_feedback:
            extra_messages = [f"USER FEEDBACK: {user_feedback}"]

        clean_state = {
            **state,
            "run_id": continuation_id,
            "results": clean_results,
            "gate_verdicts": clean_verdicts,
            "completed_phases": clean_phases,
            "should_halt": False,
            "halt_reason": "",
            "messages": extra_messages,
            "retry_features": retry_feature_ids,
        }

        trace = str(Path(db_path).parent / f"trace-{continuation_id}.jsonl")
        set_trace_path(trace)

        # Update the chain mapping
        with _lock:
            _run_chains[original_run_id] = continuation_id
            _active_threads[continuation_id] = threading.current_thread()

        log.info(f"Pipeline resumed: {current_thread_id} -> {continuation_id}")
        graph.invoke(clean_state, config=new_thread_config)
        log.info(f"Pipeline halted/completed: {continuation_id}")
    except Exception as e:
        log.error(f"Pipeline resume error: {e}", exc_info=True)
    finally:
        with _lock:
            _active_threads.pop(new_run_id, None)


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------

class StartRunRequest(BaseModel):
    brief: str


class FeedbackRequest(BaseModel):
    feedback: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(500, "UI not found — missing static/index.html")
    return HTMLResponse(index.read_text(encoding="utf-8"))


@app.post("/api/runs")
async def start_run(req: StartRunRequest):
    if not req.brief.strip():
        raise HTTPException(400, "Brief cannot be empty")

    run_id = new_run_id()

    with _lock:
        _run_chains[run_id] = run_id

    t = threading.Thread(
        target=_run_pipeline_thread,
        args=(req.brief.strip(), run_id),
        daemon=True,
    )
    with _lock:
        _active_threads[run_id] = t
    t.start()

    return {"run_id": run_id, "status": "started"}


@app.get("/api/runs")
async def list_runs():
    runs = _get_all_runs()
    # Annotate with original_run_id info
    with _lock:
        chains = dict(_run_chains)
    # Only show the latest continuation for each logical run
    latest_ids = set(chains.values()) if chains else set()
    # Include runs not in any chain, or the latest in each chain
    visible = []
    for r in runs:
        rid = r["run_id"]
        if rid in latest_ids or rid not in {v for v in chains.values() if v != rid}:
            visible.append(r)
    return {"runs": runs}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    effective_id = _get_effective_run_id(run_id)
    state = _read_checkpoint_state(effective_id)
    if not state:
        # Try the original run_id directly
        state = _read_checkpoint_state(run_id)
    if not state:
        raise HTTPException(404, f"Run '{run_id}' not found")
    return _state_to_response(state)


@app.post("/api/runs/{run_id}/approve")
async def approve_run(run_id: str):
    effective_id = _get_effective_run_id(run_id)
    state = _read_checkpoint_state(effective_id)
    if not state:
        raise HTTPException(404, f"Run '{run_id}' not found")

    if not state.get("should_halt"):
        raise HTTPException(400, "Run is not halted")

    t = threading.Thread(
        target=_resume_pipeline_thread,
        args=(run_id, effective_id, None),
        daemon=True,
    )
    t.start()

    return {"status": "resuming", "from_thread": effective_id}


@app.post("/api/runs/{run_id}/feedback")
async def feedback_run(run_id: str, req: FeedbackRequest):
    effective_id = _get_effective_run_id(run_id)
    state = _read_checkpoint_state(effective_id)
    if not state:
        raise HTTPException(404, f"Run '{run_id}' not found")

    if not state.get("should_halt"):
        raise HTTPException(400, "Run is not halted")

    t = threading.Thread(
        target=_resume_pipeline_thread,
        args=(run_id, effective_id, req.feedback),
        daemon=True,
    )
    t.start()

    return {"status": "resuming_with_feedback", "from_thread": effective_id}


@app.get("/api/runs/{run_id}/stream")
async def stream_run(run_id: str, request: Request):
    """SSE stream — polls checkpoint DB and emits state updates."""

    async def event_generator():
        last_sig = None
        while True:
            if await request.is_disconnected():
                break

            effective_id = _get_effective_run_id(run_id)
            state = _read_checkpoint_state(effective_id)

            if state:
                # Simple change detection via result/verdict count + halt state
                sig = (
                    len(state.get("results", [])),
                    len(state.get("gate_verdicts", [])),
                    state.get("current_phase", ""),
                    state.get("should_halt", False),
                    state.get("halt_reason", ""),
                    effective_id,
                )
                if sig != last_sig:
                    last_sig = sig
                    payload = json.dumps(_state_to_response(state))
                    yield f"data: {payload}\n\n"
            else:
                # No checkpoint yet — send a waiting status so the UI knows we're connected
                if last_sig is None:
                    waiting = json.dumps({
                        "run_id": effective_id,
                        "project_brief": "",
                        "phases": list_phases(),
                        "phase_statuses": {p: "pending" for p in list_phases()},
                        "current_phase": "",
                        "completed_phases": [],
                        "features": [],
                        "results_grouped": {},
                        "verdicts_grouped": {},
                        "messages": ["Pipeline starting..."],
                        "should_halt": False,
                        "halt_reason": "",
                        "total_results": 0,
                        "total_verdicts": 0,
                    })
                    yield f"data: {waiting}\n\n"
                    last_sig = "waiting"

            await asyncio.sleep(2)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/phases")
async def get_phases():
    return {"phases": list_phases()}
