"""
LangGraph graph construction — durable, parallel, long-lived pipeline.

Topology:

  START
    |
  decompose  (supervisor splits brief into features)
    |
  phase_discover --> gate_discover --+
                                     | (pass)
  phase_strategize --> gate_strategize --+
                                         |
  phase_scaffold --> gate_scaffold --+
                                      |
  phase_build --> gate_build --+
                                |
  phase_harden --> gate_harden --+
                                  |
  phase_launch --> gate_launch --+
                                  |
  phase_operate --> gate_operate --> END

  Any gate failure --> halt --> END

Features run in parallel WITHIN each phase node (via asyncio).
State is checkpointed after every node for durability.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from .agents import list_phases
from .config import PipelineConfig
from .nodes import make_decompose_node, make_gate_node, make_phase_node
from .state import PipelineState


def _gate_router(state: PipelineState) -> str:
    if state.get("should_halt"):
        return "halt"
    return "continue"


def build_graph(config: PipelineConfig, checkpointer=None):
    """
    Construct and compile the full NEXUS pipeline graph.

    If *checkpointer* is provided, the graph persists state to it (SQLite).
    Otherwise, runs without persistence.
    """
    phases = list_phases()
    graph = StateGraph(PipelineState)

    # --- Decompose node (supervisor breaks brief into features) ---
    graph.add_node("decompose", make_decompose_node(config))

    # --- Phase + gate nodes ---
    for phase in phases:
        graph.add_node(f"phase_{phase}", make_phase_node(phase, config))
        graph.add_node(f"gate_{phase}", make_gate_node(phase, config))

    # --- Halt node ---
    def halt_node(state: PipelineState) -> dict:
        return {"messages": ["HALT: Pipeline stopped - quality gate failed. Resume with --resume."]}
    graph.add_node("halt", halt_node)

    # --- Wiring ---
    graph.add_edge(START, "decompose")
    graph.add_edge("decompose", f"phase_{phases[0]}")

    for i, phase in enumerate(phases):
        graph.add_edge(f"phase_{phase}", f"gate_{phase}")

        if i < len(phases) - 1:
            next_phase = phases[i + 1]
            graph.add_conditional_edges(
                f"gate_{phase}",
                _gate_router,
                {"continue": f"phase_{next_phase}", "halt": "halt"},
            )
        else:
            graph.add_conditional_edges(
                f"gate_{phase}",
                _gate_router,
                {"continue": END, "halt": "halt"},
            )

    graph.add_edge("halt", END)

    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Durable graph factory with SQLite checkpointing
# ---------------------------------------------------------------------------

DEFAULT_DB_PATH = ".nexus-pipeline/checkpoints.db"


def get_checkpointer(db_path: str | None = None) -> SqliteSaver:
    """Create a SQLite checkpointer, ensuring the directory exists."""
    path = db_path or DEFAULT_DB_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(conn)


def build_durable_graph(config: PipelineConfig, db_path: str | None = None):
    """Build a graph with SQLite-backed durable checkpointing."""
    checkpointer = get_checkpointer(db_path)
    return build_graph(config, checkpointer=checkpointer), checkpointer
