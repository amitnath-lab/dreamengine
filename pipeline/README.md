# DreamEngine NEXUS Pipeline

A **durable, parallel, long-lived** AI development pipeline powered by LangGraph.
Orchestrates all 150+ DreamEngine agent specs through a 7-phase NEXUS workflow.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   Claude Opus (Supervisor)                       │
│  Decomposes features · Plans phases · Routes models · Gates QA  │
└──────────┬──────────────────────────────────────────────────────┘
           │
     ┌─────┴─────┐
     │ LangGraph  │──── SQLite checkpoint (durable state)
     │   FSM      │
     └─────┬─────┘
           │
  decompose (split brief into features)
           │
           ├──────────────────────────────────────┐
           │        Features run in PARALLEL       │
           │     within each phase via asyncio     │
           ├──────────────────────────────────────┘
           │
  discover ──► gate ──► strategize ──► gate ──► scaffold ──► gate
                                                               │
  build ──► gate ──► harden ──► gate ──► launch ──► gate ──► operate ──► gate ──► END
           │
           │ (gate fails)
           └──► HALT  (resume later with --resume)
           │
     ┌─────┴─────┐
     │   Ollama   │  (local workers — free models)
     │            │
     │  qwen2.5   │  ← fast tier (most tasks)
     │  llama3.1  │  ← fast tier
     │  deepseek  │  ← code tasks
     │  mixtral   │  ← heavy tier (rare)
     └───────────┘
```

## Key Capabilities

| Capability | How |
|---|---|
| **Parallel features** | Supervisor decomposes brief into independent features; all execute concurrently within each phase via `asyncio` |
| **Durable / long-lived** | SQLite checkpointing after every node — crash, close terminal, come back tomorrow |
| **Resumable** | `--resume run-id` picks up exactly where it stopped |
| **Model routing** | Supervisor picks cheapest viable Ollama model per task; free_fast handles ~80% |
| **Quality gates** | Supervisor reviews every phase; failed gate halts pipeline (resume after fixing) |
| **130+ agent personas** | Full DreamEngine specs used as system prompts |

## Model Tiers (cheapest first)

| Tier | Models | Used For |
|------|--------|----------|
| `free_fast` | qwen2.5:7b, llama3.1:8b, gemma2:9b, phi3:mini, deepseek-coder-v2:16b | ~80% of tasks |
| `free_medium` | qwen2.5:32b, llama3.1:70b | Complex architecture, deep analysis |
| `free_heavy` | deepseek-coder-v2:236b, mixtral:8x22b | Security audits, heavy reasoning |
| `paid_supervisor` | claude-opus-4-6 | Orchestration decisions only |

## Setup

```bash
# 1. Install
pip install -r pipeline/requirements.txt

# 2. Pull Ollama models (at minimum the fast tier)
ollama pull qwen2.5:7b
ollama pull llama3.1:8b
ollama pull deepseek-coder-v2:16b

# 3. Set supervisor API key
export ANTHROPIC_API_KEY=sk-ant-...

# 4. Verify
python -m pipeline --check-ollama
```

## Usage

```bash
# --- Start a new run ---
python -m pipeline --brief "Build a SaaS task manager with auth, billing, and team features"

# From a file
python -m pipeline --brief-file project-brief.txt

# --- Durability ---
# If the run halts (gate failure) or you kill the process:
python -m pipeline --resume run-a1b2c3d4

# Check status of any run
python -m pipeline --status run-a1b2c3d4

# List all runs in the database
python -m pipeline --list-runs

# --- Save & debug ---
python -m pipeline --brief "..." --output results.json
python -m pipeline --brief "..." --verbose

# --- Diagnostics ---
python -m pipeline --list-agents    # All 130+ agents by phase
python -m pipeline --list-models    # Model tier config
python -m pipeline --check-ollama   # Verify connectivity + pulled models
```

## Pipeline Phases (NEXUS)

| # | Phase | Purpose | Key Agents |
|---|-------|---------|------------|
| 0 | **Discover** | Market / user / tech intelligence | Trend Researcher, UX Researcher, Tool Evaluator |
| 1 | **Strategize** | Architecture & planning | Software Architect, Sprint Prioritizer, UX Architect |
| 2 | **Scaffold** | Foundation setup | DevOps Automator, Frontend / Backend devs |
| 3 | **Build** | Dev + QA continuous loop | All engineering + testing agents |
| 4 | **Harden** | Security, perf, accessibility | Reality Checker, Security Engineer, Benchmarker |
| 5 | **Launch** | Go-to-market | Content, SEO, Social Media, Growth agents |
| 6 | **Operate** | Sustained operations | SRE, Support, Analytics, Compliance agents |

## How It Works

1. **Decompose**: Supervisor breaks the brief into independent, parallelisable features with dependency ordering
2. **Phase loop**: For each of 7 phases:
   - Supervisor plans tasks for each feature (which agents, what work, which model tier)
   - All features execute **in parallel** via `asyncio`
   - Within each feature, tasks in the same `parallel_group` run concurrently
   - Supervisor reviews outputs and issues pass/fail quality gate
3. **Checkpoint**: State is persisted to SQLite after every node — the run survives crashes
4. **Resume**: `--resume` reloads the checkpoint and continues from the last completed node

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | (required) | Claude Opus supervisor |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `--db-path` | `.nexus-pipeline/checkpoints.db` | Checkpoint database |

Edit [config.py](config.py) to add/remove models or adjust task-strength mappings.

## File Structure

```
pipeline/
├── __init__.py          # Package exports
├── __main__.py          # python -m pipeline
├── main.py              # CLI: run, resume, status, list-runs, diagnostics
├── config.py            # Model tiers, task→strength maps, agent→task maps
├── state.py             # LangGraph state: features, results, gates, run identity
├── agents.py            # Loads 130+ agent specs, maps to NEXUS phases
├── models.py            # Sync + async Ollama/Anthropic clients, model picker
├── supervisor.py        # Claude Opus: decompose, plan, gate, route
├── graph.py             # LangGraph FSM: 16 nodes, SQLite checkpointer
└── requirements.txt     # langgraph, langgraph-checkpoint-sqlite, httpx
```
