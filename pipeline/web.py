#!/usr/bin/env python3
"""
DreamEngine NEXUS Pipeline — Web UI entry point.

Usage:
    python -m pipeline.web
    python -m pipeline.web --port 8080
    python -m pipeline.web --host 0.0.0.0 --port 8000
"""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="NEXUS Pipeline Web Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    print(f"\n  NEXUS Pipeline Dashboard")
    print(f"  http://{args.host}:{args.port}\n")

    uvicorn.run(
        "pipeline.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
