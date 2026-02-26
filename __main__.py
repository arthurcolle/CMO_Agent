"""Entry point for running cmo_agent as a module: python -m cmo_agent"""
import sys


def cli():
    """Route to the right entry point based on first argument."""
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        # python -m cmo_agent train [--mode GSE] [--timesteps 200000] [--benchmark]
        sys.argv = sys.argv[1:]  # Strip 'train' from args
        from .train import __name__ as _  # noqa - triggers __main__ block
        import runpy
        runpy.run_module("cmo_agent.train", run_name="__main__")
    elif len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        from .train import benchmark_agents
        benchmark_agents()
    elif len(sys.argv) > 1 and sys.argv[1] == "serve":
        # python -m cmo_agent serve [--host 0.0.0.0] [--port 8000] [--reload]
        import argparse
        parser = argparse.ArgumentParser(prog="cmo_agent serve", description="Run CMO Agent API server")
        parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
        parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
        parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
        args = parser.parse_args(sys.argv[2:])
        from .server import run_server
        run_server(host=args.host, port=args.port, reload=args.reload)
    elif len(sys.argv) > 1 and sys.argv[1] == "dashboard":
        # python -m cmo_agent dashboard [-- streamlit args]
        import os
        import subprocess
        dashboard_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.py")
        if not os.path.exists(dashboard_path):
            print(f"Error: dashboard.py not found at {dashboard_path}")
            sys.exit(1)
        extra_args = sys.argv[2:]
        cmd = [sys.executable, "-m", "streamlit", "run", dashboard_path] + extra_args
        subprocess.run(cmd)
    else:
        from .agent import main
        main()


if __name__ == "__main__":
    cli()
