"""FastAPI server for CMO Agent.

Provides REST + WebSocket API for:
- Chat sessions with Claude-backed CMO structuring agent
- Direct tool invocation (42 tools)
- Model registry (list, inspect, backtest)
- RL training launch + live progress via WebSocket
- Live market data snapshots
- Deal structuring via RL policy
"""
import asyncio
import json
import time
import uuid
import traceback
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


class ToolRequest(BaseModel):
    input: dict = {}


class TrainRequest(BaseModel):
    mode: str = "AGENCY"
    timesteps: int = 200_000
    model_name: Optional[str] = None


class BacktestRequest(BaseModel):
    n_deals: int = 100


class DealRequest(BaseModel):
    collateral_balance: float = 100_000_000
    agency: str = "GNMA"
    coupon: Optional[float] = None
    model_id: Optional[str] = None


# ---------------------------------------------------------------------------
# In-memory stores
# ---------------------------------------------------------------------------

SESSION_TTL = 30 * 60  # 30 minutes

# session_id -> {"agent": CMOAgent, "last_access": float}
_sessions: dict[str, dict] = {}

# run_id -> {"status": str, "progress": float, "metrics": dict, "error": str|None, "model_path": str|None, ...}
_training_runs: dict[str, dict] = {}

# model_id -> {"status": str, "result": dict|None, "error": str|None}
_backtest_runs: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def _get_or_create_session(session_id: Optional[str] = None) -> tuple:
    """Return (agent, session_id).  Reuses existing session or creates new one."""
    from .agent import CMOAgent

    now = time.time()

    # Purge expired sessions
    expired = [sid for sid, s in _sessions.items() if now - s["last_access"] > SESSION_TTL]
    for sid in expired:
        _sessions.pop(sid, None)

    if session_id and session_id in _sessions:
        _sessions[session_id]["last_access"] = now
        return _sessions[session_id]["agent"], session_id

    # Create new session
    new_id = session_id or str(uuid.uuid4())
    agent = CMOAgent()
    _sessions[new_id] = {"agent": agent, "last_access": now}
    return agent, new_id


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Scan for unregistered models on startup."""
    try:
        from .model_registry import scan_unregistered
        new_models = scan_unregistered()
        if new_models:
            print(f"[server] Auto-registered {len(new_models)} model(s): "
                  f"{[m.model_id for m in new_models]}")
        else:
            print("[server] Model registry up to date.")
    except Exception as exc:
        print(f"[server] Warning: model scan failed: {exc}")
    yield
    # Cleanup on shutdown
    _sessions.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CMO Agent API",
    description="Autonomous CMO structuring, pricing, and RL training server.",
    version="0.8.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "active_sessions": len(_sessions),
        "active_training_runs": sum(1 for r in _training_runs.values() if r["status"] == "running"),
    }


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Send a message to the CMO Agent and get a response (with tool use)."""
    agent, session_id = _get_or_create_session(req.session_id)
    try:
        # Run in executor to avoid blocking the event loop (Claude API is sync)
        loop = asyncio.get_running_loop()
        response_text = await loop.run_in_executor(None, agent.chat, req.message)
        return ChatResponse(response=response_text, session_id=session_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@app.get("/api/tools")
async def list_tools():
    """List all 42 agent tools with name + description."""
    from .agent import TOOLS
    return {
        "count": len(TOOLS),
        "tools": [
            {"name": t["name"], "description": t["description"]}
            for t in TOOLS
        ],
    }


@app.post("/api/tools/{tool_name}")
async def invoke_tool(tool_name: str, req: ToolRequest):
    """Directly invoke an agent tool by name."""
    from .agent import CMOAgent

    # Use a one-off agent for direct tool calls (no session needed)
    agent = CMOAgent()
    loop = asyncio.get_running_loop()
    try:
        raw = await loop.run_in_executor(None, agent.execute_tool, tool_name, req.input)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # Attempt to parse JSON so the response is structured
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        parsed = {"result": raw}

    if isinstance(parsed, dict) and "error" in parsed:
        raise HTTPException(status_code=400, detail=parsed)

    return parsed


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@app.get("/api/models")
async def list_models():
    """List all registered models."""
    from .model_registry import list_models as _list_models
    models = _list_models()
    return {"count": len(models), "models": models}


@app.get("/api/models/{model_id}")
async def get_model(model_id: str):
    """Get metadata for a single model."""
    from .model_registry import get_model as _get_model
    meta = _get_model(model_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found in registry")
    return meta


@app.post("/api/models/{model_id}/backtest")
async def trigger_backtest(model_id: str, req: BacktestRequest, background_tasks: BackgroundTasks):
    """Launch a backtest as a background task."""
    from .model_registry import get_model as _get_model

    meta = _get_model(model_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    # Check if already running
    existing = _backtest_runs.get(model_id)
    if existing and existing["status"] == "running":
        return {"model_id": model_id, "status": "already_running"}

    _backtest_runs[model_id] = {"status": "running", "result": None, "error": None}
    background_tasks.add_task(_run_backtest, model_id, meta["file_path"], req.n_deals)
    return {"model_id": model_id, "status": "started", "n_deals": req.n_deals}


def _run_backtest(model_id: str, model_path: str, n_deals: int):
    """Execute backtest in background thread."""
    import os
    from .backtest import backtest_agent_vs_experts, backtest_to_json
    from .model_registry import MODELS_DIR

    full_path = model_path
    if not os.path.isabs(full_path):
        full_path = os.path.join(MODELS_DIR, full_path)

    try:
        summary = backtest_agent_vs_experts(
            model_path=full_path,
            n_demos=n_deals,
            verbose=False,
        )
        result = backtest_to_json(summary)
        _backtest_runs[model_id] = {"status": "completed", "result": result, "error": None}
    except Exception as exc:
        _backtest_runs[model_id] = {"status": "failed", "result": None, "error": str(exc)}


@app.get("/api/models/{model_id}/backtest/status")
async def backtest_status(model_id: str):
    """Poll backtest status."""
    run = _backtest_runs.get(model_id)
    if run is None:
        raise HTTPException(status_code=404, detail="No backtest found for this model")
    return {"model_id": model_id, **run}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@app.post("/api/train")
async def launch_training(req: TrainRequest, background_tasks: BackgroundTasks):
    """Launch a PPO training run as a background task. Returns run_id for polling."""
    run_id = str(uuid.uuid4())[:12]
    _training_runs[run_id] = {
        "status": "starting",
        "progress": 0.0,
        "timestep": 0,
        "total_timesteps": req.timesteps,
        "mode": req.mode,
        "model_name": req.model_name,
        "metrics": {},
        "error": None,
        "model_path": None,
        "started_at": time.time(),
        "finished_at": None,
    }
    background_tasks.add_task(_run_training, run_id, req.mode, req.timesteps, req.model_name)
    return {"run_id": run_id, "status": "starting"}


def _run_training(run_id: str, mode: str, timesteps: int, model_name: Optional[str]):
    """Execute PPO training in a background thread."""
    import os
    from .yield_book_env import make_yield_book_env
    from .train import PPOTrainer, PPOConfig

    run = _training_runs[run_id]
    run["status"] = "running"

    try:
        env = make_yield_book_env(mode=mode)
        eval_env = make_yield_book_env(mode=mode)

        config = PPOConfig()
        config.total_timesteps = timesteps

        trainer = PPOTrainer(env=env, config=config, eval_env=eval_env)

        # Monkey-patch the trainer to report progress into our run dict.
        # PPOTrainer.train() logs internally; we hook into the env step count.
        _original_train = trainer.train

        def _patched_train():
            result = _original_train()
            return result

        # We cannot easily hook mid-loop, so we poll the env step_count in a
        # separate mechanism.  Instead, just run train() and capture the result.
        result = trainer.train()

        # Determine save path
        save_name = model_name or f"ppo_{mode.lower()}_{timesteps // 1000}k"
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        os.makedirs(models_dir, exist_ok=True)
        save_path = os.path.join(models_dir, f"{save_name}.pt")

        # Save checkpoint
        import torch
        torch.save({
            "policy_state_dict": trainer.policy.state_dict(),
            "config": {
                "mode": mode,
                "timesteps": timesteps,
                "obs_dim": env.observation_space.shape[0],
                "action_dims": list(env.action_space.nvec),
                "hidden_dim": config.hidden_dim,
            },
            "timestep": timesteps,
            "best_eval_reward": result.get("best_eval_reward", 0.0) if isinstance(result, dict) else 0.0,
        }, save_path)

        # Register in model registry
        from .model_registry import register_model, ModelMetadata
        meta = ModelMetadata(
            model_id=save_name,
            file_path=f"{save_name}.pt",
            obs_dim=env.observation_space.shape[0],
            action_dims=list(env.action_space.nvec),
            hidden_dim=config.hidden_dim,
            best_eval_reward=result.get("best_eval_reward", 0.0) if isinstance(result, dict) else 0.0,
            training_config={"mode": mode, "timesteps": timesteps},
            compatible_envs=[mode],
            timesteps=timesteps,
        )
        register_model(meta)

        run["status"] = "completed"
        run["progress"] = 1.0
        run["timestep"] = timesteps
        run["model_path"] = save_path
        run["metrics"] = result if isinstance(result, dict) else {"raw": str(result)}
        run["finished_at"] = time.time()

    except Exception as exc:
        run["status"] = "failed"
        run["error"] = str(exc)
        run["finished_at"] = time.time()


@app.get("/api/train/{run_id}/status")
async def training_status(run_id: str):
    """Poll training run status."""
    run = _training_runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Training run '{run_id}' not found")

    elapsed = None
    if run["started_at"]:
        end = run["finished_at"] or time.time()
        elapsed = round(end - run["started_at"], 1)

    return {
        "run_id": run_id,
        "status": run["status"],
        "progress": run["progress"],
        "timestep": run["timestep"],
        "total_timesteps": run["total_timesteps"],
        "mode": run["mode"],
        "metrics": run["metrics"],
        "error": run["error"],
        "model_path": run["model_path"],
        "elapsed_seconds": elapsed,
    }


@app.get("/api/train")
async def list_training_runs():
    """List all training runs."""
    runs = []
    for run_id, run in _training_runs.items():
        elapsed = None
        if run["started_at"]:
            end = run["finished_at"] or time.time()
            elapsed = round(end - run["started_at"], 1)
        runs.append({
            "run_id": run_id,
            "status": run["status"],
            "progress": run["progress"],
            "mode": run["mode"],
            "elapsed_seconds": elapsed,
        })
    return {"count": len(runs), "runs": runs}


# ---------------------------------------------------------------------------
# WebSocket: live training metrics
# ---------------------------------------------------------------------------

@app.websocket("/ws/train/{run_id}")
async def ws_training(websocket: WebSocket, run_id: str):
    """WebSocket endpoint for live training progress.

    Client connects and receives periodic JSON updates until the run
    finishes or the client disconnects.
    """
    await websocket.accept()

    run = _training_runs.get(run_id)
    if run is None:
        await websocket.send_json({"error": f"Training run '{run_id}' not found"})
        await websocket.close()
        return

    try:
        prev_snapshot = None
        while True:
            run = _training_runs.get(run_id)
            if run is None:
                await websocket.send_json({"error": "run disappeared"})
                break

            snapshot = {
                "run_id": run_id,
                "status": run["status"],
                "progress": run["progress"],
                "timestep": run["timestep"],
                "total_timesteps": run["total_timesteps"],
                "metrics": run["metrics"],
                "error": run["error"],
                "model_path": run["model_path"],
            }

            # Only send when something changed
            if snapshot != prev_snapshot:
                await websocket.send_json(snapshot)
                prev_snapshot = snapshot

            if run["status"] in ("completed", "failed"):
                # Send final update then close
                break

            await asyncio.sleep(2.0)

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Market Data
# ---------------------------------------------------------------------------

@app.get("/api/market/snapshot")
async def market_snapshot():
    """Get a comprehensive live market data snapshot."""
    from .data_sources import get_full_market_snapshot

    loop = asyncio.get_running_loop()
    try:
        snapshot = await loop.run_in_executor(None, get_full_market_snapshot)
        return snapshot
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Market data fetch failed: {exc}")


@app.get("/api/market/treasury-curve")
async def treasury_curve():
    """Get the current live Treasury yield curve."""
    from .data_sources import build_live_treasury_curve

    loop = asyncio.get_running_loop()
    try:
        curve = await loop.run_in_executor(None, build_live_treasury_curve)
        return curve
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Treasury curve fetch failed: {exc}")


@app.get("/api/market/mortgage-rates")
async def mortgage_rates():
    """Get current mortgage rates."""
    from .data_sources import fetch_current_mortgage_rates

    loop = asyncio.get_running_loop()
    try:
        rates = await loop.run_in_executor(None, fetch_current_mortgage_rates)
        return rates
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Mortgage rate fetch failed: {exc}")


@app.get("/api/market/sources")
async def data_source_status():
    """Check connectivity to all market data sources."""
    from .data_sources import check_all_sources

    loop = asyncio.get_running_loop()
    sources = await loop.run_in_executor(None, check_all_sources)
    return {
        "sources": [
            {
                "name": s.name,
                "available": s.available,
                "last_fetch": s.last_fetch,
                "record_count": s.record_count,
                "error": s.error,
            }
            for s in sources
        ]
    }


# ---------------------------------------------------------------------------
# Deal Structuring
# ---------------------------------------------------------------------------

@app.post("/api/deal/structure")
async def structure_deal(req: DealRequest):
    """Use the RL agent (or Claude agent) to suggest an optimal deal structure.

    If a model_id is provided, loads that RL policy and runs it through the
    YieldBook environment for one episode.  Otherwise, uses the Claude-based
    CMOAgent in autonomous mode.
    """
    loop = asyncio.get_running_loop()

    if req.model_id:
        # RL-based structuring
        try:
            result = await loop.run_in_executor(
                None, _rl_structure_deal, req.model_id, req.collateral_balance, req.agency, req.coupon
            )
            return result
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
    else:
        # Claude agent autonomous structuring
        try:
            result = await loop.run_in_executor(
                None, _agent_structure_deal, req.collateral_balance, req.agency, req.coupon
            )
            return result
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))


def _rl_structure_deal(model_id: str, collateral_balance: float, agency: str, coupon: Optional[float]) -> dict:
    """Run an RL policy through one episode and return the deal structure."""
    import torch
    import numpy as np
    from .model_registry import load_model
    from .yield_book_env import make_yield_book_env

    policy, meta = load_model(model_id)

    mode = agency.upper()
    if mode not in ("AGENCY", "GNMA", "FNMA", "FHLMC", "GSE", "CMBS"):
        mode = "AGENCY"

    env = make_yield_book_env(mode=mode, collateral_balance=collateral_balance)
    obs, info = env.reset()

    actions_taken = []
    total_reward = 0.0
    step = 0
    done = False

    while not done and step < 50:
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        actions, _, _ = policy.get_action(obs_tensor, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(actions)
        total_reward += reward
        actions_taken.append(actions)
        done = terminated or truncated
        step += 1

    # Extract deal info from environment
    deal_info = info.get("deal", {}) if isinstance(info, dict) else {}

    return {
        "model_id": model_id,
        "mode": mode,
        "collateral_balance": collateral_balance,
        "total_reward": round(total_reward, 4),
        "steps": step,
        "actions": actions_taken,
        "deal": deal_info,
    }


def _agent_structure_deal(collateral_balance: float, agency: str, coupon: Optional[float]) -> dict:
    """Use Claude CMO Agent in autonomous mode to structure a deal."""
    from .agent import CMOAgent

    agent = CMOAgent()

    coupon_str = f" at {coupon}% coupon" if coupon else ""
    objective = (
        f"Structure an optimal CMO deal using ${collateral_balance:,.0f} of {agency} collateral{coupon_str}. "
        f"Buy spec pools, create the REMIC structure, price all tranches, compute structuring P&L, "
        f"and recommend whether to execute."
    )

    response = agent.run_autonomous(objective)
    return {
        "mode": "autonomous_agent",
        "agency": agency,
        "collateral_balance": collateral_balance,
        "coupon": coupon,
        "response": response,
    }


# ---------------------------------------------------------------------------
# Sessions management (admin)
# ---------------------------------------------------------------------------

@app.get("/api/sessions")
async def list_sessions():
    """List active chat sessions."""
    now = time.time()
    sessions = []
    for sid, s in _sessions.items():
        age = now - s["last_access"]
        sessions.append({
            "session_id": sid,
            "idle_seconds": round(age, 1),
            "history_length": len(s["agent"].conversation_history),
        })
    return {"count": len(sessions), "sessions": sessions}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    _sessions.pop(session_id)
    return {"deleted": session_id}


# ---------------------------------------------------------------------------
# Convenience: run with uvicorn
# ---------------------------------------------------------------------------

def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the server using uvicorn."""
    import uvicorn
    uvicorn.run(
        "cmo_agent.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    run_server()
