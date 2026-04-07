"""
FastAPI server — exposes JuniorDevOpsEnv over HTTP (OpenEnv-compatible).

Endpoints
---------
POST /reset          Reset the environment (body: {"difficulty": "easy|medium|hard", "seed": int})
POST /step           Take an action     (body: {"action": "..."})
GET  /state          Read current state
GET  /health         Health check
GET  /docs           Auto-generated Swagger UI
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from devops_env import JuniorDevOpsEnv, Difficulty

app = FastAPI(
    title="Junior DevOps Environment",
    description=(
        "A stateful Linux-server simulation for AI agents. "
        "The agent diagnoses and fixes server problems using shell-like commands."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session store (single environment per server instance) ──────────
app.state.env: JuniorDevOpsEnv | None = None


# ── Request / Response schemas ────────────────────────────────────────────────

class ResetRequest(BaseModel):
    difficulty: str = "easy"   # "easy" | "medium" | "hard"
    seed: int | None = None

class StepRequest(BaseModel):
    action: str                 # Shell command string, e.g. "cat /var/log/app.log"

class StepResponse(BaseModel):
    observation: str
    reward: float
    done: bool
    info: dict


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "env_initialized": app.state.env is not None}


@app.post("/reset")
def reset(req: ResetRequest):
    try:
        diff = Difficulty(req.difficulty.lower())
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid difficulty '{req.difficulty}'. Choose from: easy, medium, hard"
        ) from exc

    kwargs = {"difficulty": diff}
    if req.seed is not None:
        kwargs["seed"] = req.seed

    app.state.env = JuniorDevOpsEnv(**kwargs)
    s = app.state.env.state()

    return {
        "message":     "Environment reset.",
        "difficulty":  s["difficulty"],
        "seed":        app.state.env.seed,
        "task":        s["task"],
        "checkpoints": s["checkpoints"],
    }


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    if app.state.env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call POST /reset first.")

    result = app.state.env.step(req.action)
    return StepResponse(
        observation=result.observation,
        reward=result.reward,
        done=result.done,
        info=result.info,
    )


@app.get("/state")
def state():
    if app.state.env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call POST /reset first.")
    return app.state.env.state()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False)
