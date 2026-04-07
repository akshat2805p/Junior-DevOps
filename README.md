---
title: Junior DevOps Environment
emoji: 🖥️
colorFrom: indigo
colorTo: cyan
sdk: docker
pinned: false
license: mit
app_port: 7860
short_description: OpenEnv-compatible Linux server simulation for AI agents
---

# 🖥️ Junior DevOps Environment

> **Meta PyTorch Hackathon Submission** — an OpenEnv-compatible real-world simulation
> where an AI agent must diagnose and fix a broken Linux server.

---

## What Is This?

A stateful simulation of a production Linux server. An AI agent interacts with it
using shell-like commands (`cat`, `grep`, `ps`, `kill`, `sed`, `restart`, …) to
complete progressively harder system administration tasks.

The environment implements the **OpenEnv API contract**:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `reset()` | `POST /reset` | Initialize/reset the environment |
| `step()` | `POST /step` | Execute one action, get observation + reward |
| `state()` | `GET /state` | Read the full current state |

---

## Task Difficulties

### 🟢 Easy — Find the Error Code
The app is crashing. Locate the error code buried in `/var/log/app.log`.

**Optimal solution:**
```bash
cat /var/log/app.log
grep ERROR /var/log/app.log
echo ERR_502
```
**Reward:** `0.4` for opening the file · `1.0` for reporting the code

---

### 🟡 Medium — Kill the CPU Hog
A rogue process is consuming 90%+ CPU and grinding the server to a halt.

**Optimal solution:**
```bash
top
kill 9999
```
**Reward:** `0.3` for inspecting processes · `1.0` for killing the rogue process

---

### 🔴 Hard — Fix the Port Conflict
`nginx` is `failed` because its config listens on port `5432` (PostgreSQL's port).
Fix it and restart nginx.

**Optimal solution:**
```bash
cat /etc/nginx/nginx.conf
sed 5432 8080 /etc/nginx/nginx.conf
restart nginx
```
**Reward:** `0.2` read · `0.4` identified · `0.7` fixed · `1.0` restarted

---

## Quick Start

### 1. Reset the Environment

```bash
curl -X POST https://your-space.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty": "hard", "seed": 42}'
```

### 2. Take an Action

```bash
curl -X POST https://your-space.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action": "cat /etc/nginx/nginx.conf"}'
```

Response:
```json
{
  "observation": "worker_processes auto;\nevents { worker_connections 1024; }\nhttp {\n    listen 5432;\n    ...",
  "reward": 0.2,
  "done": false,
  "info": {
    "step": 1,
    "checkpoints": {
      "read_config": true,
      "identified_conflict": false,
      "fixed_config": false,
      "restarted_nginx": false
    }
  }
}
```

### 3. Read Current State

```bash
curl https://your-space.hf.space/state
```

---

## Running the AI Agent

```bash
# Install dependencies
pip install -r requirements.txt

# Solve easy task
python agent.py --difficulty easy --env-url http://localhost:7860

# Train via REINFORCE for 100 episodes
python agent.py --train --difficulty hard --episodes 100
```

---

## Architecture

```
┌─────────────────────────────────────┐
│         DevOps Agent (PyTorch)      │
│  ┌─────────────┐  ┌──────────────┐  │
│  │ Observation │  │   Policy     │  │
│  │  Encoder   │→ │  Network     │  │
│  │  (8-dim)   │  │  (MLP 8→64  │  │
│  └─────────────┘  │   →64→13)  │  │
│                   └──────┬───────┘  │
│              action_idx  │          │
│                   ┌──────▼───────┐  │
│                   │  LLM Filler  │  │
│                   │  (heuristic) │  │
│                   └──────┬───────┘  │
└──────────────────────────┼──────────┘
                    action │ (shell cmd)
                   ┌───────▼──────────┐
                   │  JuniorDevOpsEnv │
                   │  FastAPI Server  │
                   │  POST /step      │
                   └──────────────────┘
```

---

## Grading

| Checkpoint | Easy | Medium | Hard |
|-----------|------|--------|------|
| Step 1    | +0.4 | +0.30  | +0.20 |
| Step 2    | +0.6 | +0.30  | +0.20 |
| Step 3    | —    | +0.40  | +0.30 |
| Step 4    | —    | —      | +0.30 |

Rewards are **cumulative and partial** — no binary 0/1 scoring.

---

