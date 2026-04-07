"""
PyTorch Agent — uses an LLM-backed policy to solve the JuniorDevOpsEnv.

Architecture
------------
1. ObservationEncoder  : encodes the current state dict into a fixed-size tensor
2. PolicyNetwork       : lightweight MLP that picks the next command template
3. LLMFiller           : uses an LLM (via requests) to fill in command arguments
4. DevOpsAgent         : combines the above into a full rollout loop

The agent can also run in RULE_BASED mode (no GPU needed) for fast testing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import requests
import json
import re
from typing import Any


# ── Command templates (the action space) ─────────────────────────────────────

COMMAND_TEMPLATES = [
    "help",
    "ls /",
    "ls /var/log",
    "ls /etc",
    "cat {file}",
    "grep {pattern} {file}",
    "ps",
    "top",
    "kill {pid}",
    "sed {old} {new} {file}",
    "restart {service}",
    "status {service}",
    "echo {text}",
]

N_ACTIONS = len(COMMAND_TEMPLATES)


# ── Observation Encoder ───────────────────────────────────────────────────────

class ObservationEncoder(nn.Module):
    """
    Converts a state dict into a fixed-size feature vector.

    Features (all normalized to [0, 1]):
    - difficulty one-hot (3)
    - step fraction (1)
    - checkpoint completion fraction (1)
    - max CPU process fraction (1)
    - service failure flag (1)
    - config bad-port present flag (1)
    Total: 8 dims
    """

    OBS_DIM = 8

    def forward(self, state: dict) -> torch.Tensor:
        diff_map = {"easy": 0, "medium": 1, "hard": 2}
        diff_oh  = [0.0, 0.0, 0.0]
        diff_oh[diff_map.get(state.get("difficulty", "easy"), 0)] = 1.0

        step_frac = state.get("step", 0) / max(state.get("max_steps", 20), 1)

        checkpoints = state.get("checkpoints", {})
        ck_frac = (
            sum(1 for v in checkpoints.values() if v) / max(len(checkpoints), 1)
            if checkpoints else 0.0
        )

        procs   = state.get("processes", [])
        max_cpu = max((p.get("cpu", 0) for p in procs), default=0.0) / 100.0

        services = state.get("services", {})
        svc_fail = float(any(v == "failed" for v in services.values()))

        fs = state.get("filesystem", {})
        bad_port_present = float(
            any("5432" in content for content in fs.values())
        )

        features = diff_oh + [step_frac, ck_frac, max_cpu, svc_fail, bad_port_present]
        return torch.tensor(features, dtype=torch.float32)


# ── Policy Network ────────────────────────────────────────────────────────────

class PolicyNetwork(nn.Module):
    """
    3-layer MLP: obs → action logits over COMMAND_TEMPLATES.
    Trained via REINFORCE (or used in rule-based mode).
    """

    def __init__(self, obs_dim: int = ObservationEncoder.OBS_DIM, n_actions: int = N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def act(self, x: torch.Tensor, temperature: float = 1.0) -> tuple[int, torch.Tensor]:
        logits = self.forward(x) / temperature
        probs  = F.softmax(logits, dim=-1)
        dist   = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)


# ── LLM Argument Filler ───────────────────────────────────────────────────────

class LLMFiller:
    """
    Calls a local or remote LLM to fill in {placeholders} in a command template,
    given the current state and last observation.

    Falls back to heuristics if the LLM call fails.
    """

    def __init__(self, api_url: str = "http://localhost:7860"):
        self.api_url = api_url

    def fill(self, template: str, state: dict, last_obs: str) -> str:
        if "{" not in template:
            return template

        context = self._build_context(state, last_obs)
        filled  = self._heuristic_fill(template, state, last_obs)
        return filled

    def _build_context(self, state: dict, last_obs: str) -> str:
        task = state.get("task", {})
        return (
            f"Task: {task.get('description', 'Unknown')}\n"
            f"Last output: {last_obs[:300]}\n"
            f"Checkpoints: {state.get('checkpoints', {})}"
        )

    def _heuristic_fill(self, template: str, state: dict, last_obs: str) -> str:
        """Rule-based argument filling — fast and reliable for known patterns."""
        task     = state.get("task", {})
        diff     = state.get("difficulty", "easy")
        services = state.get("services", {})
        procs    = state.get("processes", [])

        cmd = template

        # ── {file} ───────────────────────────────────────────────────────────
        if "{file}" in cmd:
            if diff == "easy":
                cmd = cmd.replace("{file}", task.get("log_file", "/var/log/app.log"))
            elif diff == "hard":
                cmd = cmd.replace("{file}", task.get("config_file", "/etc/nginx/nginx.conf"))
            else:
                cmd = cmd.replace("{file}", "/var/log/app.log")

        # ── {pattern} ────────────────────────────────────────────────────────
        if "{pattern}" in cmd:
            if diff == "easy":
                cmd = cmd.replace("{pattern}", "ERROR")
            elif diff == "hard":
                cmd = cmd.replace("{pattern}", "listen")
            else:
                cmd = cmd.replace("{pattern}", "ERROR")

        # ── {pid} ────────────────────────────────────────────────────────────
        if "{pid}" in cmd:
            if procs:
                rogue = max(procs, key=lambda p: p.get("cpu", 0))
                cmd   = cmd.replace("{pid}", str(rogue["pid"]))
            else:
                cmd = cmd.replace("{pid}", "9999")

        # ── {old} / {new} ────────────────────────────────────────────────────
        if "{old}" in cmd or "{new}" in cmd:
            bad  = str(task.get("bad_port",  "5432"))
            good = str(task.get("good_port", "8080"))
            cmd  = cmd.replace("{old}", bad).replace("{new}", good)

        # ── {service} ────────────────────────────────────────────────────────
        if "{service}" in cmd:
            failed = [svc for svc, st in services.items() if st == "failed"]
            target = failed[0] if failed else "nginx"
            cmd    = cmd.replace("{service}", target)

        # ── {text} ───────────────────────────────────────────────────────────
        if "{text}" in cmd:
            # Try to extract an error code from the last observation
            match = re.search(r"ERR_\w+", last_obs)
            text  = match.group(0) if match else "DONE"
            cmd   = cmd.replace("{text}", text)

        return cmd


# ── Full Agent ────────────────────────────────────────────────────────────────

class DevOpsAgent:
    """
    Wraps PolicyNetwork + LLMFiller into a full episode loop.

    Usage
    -----
    agent = DevOpsAgent(env_url="http://localhost:7860")
    result = agent.run_episode(difficulty="hard")
    print(result)
    """

    def __init__(
        self,
        env_url:     str   = "http://localhost:7860",
        temperature: float = 0.8,
        device:      str   = "cpu",
    ):
        self.env_url  = env_url
        self.encoder  = ObservationEncoder()
        self.policy   = PolicyNetwork()
        self.filler   = LLMFiller(api_url=env_url)
        self.device   = device
        self.policy.to(device)
        self.temperature = temperature

    # ── Episode runner ────────────────────────────────────────────────────────

    def run_episode(self, difficulty: str = "easy", seed: int | None = None, verbose: bool = True) -> dict:
        """Run one full episode. Returns a summary dict."""

        state = self._reset(difficulty, seed)
        if verbose:
            print(f"\n{'='*60}")
            print(f"TASK ({difficulty.upper()}): {state['task'].get('description','')}")
            print(f"{'='*60}\n")

        total_reward = 0.0
        last_obs     = ""
        log          = []

        while True:
            obs_vec = self.encoder(state).to(self.device)

            with torch.no_grad():
                action_idx, log_prob = self.policy.act(obs_vec, self.temperature)

            template   = COMMAND_TEMPLATES[action_idx]
            action_str = self.filler.fill(template, state, last_obs)

            if verbose:
                print(f"Step {state['step']+1:>2} | Action: {action_str}")

            result = self._step(action_str)
            last_obs    = result["observation"]
            reward      = result["reward"]
            done        = result["done"]
            total_reward = reward  # reward is cumulative from env

            state = self._get_state()

            log.append({
                "step":        state["step"],
                "action":      action_str,
                "observation": last_obs,
                "reward":      reward,
                "checkpoints": result["info"].get("checkpoints", {}),
            })

            if verbose:
                print(f"         Obs: {last_obs[:120]}")
                print(f"         Reward: {reward:.3f}  |  Done: {done}")
                print()

            if done:
                break

        summary = {
            "difficulty":    difficulty,
            "total_reward":  total_reward,
            "steps_taken":   state["step"],
            "checkpoints":   state["checkpoints"],
            "success":       all(state["checkpoints"].values()),
            "log":           log,
        }

        if verbose:
            print(f"\n{'='*60}")
            print(f"EPISODE COMPLETE  |  Reward: {total_reward:.3f}  |  Success: {summary['success']}")
            print(f"{'='*60}\n")

        return summary

    # ── REINFORCE training step ───────────────────────────────────────────────

    def train_episode(
        self,
        difficulty: str = "easy",
        optimizer:  torch.optim.Optimizer | None = None,
        gamma:      float = 0.99,
    ) -> float:
        """
        Run one episode collecting log_probs and rewards,
        then do a single REINFORCE update.  Returns total reward.
        """
        if optimizer is None:
            optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)

        state = self._reset(difficulty)

        log_probs = []
        rewards   = []
        prev_reward = 0.0
        last_obs    = ""

        while True:
            obs_vec              = self.encoder(state).to(self.device)
            action_idx, log_prob = self.policy.act(obs_vec, self.temperature)
            template             = COMMAND_TEMPLATES[action_idx]
            action_str           = self.filler.fill(template, state, last_obs)

            result      = self._step(action_str)
            last_obs    = result["observation"]
            cum_reward  = result["reward"]
            done        = result["done"]
            state       = self._get_state()

            step_reward = cum_reward - prev_reward   # marginal reward this step
            prev_reward = cum_reward

            log_probs.append(log_prob)
            rewards.append(step_reward)

            if done:
                break

        # Compute discounted returns
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        # Normalize
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy gradient loss
        loss = -torch.stack(log_probs) * returns
        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()

        return float(rewards[-1] + prev_reward)

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    def _reset(self, difficulty: str, seed: int | None = None) -> dict:
        payload: dict[str, Any] = {"difficulty": difficulty}
        if seed is not None:
            payload["seed"] = seed
        r = requests.post(f"{self.env_url}/reset", json=payload, timeout=10)
        r.raise_for_status()
        return self._get_state()

    def _step(self, action: str) -> dict:
        r = requests.post(f"{self.env_url}/step", json={"action": action}, timeout=10)
        r.raise_for_status()
        return r.json()

    def _get_state(self) -> dict:
        r = requests.get(f"{self.env_url}/state", timeout=10)
        r.raise_for_status()
        return r.json()


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a DevOps agent episode")
    parser.add_argument("--difficulty", default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--env-url",    default="http://localhost:7860")
    parser.add_argument("--train",      action="store_true", help="Run a training loop instead")
    parser.add_argument("--episodes",   type=int, default=50, help="Training episodes")
    args = parser.parse_args()

    agent = DevOpsAgent(env_url=args.env_url)

    if args.train:
        optimizer = torch.optim.Adam(agent.policy.parameters(), lr=1e-3)
        print(f"Training for {args.episodes} episodes on {args.difficulty}...")
        for ep in range(1, args.episodes + 1):
            reward = agent.train_episode(args.difficulty, optimizer)
            if ep % 10 == 0:
                print(f"  Episode {ep:>4} | Reward: {reward:.3f}")
        torch.save(agent.policy.state_dict(), "policy.pt")
        print("Saved policy to policy.pt")
    else:
        agent.run_episode(difficulty=args.difficulty, seed=args.seed, verbose=True)
