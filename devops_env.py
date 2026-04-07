"""
Junior DevOps Environment — OpenEnv-Compatible
================================================
A stateful simulation of a broken Linux server.
The agent must diagnose and fix issues using terminal-like commands.

Implements: reset(), step(), state() — required by OpenEnv API
"""

import random
import re
import copy
import time
from dataclasses import dataclass, field
from typing import Any
from enum import Enum


# ── Task difficulty levels ────────────────────────────────────────────────────

class Difficulty(str, Enum):
    EASY   = "easy"    # Find error code in a log file
    MEDIUM = "medium"  # Identify + kill CPU-hogging process
    HARD   = "hard"    # Fix port conflict in config, restart service


# ── Grading result ────────────────────────────────────────────────────────────

@dataclass
class StepResult:
    observation: str        # What the agent sees after the action
    reward:      float      # Partial reward [0.0, 1.0]
    done:        bool       # Episode finished?
    info:        dict       # Debug / judge metadata


# ── The Environment ───────────────────────────────────────────────────────────

class JuniorDevOpsEnv:
    """
    Simulated server environment.  The 'filesystem', 'processes', and
    'services' are plain Python dicts — no real OS calls needed.
    """

    # Supported commands (used for help text and validation)
    COMMANDS = [
        "ls [path]",
        "cat <file>",
        "grep <pattern> <file>",
        "ps",
        "kill <pid>",
        "top",
        "sed <old> <new> <file>",
        "restart <service>",
        "status <service>",
        "echo <text>",
        "help",
    ]

    def __init__(self, difficulty: Difficulty = Difficulty.EASY, seed: int | None = None):
        self.difficulty = difficulty
        self.seed       = seed if seed is not None else random.randint(0, 9999)
        self._rng       = random.Random(self.seed)

        # State is populated on reset()
        self._fs:       dict[str, str]  = {}   # path -> file content
        self._procs:    list[dict]      = []   # running processes
        self._services: dict[str, str]  = {}   # service -> status
        self._config:   dict[str, str]  = {}   # flat key=value config pairs

        self._step_count = 0
        self._max_steps  = 20
        self._done       = False

        # Grading checkpoints (task-specific, set in reset)
        self._checkpoints: dict[str, bool] = {}
        self._target: dict[str, Any] = {}

        # Reward already awarded per checkpoint (so it's never double-counted)
        self._awarded: set[str] = set()

        self.reset()

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> dict:
        """Reset the environment and return the initial state observation."""
        self._rng       = random.Random(self.seed)
        self._step_count = 0
        self._done       = False
        self._awarded    = set()
        self._checkpoints = {}

        self._build_filesystem()
        self._build_processes()
        self._build_services()

        if self.difficulty == Difficulty.EASY:
            self._setup_easy_task()
        elif self.difficulty == Difficulty.MEDIUM:
            self._setup_medium_task()
        else:
            self._setup_hard_task()

        return self.state()

    def state(self) -> dict:
        """Return a full snapshot of the current environment state."""
        return {
            "difficulty":    self.difficulty,
            "step":          self._step_count,
            "max_steps":     self._max_steps,
            "done":          self._done,
            "filesystem":    copy.deepcopy(self._fs),
            "processes":     copy.deepcopy(self._procs),
            "services":      copy.deepcopy(self._services),
            "task":          copy.deepcopy(self._target),
            "checkpoints":   copy.deepcopy(self._checkpoints),
            "available_cmds": self.COMMANDS,
        }

    def step(self, action: str) -> StepResult:
        """
        Execute one agent action (a shell-like command string).
        Returns a StepResult with observation, reward, done, info.
        """
        if self._done:
            return StepResult(
                observation="Episode already finished. Call reset() to start again.",
                reward=0.0,
                done=True,
                info={"error": "episode_done"},
            )

        self._step_count += 1
        action = action.strip()

        # Dispatch command
        try:
            obs = self._dispatch(action)
        except Exception as exc:
            obs = f"[ERROR] {exc}"

        # Grade current state
        reward, new_checkpoints = self._grade()

        # Check terminal conditions
        all_done  = all(self._checkpoints.values())
        timed_out = self._step_count >= self._max_steps

        if all_done or timed_out:
            self._done = True

        if timed_out and not all_done:
            obs += "\n⏰ Max steps reached. Episode ended."

        return StepResult(
            observation=obs,
            reward=reward,
            done=self._done,
            info={
                "step":             self._step_count,
                "checkpoints":      copy.deepcopy(self._checkpoints),
                "new_checkpoints":  new_checkpoints,
                "action":           action,
            },
        )

    # ── Command Dispatcher ────────────────────────────────────────────────────

    def _dispatch(self, action: str) -> str:
        parts = action.split()
        if not parts:
            return "[SHELL] No command entered."

        cmd  = parts[0].lower()
        args = parts[1:]

        if cmd == "help":
            return "Available commands:\n" + "\n".join(f"  {c}" for c in self.COMMANDS)

        elif cmd == "ls":
            path = args[0] if args else "/"
            # List files whose path starts with the requested directory
            entries = []
            for fp in self._fs:
                if fp.startswith(path.rstrip("/") + "/") or fp == path:
                    rel = fp[len(path.rstrip("/")):]
                    # Show only the immediate child
                    rel = rel.strip("/")
                    if "/" not in rel:
                        entries.append(rel)
            if not entries:
                return f"ls: cannot access '{path}': No such file or directory"
            return "\n".join(sorted(set(entries)))

        elif cmd == "cat":
            if not args:
                return "Usage: cat <file>"
            fp = args[0]
            if fp not in self._fs:
                return f"cat: {fp}: No such file or directory"
            return self._fs[fp]

        elif cmd == "grep":
            if len(args) < 2:
                return "Usage: grep <pattern> <file>"
            pattern, fp = args[0], args[1]
            if fp not in self._fs:
                return f"grep: {fp}: No such file or directory"
            matches = [ln for ln in self._fs[fp].splitlines() if re.search(pattern, ln)]
            if not matches:
                return f"(no matches for '{pattern}' in {fp})"
            return "\n".join(matches)

        elif cmd == "ps":
            if not self._procs:
                return "No running processes."
            header = f"{'PID':>6}  {'USER':<10}  {'%CPU':>5}  {'COMMAND'}"
            rows   = [f"{p['pid']:>6}  {p['user']:<10}  {p['cpu']:>4}%  {p['cmd']}" for p in self._procs]
            return header + "\n" + "\n".join(rows)

        elif cmd == "top":
            # Like ps but sorted by CPU descending
            sorted_procs = sorted(self._procs, key=lambda p: p["cpu"], reverse=True)
            header = f"{'PID':>6}  {'USER':<10}  {'%CPU':>5}  {'COMMAND'}"
            rows   = [f"{p['pid']:>6}  {p['user']:<10}  {p['cpu']:>4}%  {p['cmd']}" for p in sorted_procs]
            return "Tasks: {} total\n".format(len(self._procs)) + header + "\n" + "\n".join(rows)

        elif cmd == "kill":
            if not args:
                return "Usage: kill <pid>"
            try:
                pid = int(args[0])
            except ValueError:
                return f"kill: invalid PID: {args[0]}"
            before = len(self._procs)
            self._procs = [p for p in self._procs if p["pid"] != pid]
            if len(self._procs) < before:
                return f"[OK] Process {pid} terminated."
            return f"kill: ({pid}) - No such process"

        elif cmd == "sed":
            # Simplified: sed <old_string> <new_string> <file>
            if len(args) < 3:
                return "Usage: sed <old> <new> <file>"
            old, new, fp = args[0], args[1], args[2]
            if fp not in self._fs:
                return f"sed: {fp}: No such file or directory"
            if old not in self._fs[fp]:
                return f"sed: pattern '{old}' not found in {fp}"
            self._fs[fp] = self._fs[fp].replace(old, new, 1)
            return f"[OK] Replaced '{old}' with '{new}' in {fp}"

        elif cmd == "restart":
            if not args:
                return "Usage: restart <service>"
            svc = args[0]
            if svc not in self._services:
                return f"restart: service '{svc}' not found"
            self._services[svc] = "running"
            return f"[OK] {svc} restarted successfully. Status: running"

        elif cmd == "status":
            if not args:
                return "Usage: status <service>"
            svc = args[0]
            if svc not in self._services:
                return f"status: service '{svc}' not found"
            return f"{svc}: {self._services[svc]}"

        elif cmd == "echo":
            return " ".join(args)

        else:
            return f"bash: {cmd}: command not found. Type 'help' for available commands."

    # ── World Builders ────────────────────────────────────────────────────────

    def _build_filesystem(self):
        """Create a realistic-looking /var/log and /etc tree."""
        self._fs = {
            "/etc/hostname": "prod-server-01\n",
            "/etc/os-release": "NAME=Ubuntu\nVERSION=22.04 LTS\n",
            "/var/log/auth.log": self._gen_auth_log(),
            "/var/log/syslog":   self._gen_syslog(),
        }

    def _build_processes(self):
        """Populate a fake process table."""
        self._procs = [
            {"pid": 1,    "user": "root",    "cpu": 0.0,  "cmd": "init"},
            {"pid": 512,  "user": "www-data","cpu": 1.2,  "cmd": "nginx: master process"},
            {"pid": 513,  "user": "www-data","cpu": 0.8,  "cmd": "nginx: worker process"},
            {"pid": 1024, "user": "postgres","cpu": 2.1,  "cmd": "postgres: checkpointer"},
            {"pid": 2048, "user": "root",    "cpu": 0.3,  "cmd": "sshd"},
        ]

    def _build_services(self):
        self._services = {
            "nginx":    "running",
            "postgres": "running",
            "redis":    "running",
        }

    # ── Task Setups ───────────────────────────────────────────────────────────

    def _setup_easy_task(self):
        """
        EASY: Find the error code in /var/log/app.log
        Checkpoints:
          1. Agent cats or greps the correct file          (+0.4)
          2. Agent identifies the exact error code         (+0.6)
        """
        codes = ["ERR_502", "ERR_503", "ERR_DB_CONN", "ERR_DISK_FULL", "ERR_OOM"]
        self._target["error_code"] = self._rng.choice(codes)
        self._target["log_file"]   = "/var/log/app.log"
        self._target["description"] = (
            f"The application is crashing. Find the error code in {self._target['log_file']} "
            f"and report it using: echo <ERROR_CODE>"
        )

        # Inject the error into the log
        self._fs["/var/log/app.log"] = self._gen_app_log(self._target["error_code"])

        self._checkpoints = {
            "opened_log_file": False,
            "found_error_code": False,
        }

    def _setup_medium_task(self):
        """
        MEDIUM: Kill the highest CPU process
        Checkpoints:
          1. Agent runs ps or top                          (+0.3)
          2. Agent identifies the rogue process PID        (+0.3)
          3. Agent kills the correct process               (+0.4)
        """
        rogue_pid = 9999
        rogue_cpu = self._rng.randint(85, 99)
        self._procs.append({
            "pid":  rogue_pid,
            "user": "ubuntu",
            "cpu":  rogue_cpu,
            "cmd":  "python3 /opt/miner.py",
        })

        self._target = {
            "rogue_pid":       rogue_pid,
            "rogue_cpu":       rogue_cpu,
            "description": (
                f"A rogue process is consuming {rogue_cpu}% CPU and grinding the server to a halt. "
                "Use 'ps' or 'top' to identify it, then kill it."
            ),
        }

        self._checkpoints = {
            "inspected_processes":    False,
            "identified_rogue_pid":   False,  # set externally by grader heuristic
            "killed_rogue_process":   False,
        }

    def _setup_hard_task(self):
        """
        HARD: Fix a port conflict in nginx config, then restart nginx.
        Checkpoints:
          1. Agent reads the config file                   (+0.2)
          2. Agent finds the conflicting port              (+0.2)
          3. Agent edits the config correctly              (+0.3)
          4. Agent restarts nginx successfully             (+0.3)
        """
        bad_port  = 5432   # conflicts with postgres
        good_port = 8080
        cfg_path  = "/etc/nginx/nginx.conf"

        self._fs[cfg_path] = (
            "worker_processes auto;\n"
            "events { worker_connections 1024; }\n"
            "http {\n"
            f"    listen {bad_port};\n"
            "    server_name prod-server-01;\n"
            "    location / { proxy_pass http://127.0.0.1:3000; }\n"
            "}\n"
        )
        self._services["nginx"] = "failed"  # it's down because of the conflict

        self._target = {
            "config_file": cfg_path,
            "bad_port":    bad_port,
            "good_port":   good_port,
            "description": (
                f"nginx is in 'failed' state because its config ({cfg_path}) is set to listen on "
                f"port {bad_port}, which conflicts with PostgreSQL. "
                f"Change the listen port to {good_port} and restart nginx."
            ),
        }

        self._checkpoints = {
            "read_config":        False,
            "identified_conflict": False,
            "fixed_config":       False,
            "restarted_nginx":    False,
        }

    # ── Grader ────────────────────────────────────────────────────────────────

    def _grade(self) -> tuple[float, list[str]]:
        """
        Inspect current world state, update checkpoints, compute cumulative reward.
        Returns (cumulative_reward, list_of_newly_unlocked_checkpoints).
        """
        newly_unlocked = []

        if self.difficulty == Difficulty.EASY:
            newly_unlocked = self._grade_easy()
        elif self.difficulty == Difficulty.MEDIUM:
            newly_unlocked = self._grade_medium()
        else:
            newly_unlocked = self._grade_hard()

        # Assign weights per checkpoint
        weights = self._checkpoint_weights()
        cumulative = sum(
            weights.get(k, 0.0)
            for k, v in self._checkpoints.items()
            if v
        )

        return float(round(min(cumulative, 1.0), 3)), newly_unlocked

    def _grade_easy(self) -> list[str]:
        new = []
        # cp1: did the agent open the log?
        # We detect this by checking if the file still matches (proxy: always true after read)
        # Better: we track via a flag set in _dispatch — but since we don't mutate on read,
        # we use the observation text.  Instead, let's track via a side-channel flag.
        # (See _dispatch — we set _opened_log on cat/grep of target file)
        if getattr(self, "_opened_log", False) and not self._checkpoints["opened_log_file"]:
            self._checkpoints["opened_log_file"] = True
            new.append("opened_log_file")

        if getattr(self, "_echoed_code", False) and not self._checkpoints["found_error_code"]:
            self._checkpoints["found_error_code"] = True
            new.append("found_error_code")

        return new

    def _grade_medium(self) -> list[str]:
        new = []
        if getattr(self, "_inspected_procs", False) and not self._checkpoints["inspected_processes"]:
            self._checkpoints["inspected_processes"] = True
            new.append("inspected_processes")

        rogue_pid = self._target["rogue_pid"]
        rogue_gone = not any(p["pid"] == rogue_pid for p in self._procs)

        if rogue_gone and not self._checkpoints["killed_rogue_process"]:
            self._checkpoints["identified_rogue_pid"]  = True
            self._checkpoints["killed_rogue_process"]  = True
            new.extend(["identified_rogue_pid", "killed_rogue_process"])

        return new

    def _grade_hard(self) -> list[str]:
        new = []
        cfg  = self._target["config_file"]
        good = str(self._target["good_port"])
        bad  = str(self._target["bad_port"])

        if getattr(self, "_read_config", False) and not self._checkpoints["read_config"]:
            self._checkpoints["read_config"] = True
            new.append("read_config")

        # Config fixed = bad port no longer present & good port present
        content = self._fs.get(cfg, "")
        if bad not in content and good in content:
            if not self._checkpoints["identified_conflict"]:
                self._checkpoints["identified_conflict"] = True
                new.append("identified_conflict")
            if not self._checkpoints["fixed_config"]:
                self._checkpoints["fixed_config"] = True
                new.append("fixed_config")

        if self._services.get("nginx") == "running" and self._checkpoints["fixed_config"]:
            if not self._checkpoints["restarted_nginx"]:
                self._checkpoints["restarted_nginx"] = True
                new.append("restarted_nginx")

        return new

    def _checkpoint_weights(self) -> dict[str, float]:
        if self.difficulty == Difficulty.EASY:
            return {"opened_log_file": 0.4, "found_error_code": 0.6}
        elif self.difficulty == Difficulty.MEDIUM:
            return {
                "inspected_processes":  0.30,
                "identified_rogue_pid": 0.30,
                "killed_rogue_process": 0.40,
            }
        else:
            return {
                "read_config":         0.20,
                "identified_conflict": 0.20,
                "fixed_config":        0.30,
                "restarted_nginx":     0.30,
            }

    # ── Side-channel flags (set inside _dispatch overrides) ──────────────────

    def _dispatch(self, action: str) -> str:  # noqa: F811  (intentional override)
        """Augmented dispatcher that sets grading side-channel flags."""
        parts = action.split()
        if not parts:
            return "[SHELL] No command entered."

        cmd  = parts[0].lower()
        args = parts[1:]

        # ── Easy: detect log access ──────────────────────────────────────────
        if cmd in ("cat", "grep"):
            target = self._target.get("log_file", "")
            if args and (args[-1] == target or (len(args) > 1 and args[-1] == target)):
                self._opened_log = True
            elif args and target in " ".join(args):
                self._opened_log = True

        # ── Easy: detect echoed error code ──────────────────────────────────
        if cmd == "echo":
            code = self._target.get("error_code", "__NONE__")
            if code in " ".join(args):
                self._echoed_code = True

        # ── Medium: detect ps/top ────────────────────────────────────────────
        if cmd in ("ps", "top"):
            self._inspected_procs = True

        # ── Hard: detect config read ─────────────────────────────────────────
        if cmd in ("cat", "grep"):
            cfg = self._target.get("config_file", "__NONE__")
            if args and cfg in " ".join(args):
                self._read_config = True

        # Delegate to the base dispatcher logic (inlined to avoid recursion)
        return self._execute(cmd, args)

    def _execute(self, cmd: str, args: list[str]) -> str:
        """Pure execution — no side-channel tracking."""
        if cmd == "help":
            return "Available commands:\n" + "\n".join(f"  {c}" for c in self.COMMANDS)

        elif cmd == "ls":
            path = args[0] if args else "/"
            entries = []
            for fp in self._fs:
                if fp.startswith(path.rstrip("/") + "/"):
                    rel = fp[len(path.rstrip("/") + "/"):]
                    top = rel.split("/")[0]
                    entries.append(top)
            if not entries:
                return f"ls: cannot access '{path}': No such file or directory"
            return "\n".join(sorted(set(entries)))

        elif cmd == "cat":
            if not args:
                return "Usage: cat <file>"
            fp = args[0]
            return self._fs.get(fp, f"cat: {fp}: No such file or directory")

        elif cmd == "grep":
            if len(args) < 2:
                return "Usage: grep <pattern> <file>"
            pattern, fp = args[0], args[1]
            if fp not in self._fs:
                return f"grep: {fp}: No such file or directory"
            matches = [ln for ln in self._fs[fp].splitlines() if re.search(pattern, ln)]
            return "\n".join(matches) if matches else f"(no matches for '{pattern}' in {fp})"

        elif cmd == "ps":
            header = f"{'PID':>6}  {'USER':<10}  {'%CPU':>5}  COMMAND"
            rows   = [f"{p['pid']:>6}  {p['user']:<10}  {p['cpu']:>5}  {p['cmd']}" for p in self._procs]
            return header + "\n" + "\n".join(rows)

        elif cmd == "top":
            sp     = sorted(self._procs, key=lambda p: p["cpu"], reverse=True)
            header = f"{'PID':>6}  {'USER':<10}  {'%CPU':>5}  COMMAND"
            rows   = [f"{p['pid']:>6}  {p['user']:<10}  {p['cpu']:>5}  {p['cmd']}" for p in sp]
            return f"Tasks: {len(self._procs)} total\n" + header + "\n" + "\n".join(rows)

        elif cmd == "kill":
            if not args:
                return "Usage: kill <pid>"
            try:
                pid = int(args[0])
            except ValueError:
                return f"kill: invalid PID: {args[0]}"
            before = len(self._procs)
            self._procs = [p for p in self._procs if p["pid"] != pid]
            return f"[OK] Process {pid} terminated." if len(self._procs) < before else f"kill: ({pid}) - No such process"

        elif cmd == "sed":
            if len(args) < 3:
                return "Usage: sed <old> <new> <file>"
            old, new, fp = args[0], args[1], args[2]
            if fp not in self._fs:
                return f"sed: {fp}: No such file or directory"
            if old not in self._fs[fp]:
                return f"sed: pattern '{old}' not found in {fp}"
            self._fs[fp] = self._fs[fp].replace(old, new, 1)
            return f"[OK] Replaced '{old}' with '{new}' in {fp}"

        elif cmd == "restart":
            if not args:
                return "Usage: restart <service>"
            svc = args[0]
            if svc not in self._services:
                return f"restart: service '{svc}' not found"
            self._services[svc] = "running"
            return f"[OK] {svc} restarted. Status: running"

        elif cmd == "status":
            if not args:
                return "Usage: status <service>"
            svc = args[0]
            if svc not in self._services:
                return f"status: service '{svc}' not found"
            return f"{svc}: {self._services[svc]}"

        elif cmd == "echo":
            return " ".join(args)

        else:
            return f"bash: {cmd}: command not found. Type 'help' for available commands."

    # ── Log generators ────────────────────────────────────────────────────────

    def _gen_app_log(self, error_code: str) -> str:
        lines = [
            "[2025-04-07 03:12:01] INFO  Server started on port 8080",
            "[2025-04-07 03:14:22] INFO  Accepted connection from 10.0.0.4",
            "[2025-04-07 03:15:05] WARN  Response time exceeded 500ms",
            "[2025-04-07 03:17:38] INFO  Health check passed",
            "[2025-04-07 03:21:11] ERROR " + error_code + " — unexpected shutdown",
            "[2025-04-07 03:21:11] ERROR Stack trace: main.go:247 handler.go:89",
            "[2025-04-07 03:21:12] INFO  Attempting restart...",
            "[2025-04-07 03:21:14] ERROR " + error_code + " — restart failed",
        ]
        self._rng.shuffle(lines[1:4])  # shuffle middle info lines only
        return "\n".join(lines) + "\n"

    def _gen_auth_log(self) -> str:
        return (
            "Apr  7 03:00:01 prod sshd[1024]: Accepted publickey for ubuntu from 192.168.1.5\n"
            "Apr  7 03:02:18 prod sshd[1024]: Failed password for root from 45.33.32.156\n"
            "Apr  7 03:02:19 prod sshd[1024]: Failed password for root from 45.33.32.156\n"
            "Apr  7 03:05:44 prod sudo: ubuntu : TTY=pts/0 ; USER=root ; COMMAND=/bin/bash\n"
        )

    def _gen_syslog(self) -> str:
        return (
            "Apr  7 03:00:00 prod kernel: Initializing cgroup subsys cpuset\n"
            "Apr  7 03:00:01 prod systemd[1]: Started Session 1 of user ubuntu.\n"
            "Apr  7 03:12:00 prod systemd[1]: nginx.service: Main process exited\n"
            "Apr  7 03:12:00 prod systemd[1]: nginx.service: Failed with result 'exit-code'\n"
        )
