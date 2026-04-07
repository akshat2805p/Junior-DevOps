# Tests for Junior DevOps Environment
# Covers all difficulty levels and grading logic

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from devops_env import JuniorDevOpsEnv, Difficulty, StepResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def easy_env():
    return JuniorDevOpsEnv(difficulty=Difficulty.EASY, seed=42)

@pytest.fixture
def medium_env():
    return JuniorDevOpsEnv(difficulty=Difficulty.MEDIUM, seed=42)

@pytest.fixture
def hard_env():
    return JuniorDevOpsEnv(difficulty=Difficulty.HARD, seed=42)


# ── Structural tests ──────────────────────────────────────────────────────────

class TestAPIContract:
    def test_reset_returns_state_dict(self, easy_env):
        s = easy_env.reset()
        assert isinstance(s, dict)
        assert "difficulty" in s
        assert "checkpoints" in s
        assert "filesystem" in s
        assert "processes" in s

    def test_step_returns_step_result(self, easy_env):
        result = easy_env.step("help")
        assert isinstance(result, StepResult)
        assert isinstance(result.observation, str)
        assert isinstance(result.reward, float)
        assert isinstance(result.done, bool)
        assert isinstance(result.info, dict)

    def test_state_is_consistent_with_reset(self, easy_env):
        s1 = easy_env.reset()
        s2 = easy_env.state()
        assert s1["difficulty"] == s2["difficulty"]
        assert s1["checkpoints"] == s2["checkpoints"]

    def test_reward_is_between_0_and_1(self, easy_env):
        for _ in range(10):
            result = easy_env.step("ps")
            assert 0.0 <= result.reward <= 1.0

    def test_seed_determinism(self):
        env_a = JuniorDevOpsEnv(difficulty=Difficulty.EASY, seed=7)
        env_b = JuniorDevOpsEnv(difficulty=Difficulty.EASY, seed=7)
        assert env_a._target["error_code"] == env_b._target["error_code"]

    def test_episode_ends_at_max_steps(self):
        env = JuniorDevOpsEnv(difficulty=Difficulty.EASY, seed=1)
        env._max_steps = 3
        for _ in range(3):
            result = env.step("echo hello")
        assert result.done

    def test_no_action_after_done(self, easy_env):
        easy_env._done = True
        result = easy_env.step("ls /")
        assert result.done
        assert result.reward == 0.0


# ── Command tests ─────────────────────────────────────────────────────────────

class TestCommands:
    def test_help(self, easy_env):
        r = easy_env.step("help")
        assert "Available commands" in r.observation

    def test_ls_root(self, easy_env):
        r = easy_env.step("ls /")
        assert "var" in r.observation or "etc" in r.observation

    def test_cat_existing_file(self, easy_env):
        r = easy_env.step("cat /etc/hostname")
        assert "prod-server-01" in r.observation

    def test_cat_missing_file(self, easy_env):
        r = easy_env.step("cat /does/not/exist")
        assert "No such file" in r.observation

    def test_grep_found(self, easy_env):
        r = easy_env.step("grep ERROR /var/log/app.log")
        assert "ERROR" in r.observation

    def test_grep_not_found(self, easy_env):
        r = easy_env.step("grep ZZZZZZ /var/log/app.log")
        assert "no matches" in r.observation

    def test_ps_output(self, easy_env):
        r = easy_env.step("ps")
        assert "PID" in r.observation
        assert "COMMAND" in r.observation

    def test_top_sorted_by_cpu(self, medium_env):
        r = medium_env.step("top")
        lines = [l for l in r.observation.splitlines() if "%" in l or "CPU" in l]
        # Rogue process (90%+ CPU) should appear near top
        assert any("miner" in l for l in r.observation.splitlines())

    def test_kill_valid_pid(self, medium_env):
        pid = medium_env._target["rogue_pid"]
        r   = medium_env.step(f"kill {pid}")
        assert "terminated" in r.observation.lower()
        assert not any(p["pid"] == pid for p in medium_env._procs)

    def test_kill_invalid_pid(self, easy_env):
        r = easy_env.step("kill 99999")
        assert "No such process" in r.observation

    def test_kill_non_numeric_pid(self, easy_env):
        r = easy_env.step("kill abc")
        assert "invalid" in r.observation.lower()

    def test_sed_replaces_text(self, hard_env):
        r = hard_env.step("sed 5432 8080 /etc/nginx/nginx.conf")
        assert "Replaced" in r.observation
        assert "5432" not in hard_env._fs["/etc/nginx/nginx.conf"]
        assert "8080" in hard_env._fs["/etc/nginx/nginx.conf"]

    def test_sed_missing_file(self, hard_env):
        r = hard_env.step("sed foo bar /no/such/file")
        assert "No such file" in r.observation

    def test_sed_pattern_not_found(self, hard_env):
        r = hard_env.step("sed NOTFOUND replacement /etc/nginx/nginx.conf")
        assert "not found" in r.observation

    def test_restart_service(self, hard_env):
        r = hard_env.step("restart nginx")
        assert "running" in r.observation.lower()
        assert hard_env._services["nginx"] == "running"

    def test_restart_unknown_service(self, easy_env):
        r = easy_env.step("restart doesnotexist")
        assert "not found" in r.observation

    def test_status_service(self, hard_env):
        r = hard_env.step("status nginx")
        assert "nginx" in r.observation

    def test_unknown_command(self, easy_env):
        r = easy_env.step("rm -rf /")
        assert "command not found" in r.observation

    def test_echo(self, easy_env):
        r = easy_env.step("echo hello world")
        assert "hello world" in r.observation


# ── Grading / Partial Rewards ─────────────────────────────────────────────────

class TestGrading:

    # EASY
    def test_easy_partial_reward_after_cat(self, easy_env):
        log_file = easy_env._target["log_file"]
        r = easy_env.step(f"cat {log_file}")
        assert r.reward == pytest.approx(0.4, abs=0.01)

    def test_easy_full_reward_after_echo_code(self, easy_env):
        log_file = easy_env._target["log_file"]
        code     = easy_env._target["error_code"]
        easy_env.step(f"cat {log_file}")
        r = easy_env.step(f"echo {code}")
        assert r.reward == pytest.approx(1.0, abs=0.01)

    def test_easy_done_after_full_reward(self, easy_env):
        log_file = easy_env._target["log_file"]
        code     = easy_env._target["error_code"]
        easy_env.step(f"cat {log_file}")
        r = easy_env.step(f"echo {code}")
        assert r.done

    def test_easy_reward_never_decreases(self, easy_env):
        prev = 0.0
        for action in ["ls /", "ps", "help", "echo hi"]:
            r = easy_env.step(action)
            assert r.reward >= prev
            prev = r.reward

    # MEDIUM
    def test_medium_partial_reward_after_ps(self, medium_env):
        r = medium_env.step("ps")
        assert r.reward == pytest.approx(0.30, abs=0.01)

    def test_medium_full_reward_after_kill(self, medium_env):
        pid = medium_env._target["rogue_pid"]
        medium_env.step("top")
        r = medium_env.step(f"kill {pid}")
        assert r.reward == pytest.approx(1.0, abs=0.01)
        assert r.done

    def test_medium_kill_wrong_pid_no_full_reward(self, medium_env):
        medium_env.step("ps")
        r = medium_env.step("kill 1")   # kills init, not the rogue process
        assert r.reward < 1.0

    # HARD
    def test_hard_partial_reward_after_cat_config(self, hard_env):
        cfg = hard_env._target["config_file"]
        r   = hard_env.step(f"cat {cfg}")
        assert r.reward >= 0.2

    def test_hard_partial_reward_after_sed(self, hard_env):
        cfg  = hard_env._target["config_file"]
        hard_env.step(f"cat {cfg}")
        r = hard_env.step(f"sed 5432 8080 {cfg}")
        assert r.reward >= 0.6   # read + identified + fixed

    def test_hard_full_reward_after_restart(self, hard_env):
        cfg = hard_env._target["config_file"]
        hard_env.step(f"cat {cfg}")
        hard_env.step(f"sed 5432 8080 {cfg}")
        r = hard_env.step("restart nginx")
        assert r.reward == pytest.approx(1.0, abs=0.01)
        assert r.done

    def test_hard_restart_before_fix_no_full_reward(self, hard_env):
        # Restarting nginx without fixing the config should not give full reward
        r = hard_env.step("restart nginx")
        assert r.reward < 1.0


# ── Reset idempotency ─────────────────────────────────────────────────────────

class TestReset:
    def test_reset_clears_checkpoints(self, easy_env):
        log = easy_env._target["log_file"]
        code = easy_env._target["error_code"]
        easy_env.step(f"cat {log}")
        easy_env.step(f"echo {code}")
        easy_env.reset()
        assert not any(easy_env._checkpoints.values())

    def test_reset_clears_done_flag(self, easy_env):
        easy_env._done = True
        easy_env.reset()
        assert not easy_env._done

    def test_reset_restores_processes(self, medium_env):
        pid = medium_env._target["rogue_pid"]
        medium_env.step(f"kill {pid}")
        medium_env.reset()
        assert any(p["pid"] == pid for p in medium_env._procs)
