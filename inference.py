"""
Inference Script — Traffic Control OpenEnv Environment
===================================
MANDATORY
- Before submitting, ensure the following variables are defined:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    IMAGE_NAME     The Docker image name (used to start the env container).

STDOUT FORMAT
- The script emits exactly three line types to stdout:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

IMAGE_NAME = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = "traffic-control-env"

# Where to reach the env container
ENV_URL = os.getenv("ENV_URL") or os.getenv("CONTAINER_URL") or "http://localhost:7860"

TASK_IDS = ["single_intersection", "corridor_green_wave", "grid_incident"]


# ─────────────────────────────────────────────
# Simple HTTP Env Client (no websockets needed)
# ─────────────────────────────────────────────

class SimpleEnvClient:
    """Minimal HTTP client for the traffic env. Uses only stdlib urllib."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def _post(self, path: str, data: dict) -> dict:
        url = f"{self.base_url}{path}"
        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get(self, path: str) -> dict:
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def health(self) -> dict:
        return self._get("/health")

    def reset(self, task_id: str) -> dict:
        return self._post("/reset", {"task_id": task_id})

    def step(self, action: dict) -> dict:
        return self._post("/step", {"action": action})

    def close(self):
        pass


# ─────────────────────────────────────────────
# Docker container management
# ─────────────────────────────────────────────

_container_id: Optional[str] = None


def start_container(image_name: str, port: int = 7860) -> str:
    """Start the Docker container and return the base URL."""
    global _container_id
    try:
        result = subprocess.run(
            [
                "docker", "run", "-d", "--rm",
                "-p", f"{port}:7860",
                image_name,
            ],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            print(f"[DEBUG] docker run failed: {result.stderr}", file=sys.stderr)
            return f"http://localhost:{port}"
        _container_id = result.stdout.strip()
        print(f"[DEBUG] Started container {_container_id[:12]}", file=sys.stderr)
    except Exception as e:
        print(f"[DEBUG] docker run error: {e}", file=sys.stderr)

    return f"http://localhost:{port}"


def stop_container():
    """Stop the Docker container if we started one."""
    global _container_id
    if _container_id:
        try:
            subprocess.run(
                ["docker", "stop", _container_id],
                capture_output=True, text=True, timeout=30,
            )
        except Exception:
            pass
        _container_id = None


def wait_for_ready(base_url: str, timeout: int = 120):
    """Poll the health endpoint until ready."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(f"{base_url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"Container at {base_url} not ready after {timeout}s")


# ─────────────────────────────────────────────
# LLM System Prompt
# ─────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
You are a Traffic Management Center AI controlling traffic signals \
at a cluster of intersections in a city network.

## Your Responsibilities
1. SET SIGNAL PHASES for intersections in your cluster to minimize vehicle delay
2. PRIORITIZE emergency vehicles — give them green lights immediately
3. READ incoming coordination messages from neighboring agents and act on them
4. SEND coordination messages to neighbors when your actions affect their traffic

## Signal Rules
- Available phases: NS_GREEN, NS_LEFT, EW_GREEN, EW_LEFT
- Switching phases costs 2 steps (yellow + all-red clearance)
- Minimum green time is enforced — commands during minimum green are ignored
- Give more green time to directions with longer queues
- Emergency vehicles WAIT at red lights — you MUST clear the path before they arrive

## Coordination Protocol
When you release heavy traffic or an emergency vehicle toward a neighbor, send a message:
- message_type: "traffic_alert", "emergency_alert", "congestion_warning", or "status_update"
- urgency: "routine", "elevated", "urgent", or "critical"
- content: Describe what's happening in natural language
- suggested_action: What the neighbor should do

## Response Format
Respond with ONLY a valid JSON object (no markdown, no explanation):
{
  "commands": [{"intersection_id": "I1", "target_phase": "NS_GREEN"}],
  "messages": [],
  "reasoning": "brief explanation"
}
""")


# ─────────────────────────────────────────────
# Observation → Text
# ─────────────────────────────────────────────


def summarize_observation(obs_data: Dict[str, Any]) -> str:
    """Convert observation dict to concise text for the LLM."""
    lines = [
        f"Step {obs_data.get('step_number', 0)} | "
        f"Time: {obs_data.get('time_seconds', 0)}s | "
        f"Your cluster: {obs_data.get('agent_cluster', [])}"
    ]

    for ix in obs_data.get("intersections", []):
        ctrl = "YOU" if ix.get("controlled_by") == "agent" else "HEURISTIC"
        phase = ix.get("current_phase", "?")
        elapsed = ix.get("phase_time_elapsed", 0)
        min_green = ix.get("min_green_remaining", 0)
        lines.append(
            f"\n[{ix.get('intersection_id', '?')}] ({ctrl}) "
            f"phase={phase} (elapsed={elapsed}, min_green_left={min_green})"
        )

        for a in ix.get("approaches", []):
            if a.get("queue_length", 0) > 0 or a.get("approaching_vehicles", 0) > 0:
                lines.append(
                    f"  {a.get('direction', '?')}: queue={a.get('queue_length', 0)}, "
                    f"approaching={a.get('approaching_vehicles', 0)}, "
                    f"wait_avg={a.get('wait_time_avg', 0):.1f}s, "
                    f"wait_max={a.get('wait_time_max', 0):.1f}s"
                )

    evs = obs_data.get("emergency_vehicles", [])
    if evs:
        lines.append("\n⚠️  EMERGENCY VEHICLES:")
        for ev in evs:
            status = "WAITING AT RED" if ev.get("waiting") else "moving"
            lines.append(
                f"  {ev.get('ev_id', '?')} (priority {ev.get('priority', '?')}): "
                f"heading {ev.get('direction', '?')} to {ev.get('target_intersection', '?')}, "
                f"distance={ev.get('distance_to_next_intersection', '?')} cells, "
                f"delay={ev.get('total_delay', 0):.0f}s, status={status}"
            )

    msgs = obs_data.get("incoming_messages", [])
    if msgs:
        lines.append("\n📨 INCOMING MESSAGES:")
        for msg in msgs:
            lines.append(
                f"  FROM {msg.get('from_node', '?')} → {msg.get('to_node', '?')} "
                f"[{msg.get('message_type', '?')}|{msg.get('urgency', '?')}]: "
                f"{msg.get('content', '')}"
            )
            if msg.get("suggested_action"):
                lines.append(f"    Suggestion: {msg['suggested_action']}")

    m = obs_data.get("network_metrics", {})
    lines.append(
        f"\nNetwork: {m.get('total_vehicles_in_network', 0)} vehicles, "
        f"{m.get('total_vehicles_waiting', 0)} waiting, "
        f"{m.get('total_vehicles_cleared', 0)} cleared, "
        f"pressure={m.get('network_pressure', 0):.2f}"
    )

    return "\n".join(lines)


# ─────────────────────────────────────────────
# LLM Response → Action dict
# ─────────────────────────────────────────────


def parse_action(response_text: str) -> dict:
    """Parse LLM response into an action dict. Falls back to noop on error."""
    try:
        text = response_text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        data = json.loads(text)
        return {
            "commands": data.get("commands", []),
            "messages": data.get("messages", []),
            "reasoning": data.get("reasoning", ""),
        }
    except Exception:
        return {"commands": [], "messages": [], "reasoning": "parse_error"}


def action_to_string(action: dict) -> str:
    """Convert action dict to a compact string for the [STEP] log."""
    parts = []
    for cmd in action.get("commands", []):
        parts.append(f"set_phase({cmd.get('intersection_id','?')},{cmd.get('target_phase','?')})")
    for msg in action.get("messages", []):
        parts.append(f"send_msg({msg.get('to_node','?')},{msg.get('message_type','?')})")
    if not parts:
        parts.append("noop()")
    return ";".join(parts)


# ─────────────────────────────────────────────
# Run one task
# ─────────────────────────────────────────────


def run_task(llm: OpenAI, env: SimpleEnvClient, task_id: str) -> float:
    """Run a single task. Returns the score."""
    rewards: List[float] = []
    step_num = 0
    success = False
    score = 0.0

    # [START]
    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}")
    sys.stdout.flush()

    try:
        # Reset returns an observation dict
        resp = env.reset(task_id)
        # The server returns either the observation directly or wrapped
        obs_data = resp.get("observation", resp) if isinstance(resp, dict) else {}
        done = resp.get("done", False) if isinstance(resp, dict) else False

        while not done:
            obs_text = summarize_observation(obs_data) if isinstance(obs_data, dict) else str(obs_data)

            # Query the LLM
            try:
                response = llm.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": obs_text},
                    ],
                    temperature=0.2,
                    max_tokens=400,
                    timeout=30,
                )
                action_text = response.choices[0].message.content or "{}"
            except Exception:
                action_text = '{"commands":[],"messages":[],"reasoning":"llm_error"}'

            # Parse action
            action = parse_action(action_text)

            # Step
            resp = env.step(action)
            obs_data = resp.get("observation", resp) if isinstance(resp, dict) else {}
            reward = float(resp.get("reward", 0.0)) if isinstance(resp, dict) else 0.0
            done = resp.get("done", False) if isinstance(resp, dict) else False
            step_num += 1
            rewards.append(reward)

            # Check for errors
            error = None
            if isinstance(obs_data, dict):
                error = obs_data.get("last_action_error")

            # [STEP]
            action_str = action_to_string(action)
            done_str = "true" if done else "false"
            error_str = str(error) if error else "null"
            print(
                f"[STEP] step={step_num} action={action_str} "
                f"reward={reward:.2f} done={done_str} error={error_str}"
            )
            sys.stdout.flush()

        # Compute score
        score = max(0.01, min(0.99, sum(rewards) / max(len(rewards), 1)))
        success = True

    except Exception as exc:
        print(f"[DEBUG] Task {task_id} error: {exc}", file=sys.stderr)
        score = 0.0
        success = False

    finally:
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        success_str = "true" if success else "false"
        print(
            f"[END] success={success_str} steps={step_num} "
            f"score={score:.2f} rewards={rewards_str}"
        )
        sys.stdout.flush()

    return score


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────


def _timeout_handler(signum, frame):
    """Hard kill if script runs too long."""
    print("[END] success=false steps=0 score=0.00 rewards=", flush=True)
    sys.exit(1)


def main() -> None:
    # Global 18-minute safety timeout
    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(18 * 60)
    except (AttributeError, ValueError):
        pass

    llm = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY or "dummy",
    )

    # Start the Docker container if IMAGE_NAME is set
    base_url = ENV_URL
    if IMAGE_NAME:
        print(f"[DEBUG] Starting container from image: {IMAGE_NAME}", file=sys.stderr)
        base_url = start_container(IMAGE_NAME)
        try:
            wait_for_ready(base_url)
            print(f"[DEBUG] Container ready at {base_url}", file=sys.stderr)
        except Exception as e:
            print(f"[DEBUG] Container not ready, trying anyway: {e}", file=sys.stderr)
    else:
        print(f"[DEBUG] No IMAGE_NAME, connecting to {base_url}", file=sys.stderr)

    env = SimpleEnvClient(base_url)

    try:
        for task_id in TASK_IDS:
            run_task(llm, env, task_id)
    finally:
        env.close()
        stop_container()


if __name__ == "__main__":
    main()
