"""
Baseline Inference — Simple rule-based agent for benchmarking.

Runs directly against the TrafficSimulation without an LLM.
Used to verify simulation correctness and generate baseline stats.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from ..models import (
    CoordinationMessage,
    MessageType,
    PhaseCommand,
    TrafficAction,
    Urgency,
)
from .environment import TrafficEnvironment
from .task import TASK_CONFIGS


class RuleBasedAgent:
    """
    Simple rule-based agent that:
    1. Gives green to the direction with the longest queue
    2. Reacts to emergency alerts by immediately switching phase
    3. Forwards emergency alerts when EVs exit its zone
    """

    def decide(self, obs_dict: Dict[str, Any]) -> TrafficAction:
        """Make a decision based on observation data."""
        commands: List[PhaseCommand] = []
        messages: List[CoordinationMessage] = []

        agent_cluster = obs_dict.get("agent_cluster", [])
        intersections = obs_dict.get("intersections", [])
        incoming = obs_dict.get("incoming_messages", [])
        evs = obs_dict.get("emergency_vehicles", [])

        # ── Handle emergency alerts first ──
        ev_targets = {}
        for msg in incoming:
            if isinstance(msg, dict):
                if msg.get("message_type") == "emergency_alert":
                    content = msg.get("content", "").lower()
                    target = msg.get("to_node", "")
                    if "northbound" in content or "southbound" in content:
                        ev_targets[target] = "NS_GREEN"
                    elif "eastbound" in content or "westbound" in content:
                        ev_targets[target] = "EW_GREEN"
            elif hasattr(msg, "message_type"):
                if msg.message_type == MessageType.EMERGENCY_ALERT:
                    content = msg.content.lower()
                    target = msg.to_node
                    if "northbound" in content or "southbound" in content:
                        ev_targets[target] = "NS_GREEN"
                    elif "eastbound" in content or "westbound" in content:
                        ev_targets[target] = "EW_GREEN"

        # ── Set phases for agent intersections ──
        for ix in intersections:
            if isinstance(ix, dict):
                i_id = ix.get("intersection_id", ix.get("id", ""))
                controlled = ix.get("controlled_by", "heuristic")
                current_phase = ix.get("current_phase", "NS_GREEN")
                min_green = ix.get("min_green_remaining", 0)
                approaches = ix.get("approaches", [])
            else:
                i_id = ix.intersection_id
                controlled = ix.controlled_by
                current_phase = ix.current_phase
                min_green = ix.min_green_remaining
                approaches = ix.approaches

            if i_id not in agent_cluster:
                continue
            if min_green > 0:
                continue

            # Emergency override
            if i_id in ev_targets:
                target_phase = ev_targets[i_id]
                if current_phase != target_phase:
                    commands.append(
                        PhaseCommand(intersection_id=i_id, target_phase=target_phase)
                    )
                continue

            # Normal: longest queue gets green
            ns_queue = 0
            ew_queue = 0
            if isinstance(approaches, list):
                for a in approaches:
                    if isinstance(a, dict):
                        d = a.get("direction", "")
                        q = a.get("queue_length", 0)
                    else:
                        d = a.direction.value if hasattr(a.direction, "value") else a.direction
                        q = a.queue_length
                    if d in ("north", "south"):
                        ns_queue += q
                    elif d in ("east", "west"):
                        ew_queue += q

            if ns_queue > ew_queue and current_phase != "NS_GREEN":
                commands.append(
                    PhaseCommand(intersection_id=i_id, target_phase="NS_GREEN")
                )
            elif ew_queue > ns_queue and current_phase != "EW_GREEN":
                commands.append(
                    PhaseCommand(intersection_id=i_id, target_phase="EW_GREEN")
                )

        return TrafficAction(
            commands=commands,
            messages=messages,
            reasoning="rule_based",
        )


def run_baseline_episode(task_id: str) -> Dict[str, Any]:
    """Run a complete episode with the rule-based agent."""
    env = TrafficEnvironment()
    agent = RuleBasedAgent()

    obs = env.reset(task_id=task_id)
    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        obs_dict = obs.model_dump()
        action = agent.decide(obs_dict)
        obs = env.step(action)
        total_reward += (obs.reward or 0.0)
        done = obs.done
        steps += 1

    state = env.state
    return {
        "task_id": task_id,
        "steps": steps,
        "total_reward": round(total_reward, 4),
        "avg_delay": state.avg_delay,
        "vehicles_cleared": state.vehicles_cleared,
        "ev_delay": state.total_emergency_delay,
    }


def run_all_baselines():
    """Run baselines for all tasks and print results."""
    print("=" * 60)
    print("Rule-Based Baseline Results")
    print("=" * 60)

    for task_id in TASK_CONFIGS:
        result = run_baseline_episode(task_id)
        print(f"\n{task_id}:")
        print(f"  Steps:      {result['steps']}")
        print(f"  Reward:     {result['total_reward']:.4f}")
        print(f"  Avg Delay:  {result['avg_delay']:.2f}s")
        print(f"  Cleared:    {result['vehicles_cleared']}")
        print(f"  EV Delay:   {result['ev_delay']:.2f}s")


if __name__ == "__main__":
    run_all_baselines()
