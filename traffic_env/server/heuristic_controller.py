"""
HeuristicController — Standalone heuristic controller for reference/testing.

Note: The core heuristic logic lives INSIDE Intersection.tick_heuristic().
This module provides a higher-level wrapper for testing heuristic-only
baselines and generating the fixed-timing baseline statistics.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .intersection import Intersection
from .simulation import TrafficSimulation
from .task import TaskConfig, TASK_CONFIGS

STEP_DURATION = 5  # seconds per step


class HeuristicController:
    """
    Runs a full episode with all intersections on fixed-timing heuristic.
    Used to compute baseline delay statistics for grading.
    """

    def __init__(self, task_config: TaskConfig) -> None:
        self._config = task_config
        self._sim: TrafficSimulation | None = None

    def run_baseline(self) -> Dict[str, Any]:
        """
        Run a full episode using only heuristic controllers.
        Returns aggregate performance metrics.
        """
        # Override: make ALL intersections heuristic
        config_copy = TaskConfig(
            task_id=self._config.task_id,
            name=self._config.name,
            difficulty=self._config.difficulty,
            description=self._config.description,
            max_steps=self._config.max_steps,
            seed=self._config.seed,
            intersection_ids=self._config.intersection_ids,
            links=self._config.links,
            agent_cluster=[],  # No agent — all heuristic
            heuristic_cluster=list(self._config.intersection_ids),
            generation_rates=self._config.generation_rates,
            emergency_vehicles=self._config.emergency_vehicles,
            incidents=self._config.incidents,
            baseline_avg_delay=self._config.baseline_avg_delay,
        )

        self._sim = TrafficSimulation(config_copy)

        # Force all intersections to heuristic mode
        for intersection in self._sim._intersections.values():
            intersection.is_heuristic = True

        # Run episode
        for step in range(config_copy.max_steps):
            self._sim.tick()

        # Collect results
        metrics = self._sim.get_network_metrics()

        # Per-approach delays for equity analysis
        all_intersections = self._sim.get_all_intersections()
        approach_delays: List[float] = []
        for i_data in all_intersections:
            for direction, approach in i_data["approaches"].items():
                if approach.get("wait_avg", 0) > 0:
                    approach_delays.append(approach["wait_avg"])

        return {
            "avg_delay": metrics.get("avg_delay", 0.0),
            "max_delay": metrics.get("max_delay", 0.0),
            "total_cleared": metrics.get("total_cleared", 0),
            "total_generated": metrics.get("total_generated", 0),
            "ev_total_delay": metrics.get("total_ev_delay", 0.0),
            "approach_delays": approach_delays,
            "max_approach_delay": max(approach_delays) if approach_delays else 0.0,
        }


def compute_all_baselines() -> Dict[str, Dict[str, Any]]:
    """
    Compute and print baseline statistics for all tasks.
    Used during development to calibrate baseline_avg_delay values.
    """
    results = {}
    for task_id, config in TASK_CONFIGS.items():
        print(f"Computing baseline for {task_id}...")
        controller = HeuristicController(config)
        baseline = controller.run_baseline()
        results[task_id] = baseline
        print(f"  avg_delay={baseline['avg_delay']:.2f}s")
        print(f"  max_delay={baseline['max_delay']:.2f}s")
        print(f"  cleared={baseline['total_cleared']}")
        print(f"  ev_delay={baseline['ev_total_delay']:.2f}s")
        print()
    return results


if __name__ == "__main__":
    compute_all_baselines()
