"""
Task configurations for the 3 traffic control tasks.

Each TaskConfig defines the network topology, traffic generation rates,
emergency vehicle schedules, incidents, and grading baselines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class EVConfig:
    """Emergency vehicle spawn configuration."""
    ev_id: str
    spawn_step: int
    route: List[str]          # List of intersection IDs in order
    priority: int             # 1=ambulance, 2=fire, 3=police
    direction: str = "south"  # Initial approach direction at first intersection


@dataclass
class IncidentConfig:
    """Road incident configuration (capacity reduction)."""
    step: int
    link: str                 # e.g., "I1_I3" — affected road segment
    capacity_reduction: float # 0.0–1.0, fraction of capacity lost
    duration: int             # Steps the incident lasts


@dataclass
class LinkConfig:
    """Configuration for a road link between two intersections."""
    source_id: str
    target_id: str
    direction: str            # Direction of travel: "north", "south", "east", "west"
    length_cells: int = 10    # Number of cells in the road segment


@dataclass
class TaskConfig:
    """Complete configuration for one task."""
    task_id: str
    name: str
    difficulty: str           # "easy", "medium", "hard"
    description: str
    max_steps: int
    seed: int

    # Network topology
    intersection_ids: List[str]
    links: List[LinkConfig]
    agent_cluster: List[str]
    heuristic_cluster: List[str]

    # Traffic generation rates per approach per step
    # Key: "{intersection_id}_{direction}" → List[float] (one rate per step)
    generation_rates: Dict[str, List[float]]

    # Emergency vehicles
    emergency_vehicles: List[EVConfig] = field(default_factory=list)

    # Incidents
    incidents: List[IncidentConfig] = field(default_factory=list)

    # Pre-computed baseline stats for grading
    baseline_avg_delay: float = 30.0


# ─────────────────────────────────────────────
# Helper to create constant or time-varying rates
# ─────────────────────────────────────────────

def _const_rate(rate: float, steps: int) -> List[float]:
    """Constant generation rate for all steps."""
    return [rate] * steps


def _ramp_rate(rates_and_durations: List[Tuple[float, int]]) -> List[float]:
    """Piecewise constant rate: [(rate, duration_steps), ...]"""
    result: List[float] = []
    for rate, duration in rates_and_durations:
        result.extend([rate] * duration)
    return result


# ─────────────────────────────────────────────
# Task 1: Single Intersection Rush Hour
# ─────────────────────────────────────────────

_TASK1_STEPS = 30

TASK1_CONFIG = TaskConfig(
    task_id="single_intersection",
    name="Single Intersection Rush Hour",
    difficulty="easy",
    description=(
        "Optimize signal timing for a single intersection with "
        "asymmetric traffic demand. NS is ~3x heavier than EW."
    ),
    max_steps=_TASK1_STEPS,
    seed=42,
    intersection_ids=["I1"],
    links=[],  # No links — single intersection
    agent_cluster=["I1"],
    heuristic_cluster=[],
    generation_rates={
        "I1_north": _const_rate(0.40, _TASK1_STEPS),  # Heavy NS
        "I1_south": _const_rate(0.35, _TASK1_STEPS),  # Heavy NS
        "I1_east":  _const_rate(0.12, _TASK1_STEPS),  # Light EW
        "I1_west":  _const_rate(0.12, _TASK1_STEPS),  # Light EW
    },
    emergency_vehicles=[],
    incidents=[],
    baseline_avg_delay=35.0,  # Fixed 30s/30s cycle achieves ~35s avg delay
)


# ─────────────────────────────────────────────
# Task 2: Corridor Green Wave with Emergency
# ─────────────────────────────────────────────

_TASK2_STEPS = 50

TASK2_CONFIG = TaskConfig(
    task_id="corridor_green_wave",
    name="Corridor Green Wave with Emergency",
    difficulty="medium",
    description=(
        "Control 2 intersections in a 3-intersection corridor. "
        "Coordinate green wave timing, handle an emergency vehicle, "
        "and communicate with the heuristic-controlled third intersection."
    ),
    max_steps=_TASK2_STEPS,
    seed=123,
    intersection_ids=["I1", "I2", "I3"],
    links=[
        LinkConfig("I1", "I2", "east",  10),
        LinkConfig("I2", "I1", "west",  10),
        LinkConfig("I2", "I3", "east",  10),
        LinkConfig("I3", "I2", "west",  10),
    ],
    agent_cluster=["I1", "I2"],
    heuristic_cluster=["I3"],
    generation_rates={
        # Corridor traffic (heavy, bidirectional)
        "I1_east":  _const_rate(0.35, _TASK2_STEPS),
        "I3_west":  _const_rate(0.35, _TASK2_STEPS),
        # Cross traffic (moderate)
        "I1_north": _const_rate(0.15, _TASK2_STEPS),
        "I1_south": _const_rate(0.15, _TASK2_STEPS),
        "I2_north": _const_rate(0.20, _TASK2_STEPS),
        "I2_south": _const_rate(0.20, _TASK2_STEPS),
        "I3_north": _const_rate(0.15, _TASK2_STEPS),
        "I3_south": _const_rate(0.15, _TASK2_STEPS),
        # External approaches for edge intersections
        "I1_west":  _const_rate(0.10, _TASK2_STEPS),
        "I3_east":  _const_rate(0.10, _TASK2_STEPS),
        "I2_east":  _const_rate(0.0, _TASK2_STEPS),  # Internal only
        "I2_west":  _const_rate(0.0, _TASK2_STEPS),  # Internal only
    },
    emergency_vehicles=[
        EVConfig(
            ev_id="AMB-1",
            spawn_step=15,
            route=["I1", "I2", "I3"],
            priority=1,
            direction="east",
        ),
    ],
    incidents=[],
    baseline_avg_delay=42.0,
)


# ─────────────────────────────────────────────
# Task 3: Grid Network Incident Response
# ─────────────────────────────────────────────

_TASK3_STEPS = 70

TASK3_CONFIG = TaskConfig(
    task_id="grid_incident",
    name="Grid Network Incident Response",
    difficulty="hard",
    description=(
        "Control 2 intersections in a 2x2 grid network. Handle demand "
        "shifts, road capacity reduction, and 2 competing emergency "
        "vehicles. Coordinate with 2 heuristic-controlled intersections."
    ),
    max_steps=_TASK3_STEPS,
    seed=456,
    intersection_ids=["I1", "I2", "I3", "I4"],
    links=[
        # Top row (horizontal)
        LinkConfig("I1", "I2", "east",  10),
        LinkConfig("I2", "I1", "west",  10),
        # Bottom row (horizontal)
        LinkConfig("I3", "I4", "east",  10),
        LinkConfig("I4", "I3", "west",  10),
        # Left column (vertical)
        LinkConfig("I1", "I3", "south", 10),
        LinkConfig("I3", "I1", "north", 10),
        # Right column (vertical)
        LinkConfig("I2", "I4", "south", 10),
        LinkConfig("I4", "I2", "north", 10),
    ],
    agent_cluster=["I1", "I2"],       # Top row
    heuristic_cluster=["I3", "I4"],   # Bottom row
    generation_rates={
        # I1 — rush hour build-up pattern
        "I1_north": _ramp_rate([(0.15, 20), (0.45, 30), (0.20, 20)]),
        "I1_south": _ramp_rate([(0.10, 20), (0.30, 30), (0.15, 20)]),
        "I1_east":  _ramp_rate([(0.20, 20), (0.40, 30), (0.20, 20)]),
        "I1_west":  _ramp_rate([(0.10, 20), (0.15, 30), (0.10, 20)]),
        # I2 — moderate with demand shift
        "I2_north": _ramp_rate([(0.15, 23), (0.35, 24), (0.20, 23)]),
        "I2_south": _ramp_rate([(0.10, 23), (0.25, 24), (0.15, 23)]),
        "I2_east":  _ramp_rate([(0.10, 23), (0.20, 24), (0.10, 23)]),
        "I2_west":  _ramp_rate([(0.15, 23), (0.30, 24), (0.15, 23)]),
        # I3 — heuristic controlled, moderate traffic
        "I3_north": _const_rate(0.15, _TASK3_STEPS),
        "I3_south": _ramp_rate([(0.15, 23), (0.35, 24), (0.20, 23)]),
        "I3_east":  _const_rate(0.15, _TASK3_STEPS),
        "I3_west":  _const_rate(0.10, _TASK3_STEPS),
        # I4 — heuristic controlled, moderate traffic
        "I4_north": _ramp_rate([(0.10, 23), (0.30, 24), (0.15, 23)]),
        "I4_south": _const_rate(0.15, _TASK3_STEPS),
        "I4_east":  _const_rate(0.10, _TASK3_STEPS),
        "I4_west":  _const_rate(0.15, _TASK3_STEPS),
    },
    emergency_vehicles=[
        EVConfig(
            ev_id="AMB-1",
            spawn_step=15,
            route=["I4", "I2"],
            priority=1,
            direction="north",
        ),
        EVConfig(
            ev_id="FIRE-1",
            spawn_step=40,
            route=["I3", "I1"],
            priority=2,
            direction="north",
        ),
    ],
    incidents=[
        IncidentConfig(
            step=20,
            link="I1_I3",
            capacity_reduction=0.5,
            duration=35,
        ),
    ],
    baseline_avg_delay=55.0,
)


# ─────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────

TASK_CONFIGS: Dict[str, TaskConfig] = {
    "single_intersection": TASK1_CONFIG,
    "corridor_green_wave": TASK2_CONFIG,
    "grid_incident":       TASK3_CONFIG,
}
