"""
Vehicle classes for the cell-based traffic simulation.

Regular vehicles move 1 cell per step if the cell ahead is empty.
Emergency vehicles follow pre-defined routes and accumulate delay
when stopped at red lights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Vehicle:
    """A regular vehicle in the simulation."""
    vehicle_id: int
    spawn_step: int                 # Step when this vehicle entered the network
    direction: str                  # "north", "south", "east", "west"
    wait_time: float = 0.0         # Accumulated wait time (seconds)
    current_step_wait: float = 0.0 # Wait time accumulated THIS step only
    is_waiting: bool = False       # True if stopped (at red or blocked)
    cleared: bool = False          # True if exited the network

    def tick_wait(self, step_duration: float) -> None:
        """Accumulate wait time for one stuck step."""
        self.wait_time += step_duration
        self.current_step_wait = step_duration
        self.is_waiting = True

    def tick_move(self) -> None:
        """Mark vehicle as having moved this step."""
        self.current_step_wait = 0.0
        self.is_waiting = False


@dataclass
class EmergencyVehicle:
    """
    An emergency vehicle following a pre-defined route.
    EVs wait at red lights (they do NOT run reds) to maximize
    the agent's impact on their travel time.
    """
    ev_id: str
    priority: int                  # 1=ambulance, 2=fire, 3=police
    route: List[str]               # Ordered list of intersection IDs
    spawn_step: int
    initial_direction: str         # Approach direction at first intersection

    # ── Runtime state ──
    active: bool = False           # True once spawned
    cleared: bool = False          # True once route completed
    current_route_index: int = 0   # Which intersection in route we're heading to
    current_road: str = ""         # e.g., "I1_to_I2", or "external_to_I1"
    cell_position: int = 0         # Position on current road segment (0 = far, 9 = at intersection)
    direction: str = ""            # Current approach direction

    # ── Metrics ──
    total_delay: float = 0.0      # Total accumulated delay (seconds)
    current_step_wait: float = 0.0
    is_waiting: bool = False
    delay_per_intersection: dict = field(default_factory=dict)  # {intersection_id: delay_seconds}

    def __post_init__(self) -> None:
        self.direction = self.initial_direction

    def spawn(self) -> None:
        """Activate the EV at its starting position."""
        self.active = True
        first_target = self.route[0]
        self.current_road = f"external_to_{first_target}"
        self.cell_position = 0     # Far end of road (will move toward intersection)
        self.current_route_index = 0

    @property
    def target_intersection(self) -> Optional[str]:
        """The next intersection this EV is heading toward."""
        if self.current_route_index < len(self.route):
            return self.route[self.current_route_index]
        return None

    @property
    def distance_to_intersection(self) -> int:
        """Cells away from the next intersection (intersection is at cell road_length-1)."""
        # This depends on road length; will be set by the simulation
        return self.cell_position

    def tick_wait(self, step_duration: float) -> None:
        """EV is stopped at a red light."""
        self.total_delay += step_duration
        self.current_step_wait = step_duration
        self.is_waiting = True
        # Track per-intersection delay
        target = self.target_intersection
        if target:
            self.delay_per_intersection[target] = (
                self.delay_per_intersection.get(target, 0.0) + step_duration
            )

    def tick_move(self) -> None:
        """EV moved forward one cell."""
        self.current_step_wait = 0.0
        self.is_waiting = False

    def advance_to_next_intersection(self) -> Optional[str]:
        """
        Called when EV passes through current target intersection.
        Returns the NEXT intersection ID, or None if route is complete.
        """
        prev_target = self.target_intersection
        self.current_route_index += 1

        if self.current_route_index >= len(self.route):
            # Route complete
            self.cleared = True
            self.active = False
            return None

        next_target = self.route[self.current_route_index]
        self.current_road = f"{prev_target}_to_{next_target}"
        self.cell_position = 0  # Start of new road segment

        # Update direction based on topology
        # (will be set by simulation based on link direction)
        return next_target

    def to_dict(self) -> dict:
        """Serialize to dict for observation building."""
        return {
            "ev_id": self.ev_id,
            "current_road": self.current_road,
            "distance": self.distance_to_intersection,
            "target_intersection": self.target_intersection or "",
            "direction": self.direction,
            "priority": self.priority,
            "waiting": self.is_waiting,
            "total_delay": self.total_delay,
            "current_step_wait": self.current_step_wait,
        }
