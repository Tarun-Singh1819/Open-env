"""
Intersection — Signal phase management and approach tracking.

Handles phase transitions (min green → yellow → all-red → new phase),
heuristic controller behavior, and emergency pre-emption from messages.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import CoordinationMessage

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

MIN_GREEN_STEPS = 3     # 15 seconds minimum green
YELLOW_STEPS = 1        # 5 seconds yellow
ALL_RED_STEPS = 1       # 5 seconds all-red clearance

# Heuristic controller cycle: fixed green durations per phase
HEURISTIC_CYCLE = [
    ("NS_GREEN", 6),     # 30 seconds green
    ("EW_GREEN", 6),     # 30 seconds green
]


class Intersection:
    """
    Manages signal phase state and transitions for one intersection.

    Phase transition sequence:
        Current Green (min green met) → YELLOW (1 step) → ALL_RED (1 step) → New Green

    The intersection can be controlled by either:
    - The agent (via request_phase_change)
    - A heuristic controller (via tick_heuristic)
    """

    def __init__(self, intersection_id: str, is_heuristic: bool = False) -> None:
        self.intersection_id = intersection_id
        self.is_heuristic = is_heuristic

        # Phase state
        self._current_phase: str = "NS_GREEN"
        self._phase_elapsed: int = 0          # Steps in current phase
        self._target_phase: Optional[str] = None  # Phase to transition to
        self._in_transition: bool = False      # In yellow/all-red

        # Heuristic controller state
        self._heuristic_cycle_index: int = 0
        self._heuristic_steps_in_phase: int = 0
        self._emergency_preemption: Optional[str] = None  # Phase to switch to ASAP
        self._extend_current_phase: int = 0  # Extra steps to hold current phase
        self._reduce_next_phase: bool = False

        # Per-approach throughput tracking
        self._throughput_last_phase: Dict[str, int] = {
            "north": 0, "south": 0, "east": 0, "west": 0,
        }
        self._throughput_current: Dict[str, int] = {
            "north": 0, "south": 0, "east": 0, "west": 0,
        }
        self._total_throughput: int = 0

    # ─────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────

    @property
    def current_phase(self) -> str:
        return self._current_phase

    @property
    def phase_elapsed(self) -> int:
        return self._phase_elapsed

    @property
    def min_green_remaining(self) -> int:
        """Steps left before phase can be changed."""
        if self._current_phase in ("YELLOW", "ALL_RED"):
            return 0  # Transition phases complete automatically
        remaining = MIN_GREEN_STEPS - self._phase_elapsed
        return max(0, remaining)

    @property
    def can_switch(self) -> bool:
        """Whether the phase can be changed right now."""
        if self._in_transition:
            return False
        return self.min_green_remaining == 0

    # ─────────────────────────────────────────
    # Agent-controlled phase changes
    # ─────────────────────────────────────────

    def request_phase_change(self, target_phase: str) -> bool:
        """
        Request a phase change (from the agent).
        Returns True if the transition was initiated.
        Returns False if:
        - Currently in transition (yellow/all-red)
        - Minimum green time hasn't elapsed
        - Already in the target phase
        """
        if target_phase == self._current_phase:
            return False
        if self._in_transition:
            return False
        if not self.can_switch:
            return False
        if target_phase in ("YELLOW", "ALL_RED"):
            return False  # Not selectable

        # Initiate: current → YELLOW → ALL_RED → target
        self._target_phase = target_phase
        self._begin_transition()
        return True

    def _begin_transition(self) -> None:
        """Start yellow phase."""
        # Save throughput for the phase that's ending
        self._throughput_last_phase = dict(self._throughput_current)
        self._throughput_current = {
            "north": 0, "south": 0, "east": 0, "west": 0,
        }

        self._current_phase = "YELLOW"
        self._phase_elapsed = 0
        self._in_transition = True

    # ─────────────────────────────────────────
    # Tick (advance one step)
    # ─────────────────────────────────────────

    def tick(self) -> None:
        """
        Advance phase timer by one step.
        Handles automatic yellow → all-red → target transitions.
        """
        self._phase_elapsed += 1

        if self._in_transition:
            if self._current_phase == "YELLOW" and self._phase_elapsed >= YELLOW_STEPS:
                # YELLOW complete → ALL_RED
                self._current_phase = "ALL_RED"
                self._phase_elapsed = 0

            elif self._current_phase == "ALL_RED" and self._phase_elapsed >= ALL_RED_STEPS:
                # ALL_RED complete → new green phase
                self._current_phase = self._target_phase or "NS_GREEN"
                self._phase_elapsed = 0
                self._target_phase = None
                self._in_transition = False

    def tick_heuristic(self) -> None:
        """
        Advance the heuristic (fixed-timing) controller.
        Called only for heuristic-controlled intersections.
        """
        # Handle emergency pre-emption first
        if self._emergency_preemption is not None:
            needed_phase = self._emergency_preemption
            if self._current_phase != needed_phase and not self._in_transition:
                if self.can_switch:
                    self._target_phase = needed_phase
                    self._begin_transition()
                    self._emergency_preemption = None
            elif self._current_phase == needed_phase:
                self._emergency_preemption = None
                # Extend green for the EV
                self._extend_current_phase = max(self._extend_current_phase, 4)

        # Normal tick (handle transitions)
        self.tick()

        # If not in transition, manage the fixed-timing cycle
        if not self._in_transition and self._current_phase not in ("YELLOW", "ALL_RED"):
            self._heuristic_steps_in_phase += 1

            # Check if current phase duration is complete
            _, duration = HEURISTIC_CYCLE[self._heuristic_cycle_index]
            effective_duration = duration + self._extend_current_phase

            if self._reduce_next_phase:
                effective_duration = max(MIN_GREEN_STEPS, duration - 2)
                self._reduce_next_phase = False

            if self._heuristic_steps_in_phase >= effective_duration:
                # Move to next phase in cycle
                self._extend_current_phase = 0
                self._heuristic_cycle_index = (
                    (self._heuristic_cycle_index + 1) % len(HEURISTIC_CYCLE)
                )
                next_phase, _ = HEURISTIC_CYCLE[self._heuristic_cycle_index]

                if next_phase != self._current_phase:
                    self._target_phase = next_phase
                    self._begin_transition()
                    self._heuristic_steps_in_phase = 0

    # ─────────────────────────────────────────
    # Message reactions (heuristic only)
    # ─────────────────────────────────────────

    def queue_emergency_preemption(self, msg: CoordinationMessage) -> None:
        """
        Queue an emergency pre-emption. The heuristic controller will
        switch to the appropriate phase as soon as possible.
        """
        # Parse the suggested action to determine needed phase
        phase = self._parse_phase_from_message(msg)
        self._emergency_preemption = phase

    def prepare_for_incoming_traffic(self, msg: CoordinationMessage) -> None:
        """
        Extend current phase if it matches the incoming traffic direction,
        or prepare to switch to the matching phase sooner.
        """
        phase = self._parse_phase_from_message(msg)
        if self._current_phase == phase:
            self._extend_current_phase += 2  # Hold green 2 extra steps
        # Otherwise the cycle will naturally get there

    def reduce_outflow(self, msg: CoordinationMessage) -> None:
        """Reduce the next green phase duration to slow outflow."""
        self._reduce_next_phase = True

    def _parse_phase_from_message(self, msg: CoordinationMessage) -> str:
        """Extract the needed phase from a coordination message."""
        if msg.suggested_action:
            sa = msg.suggested_action.upper()
            if "NS_GREEN" in sa:
                return "NS_GREEN"
            if "EW_GREEN" in sa:
                return "EW_GREEN"
            if "NS_LEFT" in sa:
                return "NS_LEFT"
            if "EW_LEFT" in sa:
                return "EW_LEFT"

        # Fallback: parse content
        content = msg.content.lower()
        if "northbound" in content or "southbound" in content:
            return "NS_GREEN"
        if "eastbound" in content or "westbound" in content:
            return "EW_GREEN"

        return "NS_GREEN"

    # ─────────────────────────────────────────
    # Throughput tracking
    # ─────────────────────────────────────────

    def record_vehicle_exit(self, direction: str) -> None:
        """Record a vehicle clearing through this intersection."""
        self._throughput_current[direction] = (
            self._throughput_current.get(direction, 0) + 1
        )
        self._total_throughput += 1

    # ─────────────────────────────────────────
    # Green direction queries
    # ─────────────────────────────────────────

    def is_green_for(self, direction: str) -> bool:
        """Check if the current phase gives green to this direction."""
        phase = self._current_phase
        if phase == "NS_GREEN":
            return direction in ("north", "south")
        if phase == "NS_LEFT":
            return direction in ("north", "south")
        if phase == "EW_GREEN":
            return direction in ("east", "west")
        if phase == "EW_LEFT":
            return direction in ("east", "west")
        # YELLOW, ALL_RED → no green for anyone
        return False

    # ─────────────────────────────────────────
    # Serialization
    # ─────────────────────────────────────────

    def to_dict(self, approach_data: Dict[str, dict]) -> dict:
        """
        Serialize intersection state for observation building.

        Args:
            approach_data: Per-direction stats from the road segments
                          {direction: {queue_length, approaching, wait_avg, ...}}
        """
        total_waiting = sum(
            a.get("queue_length", 0) for a in approach_data.values()
        )

        return {
            "id": self.intersection_id,
            "current_phase": self._current_phase,
            "phase_elapsed": self._phase_elapsed,
            "min_green_remaining": self.min_green_remaining,
            "approaches": approach_data,
            "total_waiting": total_waiting,
            "total_throughput": self._total_throughput,
            "throughput_last_phase": dict(self._throughput_last_phase),
        }
