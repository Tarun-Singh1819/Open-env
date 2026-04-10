"""
Road — A cell-based road segment connecting two intersections.

Each road is an array of cells. Each cell holds at most 1 vehicle.
Vehicles move from cell 0 (far end) toward cell [length-1] (at the
downstream intersection). A vehicle at the last cell can exit only
if the downstream intersection has a green phase for its direction.
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .vehicle import Vehicle

STEP_DURATION = 5  # seconds per simulation step


class Road:
    """
    A one-directional road segment between two intersections.

    Attributes:
        source_id:  ID of the upstream intersection
        target_id:  ID of the downstream intersection
        direction:  Travel direction ("north", "south", "east", "west")
        length:     Number of cells
    """

    def __init__(
        self,
        source_id: str,
        target_id: str,
        direction: str,
        length: int = 10,
    ) -> None:
        self.source_id = source_id
        self.target_id = target_id
        self.direction = direction
        self.length = length

        # Cell array: each slot is None (empty) or a Vehicle
        self.cells: List[Optional[Vehicle]] = [None] * length

        # Capacity reduction factor (1.0 = full capacity, 0.5 = half)
        self._capacity_factor: float = 1.0

        # Stats
        self._throughput_this_step: int = 0

    # ─────────────────────────────────────────
    # Core movement
    # ─────────────────────────────────────────

    def tick(self, downstream_green: bool) -> dict:
        """
        Advance all vehicles by one step.

        Movement rules:
        - Vehicles move from cell[i] → cell[i+1] if cell[i+1] is empty
        - Vehicle at cell[length-1] exits if downstream intersection is green
        - Movement is applied right-to-left to avoid double-moves

        Args:
            downstream_green: True if the downstream intersection has green
                              for this road's direction.

        Returns:
            Dict with step events:
              - "exited": list of vehicles that exited the road
              - "moved": count of vehicles that moved forward
              - "waiting": count of vehicles that couldn't move
        """
        exited: List[Vehicle] = []
        moved = 0
        waiting = 0

        # Process right-to-left (from intersection back to entry)
        for i in range(self.length - 1, -1, -1):
            vehicle = self.cells[i]
            if vehicle is None:
                continue

            if i == self.length - 1:
                # At the intersection — can exit if green
                if downstream_green:
                    vehicle.tick_move()
                    vehicle.cleared = True
                    exited.append(vehicle)
                    self.cells[i] = None
                    moved += 1
                else:
                    vehicle.tick_wait(STEP_DURATION)
                    waiting += 1
            else:
                # Mid-road — move forward if next cell is empty
                if self.cells[i + 1] is None:
                    vehicle.tick_move()
                    self.cells[i + 1] = vehicle
                    self.cells[i] = None
                    moved += 1
                else:
                    vehicle.tick_wait(STEP_DURATION)
                    waiting += 1

        self._throughput_this_step = len(exited)
        return {
            "exited": exited,
            "moved": moved,
            "waiting": waiting,
        }

    # ─────────────────────────────────────────
    # Vehicle insertion
    # ─────────────────────────────────────────

    def try_insert(self, vehicle: Vehicle) -> bool:
        """
        Try to insert a vehicle at the entry end (cell 0).
        Returns True if successful, False if cell 0 is occupied.
        """
        if self.cells[0] is None:
            self.cells[0] = vehicle
            return True
        return False

    def try_insert_at(self, vehicle: Vehicle, position: int) -> bool:
        """Insert a vehicle at a specific cell position."""
        if 0 <= position < self.length and self.cells[position] is None:
            self.cells[position] = vehicle
            return True
        return False

    # ─────────────────────────────────────────
    # Queries
    # ─────────────────────────────────────────

    @property
    def vehicle_count(self) -> int:
        """Total vehicles currently on this road."""
        return sum(1 for c in self.cells if c is not None)

    @property
    def queue_length(self) -> int:
        """
        Vehicles queued at the downstream end (consecutive occupied cells
        starting from the intersection end).
        """
        count = 0
        for i in range(self.length - 1, -1, -1):
            if self.cells[i] is not None:
                count += 1
            else:
                break
        return count

    @property
    def approaching_vehicles(self) -> int:
        """Vehicles NOT in the queue (still moving in approach zone)."""
        return self.vehicle_count - self.queue_length

    @property
    def occupancy_percent(self) -> float:
        """Percentage of road cells occupied."""
        if self.length == 0:
            return 0.0
        return (self.vehicle_count / self.length) * 100.0

    @property
    def is_near_capacity(self) -> bool:
        """True if road is >= 80% full."""
        return self.occupancy_percent >= 80.0

    @property
    def effective_length(self) -> int:
        """Effective capacity considering incidents."""
        return max(1, int(self.length * self._capacity_factor))

    def get_avg_wait(self) -> float:
        """Average wait time of vehicles on this road."""
        vehicles = [c for c in self.cells if c is not None]
        if not vehicles:
            return 0.0
        return sum(v.wait_time for v in vehicles) / len(vehicles)

    def get_max_wait(self) -> float:
        """Max wait time of any vehicle on this road."""
        vehicles = [c for c in self.cells if c is not None]
        if not vehicles:
            return 0.0
        return max(v.wait_time for v in vehicles)

    def get_eta_steps(self) -> int:
        """
        Estimated steps for a vehicle at cell 0 to reach the intersection.
        Simplified: assumes no blocking = road length steps.
        """
        return self.length

    # ─────────────────────────────────────────
    # Incidents
    # ─────────────────────────────────────────

    def apply_incident(self, capacity_reduction: float) -> None:
        """
        Reduce road capacity (simulates lane closure / accident).
        capacity_reduction: 0.0–1.0, fraction of capacity LOST.
        """
        self._capacity_factor = max(0.0, 1.0 - capacity_reduction)
        # Block cells beyond effective length
        effective = self.effective_length
        for i in range(effective, self.length):
            if self.cells[i] is not None:
                # Push vehicle backward if possible
                for j in range(effective - 1, -1, -1):
                    if self.cells[j] is None:
                        self.cells[j] = self.cells[i]
                        self.cells[i] = None
                        break

    def clear_incident(self) -> None:
        """Restore full road capacity."""
        self._capacity_factor = 1.0

    # ─────────────────────────────────────────
    # Serialization
    # ─────────────────────────────────────────

    def to_link_dict(self) -> dict:
        """Serialize for environment queries."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "direction": self.direction,
            "vehicles_in_transit": self.vehicle_count,
            "queue_length": self.queue_length,
            "occupancy_pct": self.occupancy_percent,
            "eta_steps": self.get_eta_steps(),
        }
