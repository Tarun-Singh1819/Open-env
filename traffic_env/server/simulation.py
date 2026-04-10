"""
TrafficSimulation — Core cell-based traffic simulation engine.

Orchestrates intersections, road segments, vehicle generation,
emergency vehicle routing, and incident management. Provides the
query interfaces that TrafficEnvironment calls.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from .intersection import Intersection
from .road import Road
from .vehicle import EmergencyVehicle, Vehicle
from .task import TaskConfig, LinkConfig, EVConfig, IncidentConfig

STEP_DURATION = 5  # seconds per step


class TrafficSimulation:
    """
    Deterministic cell-based traffic simulation.

    Built from a TaskConfig. Manages:
    - Intersections (agent + heuristic)
    - Road segments (cell arrays)
    - Vehicle generation (seeded RNG)
    - Emergency vehicle routing
    - Incidents (capacity reduction)
    """

    def __init__(self, config: TaskConfig) -> None:
        self._config = config
        self._rng = random.Random(config.seed)
        self._step: int = 0
        self._vehicle_counter: int = 0

        # ── Build intersections ──
        self._intersections: Dict[str, Intersection] = {}
        for i_id in config.intersection_ids:
            is_heuristic = i_id in config.heuristic_cluster
            self._intersections[i_id] = Intersection(i_id, is_heuristic)

        # ── Build road segments ──
        # Key: "source_target" → Road
        self._roads: Dict[str, Road] = {}
        for link in config.links:
            key = f"{link.source_id}_{link.target_id}"
            self._roads[key] = Road(
                source_id=link.source_id,
                target_id=link.target_id,
                direction=link.direction,
                length=link.length_cells,
            )

        # ── Build external approach roads ──
        # For each intersection+direction that has a generation rate but no
        # incoming road from another intersection, create an "external" road.
        self._external_roads: Dict[str, Road] = {}
        for rate_key in config.generation_rates:
            # rate_key format: "I1_north"
            parts = rate_key.rsplit("_", 1)
            if len(parts) != 2:
                continue
            i_id, direction = parts
            if i_id not in self._intersections:
                continue

            # Check if there's already an internal road feeding this approach
            has_internal = any(
                road.target_id == i_id and road.direction == direction
                for road in self._roads.values()
            )
            if not has_internal:
                ext_key = f"ext_{i_id}_{direction}"
                self._external_roads[ext_key] = Road(
                    source_id="external",
                    target_id=i_id,
                    direction=direction,
                    length=10,
                )

        # ── Emergency vehicles ──
        self._emergency_vehicles: List[EmergencyVehicle] = []
        for ev_cfg in config.emergency_vehicles:
            self._emergency_vehicles.append(
                EmergencyVehicle(
                    ev_id=ev_cfg.ev_id,
                    priority=ev_cfg.priority,
                    route=ev_cfg.route,
                    spawn_step=ev_cfg.spawn_step,
                    initial_direction=ev_cfg.direction,
                )
            )

        # ── Incident tracking ──
        self._active_incidents: Dict[str, IncidentConfig] = {}

        # ── Cumulative stats ──
        self._total_generated: int = 0
        self._total_cleared: int = 0
        self._all_vehicle_delays: List[float] = []
        self._step_metrics: Dict[str, Any] = {}

    # ─────────────────────────────────────────
    # Main tick
    # ─────────────────────────────────────────

    def tick(self) -> Dict[str, Any]:
        """
        Advance simulation by one step. Returns event dict:
        {
            "ev_exits":  [{ev_id, next_node, intersection_id}, ...],
            "ev_passes": [{ev_id, intersection_id, delay}, ...],
        }
        """
        events: Dict[str, List] = {"ev_exits": [], "ev_passes": []}

        # 1. Handle incidents
        self._update_incidents()

        # 2. Spawn new vehicles
        self._generate_vehicles()

        # 3. Spawn / advance emergency vehicles
        self._handle_emergency_vehicles(events)

        # 4. Advance all intersections (phase timers)
        for intersection in self._intersections.values():
            if intersection.is_heuristic:
                intersection.tick_heuristic()
            else:
                intersection.tick()

        # 5. Move vehicles on all roads
        vehicles_moved = 0
        vehicles_waiting = 0
        vehicles_could_move = 0

        for road in self._all_roads():
            target_intersection = self._intersections.get(road.target_id)
            downstream_green = False
            if target_intersection:
                downstream_green = target_intersection.is_green_for(road.direction)

            result = road.tick(downstream_green)

            vehicles_moved += result["moved"]
            vehicles_waiting += result["waiting"]
            vehicles_could_move += result["moved"] + result["waiting"]

            # Record exits at the intersection
            for v in result["exited"]:
                if target_intersection:
                    target_intersection.record_vehicle_exit(road.direction)
                self._total_cleared += 1
                self._all_vehicle_delays.append(v.wait_time)

        # 6. Compute step metrics
        self._compute_step_metrics(vehicles_moved, vehicles_waiting, vehicles_could_move)

        self._step += 1
        return events

    # ─────────────────────────────────────────
    # Vehicle generation
    # ─────────────────────────────────────────

    def _generate_vehicles(self) -> None:
        """
        Generate vehicles based on per-approach rates.
        Uses seeded RNG for determinism.
        """
        for rate_key, rates in self._config.generation_rates.items():
            if self._step >= len(rates):
                continue
            rate = rates[self._step]
            if rate <= 0:
                continue

            # Bernoulli trial: spawn with probability = rate
            if self._rng.random() < rate:
                parts = rate_key.rsplit("_", 1)
                if len(parts) != 2:
                    continue
                i_id, direction = parts

                vehicle = Vehicle(
                    vehicle_id=self._vehicle_counter,
                    spawn_step=self._step,
                    direction=direction,
                )
                self._vehicle_counter += 1

                # Find the road to insert into
                road = self._find_approach_road(i_id, direction)
                if road and road.try_insert(vehicle):
                    self._total_generated += 1

    def _find_approach_road(self, target_id: str, direction: str) -> Optional[Road]:
        """Find the road that feeds into target_id from the given direction."""
        # Check internal roads first
        for road in self._roads.values():
            if road.target_id == target_id and road.direction == direction:
                return road

        # Check external approach roads
        ext_key = f"ext_{target_id}_{direction}"
        return self._external_roads.get(ext_key)

    # ─────────────────────────────────────────
    # Emergency vehicles
    # ─────────────────────────────────────────

    def _handle_emergency_vehicles(self, events: Dict[str, List]) -> None:
        """Spawn and advance emergency vehicles."""
        for ev in self._emergency_vehicles:
            # Spawn check
            if not ev.active and not ev.cleared and self._step == ev.spawn_step:
                ev.spawn()
                # Place on the approach road to the first intersection
                target = ev.route[0]
                road = self._find_approach_road(target, ev.direction)
                if road:
                    road.try_insert_at(
                        Vehicle(
                            vehicle_id=-1,  # Placeholder — EV uses its own tracking
                            spawn_step=self._step,
                            direction=ev.direction,
                        ),
                        0,
                    )
                    ev.cell_position = road.length  # Full distance away

            if not ev.active:
                continue

            # Move EV toward its target intersection
            target = ev.target_intersection
            if target is None:
                continue

            intersection = self._intersections.get(target)
            if intersection is None:
                continue

            if ev.cell_position > 0:
                # Still on approach — move 1 cell closer
                ev.cell_position -= 1
                ev.tick_move()
            elif ev.cell_position == 0:
                # At the intersection — check if green
                if intersection.is_green_for(ev.direction):
                    # Pass through!
                    delay_here = ev.delay_per_intersection.get(target, 0.0)
                    events["ev_passes"].append({
                        "ev_id": ev.ev_id,
                        "intersection_id": target,
                        "delay": delay_here,
                    })

                    # Check if there was a correct phase on arrival
                    phase_correct = delay_here == 0.0

                    # Advance to next intersection
                    next_node = ev.advance_to_next_intersection()
                    if next_node is not None:
                        # Update direction based on link
                        link_key = f"{target}_{next_node}"
                        road = self._roads.get(link_key)
                        if road:
                            ev.direction = road.direction
                            ev.cell_position = road.length
                        else:
                            ev.cell_position = 10  # Default

                        events["ev_exits"].append({
                            "ev_id": ev.ev_id,
                            "intersection_id": target,
                            "next_node": next_node,
                        })
                    else:
                        # Route complete — EV cleared
                        events["ev_exits"].append({
                            "ev_id": ev.ev_id,
                            "intersection_id": target,
                            "next_node": None,
                        })
                else:
                    # Red light — wait
                    ev.tick_wait(STEP_DURATION)

    # ─────────────────────────────────────────
    # Incidents
    # ─────────────────────────────────────────

    def _update_incidents(self) -> None:
        """Apply or clear road incidents based on current step."""
        for inc in self._config.incidents:
            road_key = inc.link.replace("_", "_", 1)  # "I1_I3" → road key
            # Find matching road
            matching_key = None
            for key in self._roads:
                if key == road_key or key == inc.link:
                    matching_key = key
                    break

            if matching_key is None:
                continue

            if self._step == inc.step:
                # Apply incident
                self._roads[matching_key].apply_incident(inc.capacity_reduction)
                self._active_incidents[matching_key] = inc
            elif (
                matching_key in self._active_incidents
                and self._step >= inc.step + inc.duration
            ):
                # Clear incident
                self._roads[matching_key].clear_incident()
                del self._active_incidents[matching_key]

    # ─────────────────────────────────────────
    # Metrics
    # ─────────────────────────────────────────

    def _compute_step_metrics(
        self, moved: int, waiting: int, could_move: int
    ) -> None:
        """Compute per-step metrics for reward computation."""
        # Network pressure = sum of |upstream - downstream| density
        pressure = 0.0
        max_pressure = 0.0
        for road in self._all_roads():
            upstream_density = road.vehicle_count / max(road.length, 1)
            # Find reverse road
            reverse_key = f"{road.target_id}_{road.source_id}"
            reverse = self._roads.get(reverse_key)
            downstream_density = 0.0
            if reverse:
                downstream_density = reverse.vehicle_count / max(reverse.length, 1)
            pressure += abs(upstream_density - downstream_density)
            max_pressure += 1.0  # Theoretical max

        # Average wait
        all_vehicles = []
        for road in self._all_roads():
            for cell in road.cells:
                if cell is not None:
                    all_vehicles.append(cell)

        avg_wait = 0.0
        if all_vehicles:
            avg_wait = sum(v.wait_time for v in all_vehicles) / len(all_vehicles)

        self._step_metrics = {
            "vehicles_moved": moved,
            "vehicles_waiting": waiting,
            "vehicles_could_move": max(could_move, 1),
            "network_pressure": pressure,
            "max_possible_pressure": max(max_pressure, 1.0),
            "avg_wait": avg_wait,
        }

    def get_step_metrics(self) -> Dict[str, Any]:
        """Return metrics from the last tick (used by reward computation)."""
        return self._step_metrics

    def get_network_metrics(self) -> Dict[str, Any]:
        """Return cumulative network-wide metrics."""
        total_in_network = sum(r.vehicle_count for r in self._all_roads())
        total_waiting = sum(
            r.queue_length for r in self._all_roads()
        )
        avg_delay = 0.0
        max_delay = 0.0
        if self._all_vehicle_delays:
            avg_delay = sum(self._all_vehicle_delays) / len(self._all_vehicle_delays)
            max_delay = max(self._all_vehicle_delays)

        ev_total = len(self._emergency_vehicles)
        ev_cleared = sum(1 for ev in self._emergency_vehicles if ev.cleared)
        total_ev_delay = sum(ev.total_delay for ev in self._emergency_vehicles)

        return {
            "total_in_network": total_in_network,
            "total_waiting": total_waiting,
            "total_cleared": self._total_cleared,
            "total_generated": self._total_generated,
            "avg_delay": round(avg_delay, 2),
            "max_delay": round(max_delay, 2),
            "network_pressure": self._step_metrics.get("network_pressure", 0.0),
            "ev_total": ev_total,
            "ev_cleared": ev_cleared,
            "total_ev_delay": round(total_ev_delay, 2),
        }

    # ─────────────────────────────────────────
    # Query interfaces (called by environment.py)
    # ─────────────────────────────────────────

    def get_intersection(self, intersection_id: str) -> Optional[Intersection]:
        """Get intersection by ID."""
        return self._intersections.get(intersection_id)

    def get_all_intersections(self) -> List[Dict[str, Any]]:
        """Get serialized state of all intersections."""
        result = []
        for i_id, intersection in self._intersections.items():
            # Build approach data from roads feeding this intersection
            approach_data: Dict[str, dict] = {}
            for direction in ["north", "south", "east", "west"]:
                road = self._find_approach_road(i_id, direction)
                if road:
                    approach_data[direction] = {
                        "queue_length": road.queue_length,
                        "approaching": road.approaching_vehicles,
                        "wait_avg": round(road.get_avg_wait(), 2),
                        "wait_max": round(road.get_max_wait(), 2),
                        "throughput_last": intersection._throughput_last_phase.get(
                            direction, 0
                        ),
                    }
                else:
                    approach_data[direction] = {
                        "queue_length": 0,
                        "approaching": 0,
                        "wait_avg": 0.0,
                        "wait_max": 0.0,
                        "throughput_last": 0,
                    }

            result.append(intersection.to_dict(approach_data))
        return result

    def get_active_emergency_vehicles(self) -> List[Dict[str, Any]]:
        """Get all active (not cleared) emergency vehicles as dicts."""
        return [ev.to_dict() for ev in self._emergency_vehicles if ev.active]

    def get_outgoing_links(self, intersection_id: str) -> List[Dict[str, Any]]:
        """Get all road segments leaving this intersection."""
        result = []
        for road in self._roads.values():
            if road.source_id == intersection_id:
                result.append(road.to_link_dict())
        return result

    def get_incoming_links(self, intersection_id: str) -> List[Dict[str, Any]]:
        """Get all road segments entering this intersection."""
        result = []
        for road in self._roads.values():
            if road.target_id == intersection_id:
                result.append(road.to_link_dict())
        return result

    def _all_roads(self) -> List[Road]:
        """All roads (internal + external)."""
        return list(self._roads.values()) + list(self._external_roads.values())
