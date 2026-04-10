"""
TrafficEnvironment — Core OpenEnv environment for Autonomous Traffic Control.

Implements the OpenEnv interface: reset(), step(), state().
Orchestrates the simulation engine, heuristic controllers, message generation,
coordination ledger, reward computation, and episode lifecycle.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Tuple

from openenv.core.env_server import Environment


from ..models import (
    ApproachState,
    CoordinationEvent,
    CoordinationLedger,
    CoordinationMessage,
    Direction,
    EmergencyVehicleObs,
    IntersectionObservation,
    MessageType,
    NetworkMetrics,
    Phase,
    PhaseCommand,
    TrafficAction,
    TrafficObservation,
    TrafficState,
    Urgency,
)
from .simulation import TrafficSimulation
from .task import TASK_CONFIGS, TaskConfig


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

STEP_DURATION_SECONDS = 5       # Each step = 5 simulated seconds
MIN_GREEN_STEPS = 3             # 15 seconds minimum green
YELLOW_STEPS = 1                # 5 seconds yellow
ALL_RED_STEPS = 1               # 5 seconds all-red clearance
TRAFFIC_ALERT_THRESHOLD = 5     # Vehicles in transit to trigger alert
MAX_TOLERABLE_WAIT = 120.0      # Seconds — normalizes wait penalty


# ─────────────────────────────────────────────
# Phase helpers
# ─────────────────────────────────────────────

# Which directions get green for each phase
PHASE_GREEN_DIRECTIONS: Dict[Phase, List[Direction]] = {
    Phase.NS_GREEN: [Direction.NORTH, Direction.SOUTH],
    Phase.NS_LEFT:  [Direction.NORTH, Direction.SOUTH],
    Phase.EW_GREEN: [Direction.EAST, Direction.WEST],
    Phase.EW_LEFT:  [Direction.EAST, Direction.WEST],
    Phase.ALL_RED:  [],
    Phase.YELLOW:   [],
}

# Map direction to the phase that serves it
DIRECTION_TO_PHASE: Dict[Direction, Phase] = {
    Direction.NORTH: Phase.NS_GREEN,
    Direction.SOUTH: Phase.NS_GREEN,
    Direction.EAST:  Phase.EW_GREEN,
    Direction.WEST:  Phase.EW_GREEN,
}


class TrafficEnvironment(Environment[TrafficAction, TrafficObservation, TrafficState]):
    """
    OpenEnv-compliant traffic control environment.

    The LLM agent controls a subset of intersections (agent_cluster).
    Remaining intersections run on heuristic fixed-timing controllers.
    The agent receives coordination messages from heuristic neighbors
    and can send outgoing messages affecting heuristic behavior.
    """

    def __init__(self) -> None:
        super().__init__()
        # Episode state
        self._episode_id: str = ""
        self._task_id: str = ""
        self._task_config: Optional[TaskConfig] = None
        self._step_count: int = 0
        self._time_seconds: int = 0
        self._is_done: bool = True
        self._cumulative_reward: float = 0.0

        # Core simulation
        self._sim: Optional[TrafficSimulation] = None

        # Coordination
        self._ledger = CoordinationLedger()
        self._messages_sent: int = 0
        self._messages_received: int = 0

        # Per-step transient state
        self._current_incoming_messages: List[CoordinationMessage] = []
        self._current_action: Optional[TrafficAction] = None
        self._ev_exit_events: List[Dict[str, Any]] = []
        self._ev_pass_events: List[Dict[str, Any]] = []
        self._num_phase_changes: int = 0

    # ─────────────────────────────────────────
    # OpenEnv API: reset()
    # ─────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: str = "single_intersection",
        **kwargs: Any,
    ) -> TrafficObservation:
        """
        Initialize a new episode for the given task.
        Returns the initial observation.
        """
        if task_id not in TASK_CONFIGS:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Available: {list(TASK_CONFIGS.keys())}"
            )

        self._episode_id = episode_id or str(uuid.uuid4())
        self._task_id = task_id
        self._task_config = TASK_CONFIGS[task_id]
        self._step_count = 0
        self._time_seconds = 0
        self._is_done = False
        self._cumulative_reward = 0.0
        self._messages_sent = 0
        self._messages_received = 0

        # Reset coordination ledger
        self._ledger.reset()
        self._reset_rubric()

        # Reset transient state
        self._current_incoming_messages = []
        self._current_action = None
        self._ev_exit_events = []
        self._ev_pass_events = []
        self._num_phase_changes = 0

        # Build simulation from task config
        self._sim = TrafficSimulation(self._task_config)

        return self._build_observation(initial=True)

    # ─────────────────────────────────────────
    # OpenEnv API: step()
    # ─────────────────────────────────────────

    def step(
        self,
        action: TrafficAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> TrafficObservation:
        """
        Execute one time step.
        Returns TrafficObservation with reward and done embedded.
        """
        if self._is_done:
            raise RuntimeError("Episode is done. Call reset() first.")
        if self._sim is None:
            raise RuntimeError("No simulation. Call reset() first.")

        self._current_action = action
        self._ev_exit_events = []
        self._ev_pass_events = []
        self._num_phase_changes = 0

        # 1. Apply agent's phase commands to agent-controlled intersections
        self._apply_phase_commands(action.commands)

        # 2. Process agent's outgoing messages (affect heuristic controllers)
        self._process_outgoing_messages(action.messages)
        self._messages_sent += len(action.messages)

        # 3. Advance simulation by one step (vehicle movement, signal transitions)
        sim_events = self._sim.tick()
        self._ev_exit_events = sim_events.get("ev_exits", [])
        self._ev_pass_events = sim_events.get("ev_passes", [])

        # 4. Generate incoming messages from heuristic neighbors
        self._current_incoming_messages = self._generate_incoming_messages()
        self._messages_received += len(self._current_incoming_messages)

        # 5. Update step counters
        self._step_count += 1
        self._time_seconds += STEP_DURATION_SECONDS

        # 6. Compute reward
        reward = self._compute_reward()
        self._cumulative_reward += reward

        # 7. Check termination
        if self._step_count >= self._task_config.max_steps:
            self._is_done = True

        # 8. Build observation (now includes reward + done)
        observation = self._build_observation()
        observation.reward = round(reward, 4)
        observation.done = self._is_done

        return observation

    # ─────────────────────────────────────────
    # OpenEnv API: state()
    # ─────────────────────────────────────────

    @property
    def state(self) -> TrafficState:
        """Return current episode state."""
        metrics = self._sim.get_network_metrics() if self._sim else {}
        return TrafficState(
            episode_id=self._episode_id,
            task_id=self._task_id,
            step_count=self._step_count,
            max_steps=self._task_config.max_steps if self._task_config else 0,
            time_seconds=self._time_seconds,
            is_done=self._is_done,
            cumulative_reward=round(self._cumulative_reward, 4),
            vehicles_cleared=metrics.get("total_cleared", 0),
            vehicles_generated=metrics.get("total_generated", 0),
            emergency_vehicles_cleared=metrics.get("ev_cleared", 0),
            emergency_vehicles_total=metrics.get("ev_total", 0),
            total_emergency_delay=metrics.get("total_ev_delay", 0.0),
            avg_delay=metrics.get("avg_delay", 0.0),
            fixed_timing_avg_delay=self._task_config.baseline_avg_delay
            if self._task_config
            else 0.0,
            messages_sent=self._messages_sent,
            messages_received=self._messages_received,
            message_quality_score=0.0,  # Computed by grader at end
        )

    # ─────────────────────────────────────────
    # Phase Command Application
    # ─────────────────────────────────────────

    def _apply_phase_commands(self, commands: List[PhaseCommand]) -> None:
        """
        Apply phase commands to agent-controlled intersections.
        Commands targeting heuristic intersections are silently ignored.
        Commands during min-green are silently ignored.
        """
        for cmd in commands:
            intersection = self._sim.get_intersection(cmd.intersection_id)
            if intersection is None:
                continue
            if intersection.intersection_id not in self._task_config.agent_cluster:
                continue  # Agent can't control heuristic intersections

            # Skip non-selectable phases
            if cmd.target_phase in (Phase.ALL_RED, Phase.YELLOW):
                continue

            # Attempt phase change (intersection enforces min-green internally)
            changed = intersection.request_phase_change(cmd.target_phase)
            if changed:
                self._num_phase_changes += 1

    # ─────────────────────────────────────────
    # Outgoing Message Processing
    # ─────────────────────────────────────────

    def _process_outgoing_messages(
        self, messages: List[CoordinationMessage]
    ) -> None:
        """
        Heuristic controllers react to agent's messages.
        This teaches the LLM that sending good messages has real consequences.
        """
        for msg in messages:
            target = self._sim.get_intersection(msg.to_node)
            if target is None:
                continue
            if target.intersection_id in self._task_config.agent_cluster:
                continue  # Self-targeted, skip

            # Emergency alert → heuristic pre-empts
            if (
                msg.message_type == MessageType.EMERGENCY_ALERT
                and msg.urgency == Urgency.CRITICAL
            ):
                target.queue_emergency_preemption(msg)

            # Traffic alert → heuristic extends/prepares green
            elif msg.message_type == MessageType.TRAFFIC_ALERT and msg.urgency in (
                Urgency.ELEVATED,
                Urgency.URGENT,
            ):
                target.prepare_for_incoming_traffic(msg)

            # Congestion warning → heuristic reduces outflow
            elif msg.message_type == MessageType.CONGESTION_WARNING:
                target.reduce_outflow(msg)

    # ─────────────────────────────────────────
    # Incoming Message Generation
    # ─────────────────────────────────────────

    def _generate_incoming_messages(self) -> List[CoordinationMessage]:
        """
        Generate realistic coordination messages from heuristic-controlled
        intersections based on actual simulation state.
        """
        messages: List[CoordinationMessage] = []

        for h_id in self._task_config.heuristic_cluster:
            h_intersection = self._sim.get_intersection(h_id)
            if h_intersection is None:
                continue

            # ── Traffic alerts: heavy traffic released toward agent ──
            for link in self._sim.get_outgoing_links(h_id):
                if link["target_id"] not in self._task_config.agent_cluster:
                    continue
                vehicles = link["vehicles_in_transit"]
                if vehicles >= TRAFFIC_ALERT_THRESHOLD:
                    urgency = (
                        Urgency.URGENT if vehicles > 15 else Urgency.ELEVATED
                    )
                    eta_low = link["eta_steps"] * STEP_DURATION_SECONDS
                    eta_high = (link["eta_steps"] + 2) * STEP_DURATION_SECONDS
                    messages.append(
                        CoordinationMessage(
                            from_node=h_id,
                            to_node=link["target_id"],
                            message_type=MessageType.TRAFFIC_ALERT,
                            urgency=urgency,
                            content=(
                                f"Released {vehicles} vehicles heading "
                                f"{link['direction']}bound toward "
                                f"{link['target_id']}. "
                                f"Platoon ETA approximately {eta_low}-{eta_high} "
                                f"seconds. Current {h_id} phase is "
                                f"{h_intersection.current_phase}."
                            ),
                            suggested_action=(
                                f"Consider preparing "
                                f"{DIRECTION_TO_PHASE.get(Direction(link['direction']), Phase.NS_GREEN).value} "
                                f"at {link['target_id']} to receive incoming platoon"
                            ),
                        )
                    )

            # ── Emergency alerts: EV approaching agent's cluster ──
            for ev in self._sim.get_active_emergency_vehicles():
                if ev["target_intersection"] not in self._task_config.agent_cluster:
                    continue
                if not ev["current_road"].startswith(h_id):
                    continue
                ev_dir = Direction(ev["direction"])
                receiving_phase = DIRECTION_TO_PHASE.get(ev_dir, Phase.NS_GREEN)
                messages.append(
                    CoordinationMessage(
                        from_node=h_id,
                        to_node=ev["target_intersection"],
                        message_type=MessageType.EMERGENCY_ALERT,
                        urgency=Urgency.CRITICAL,
                        content=(
                            f"Emergency vehicle {ev['ev_id']} "
                            f"(priority {ev['priority']}) heading "
                            f"{ev['direction']}bound toward "
                            f"{ev['target_intersection']}. Currently "
                            f"{ev['distance']} cells away "
                            f"(~{ev['distance'] * STEP_DURATION_SECONDS} seconds). "
                            f"Total delay so far: {ev['total_delay']:.0f}s."
                        ),
                        suggested_action=(
                            f"Immediate {receiving_phase.value} at "
                            f"{ev['target_intersection']} to provide clear path"
                        ),
                    )
                )

            # ── Congestion warnings: spillback toward agent's nodes ──
            for link in self._sim.get_incoming_links(h_id):
                if link["source_id"] not in self._task_config.agent_cluster:
                    continue
                if link["occupancy_pct"] >= 80:
                    messages.append(
                        CoordinationMessage(
                            from_node=h_id,
                            to_node=link["source_id"],
                            message_type=MessageType.CONGESTION_WARNING,
                            urgency=Urgency.URGENT,
                            content=(
                                f"Link from {link['source_id']} to {h_id} is "
                                f"near capacity ({link['occupancy_pct']:.0f}% full). "
                                f"Risk of spillback. Please reduce "
                                f"{link['direction']}bound flow from "
                                f"{link['source_id']}."
                            ),
                            suggested_action=(
                                f"Reduce green time for "
                                f"{link['direction']}bound at {link['source_id']}"
                            ),
                        )
                    )

        return messages

    # ─────────────────────────────────────────
    # Reward Computation
    # ─────────────────────────────────────────

    def _compute_reward(self) -> float:
        """
        Per-step reward in approximately [-1.0, +1.0].
        Two layers: steady traffic signal + event-triggered EV coordination.
        """
        # ═══ LAYER 1: STEADY TRAFFIC PERFORMANCE ═══

        metrics = self._sim.get_step_metrics()

        # 1. Throughput
        moved = metrics.get("vehicles_moved", 0)
        could_move = max(metrics.get("vehicles_could_move", 1), 1)
        throughput_reward = moved / could_move  # [0, 1]

        # 2. Pressure
        pressure = metrics.get("network_pressure", 0.0)
        max_pressure = max(metrics.get("max_possible_pressure", 1.0), 1.0)
        pressure_reward = -pressure / max_pressure  # [-1, 0]

        # 3. Wait time
        avg_wait = metrics.get("avg_wait", 0.0)
        wait_penalty = -min(avg_wait / MAX_TOLERABLE_WAIT, 1.0)  # [-1, 0]

        # 4. Emergency delay (EV stuck at red)
        emergency_penalty = 0.0
        active_evs = self._sim.get_active_emergency_vehicles()
        for ev in active_evs:
            if ev.get("waiting", False):
                step_wait = ev.get("current_step_wait", 0.0)
                emergency_penalty -= min(step_wait / 30.0, 1.0)
        if active_evs:
            emergency_penalty /= len(active_evs)
        emergency_penalty = max(emergency_penalty, -1.0)  # [-1, 0]

        # 5. Switch cost
        switch_penalty = -0.1 * self._num_phase_changes  # [-0.4, 0]

        # ═══ LAYER 2: EV COORDINATION CHAIN (event-triggered) ═══
        ev_coord_reward = self._compute_ev_coordination_reward()

        # ═══ LAYER 3: GENERAL MESSAGE QUALITY (soft) ═══
        general_msg_reward = self._compute_general_message_reward()

        # ═══ WEIGHTED COMBINATION ═══
        reward = (
            0.25 * throughput_reward
            + 0.20 * pressure_reward
            + 0.15 * wait_penalty
            + 0.15 * emergency_penalty
            + 0.15 * ev_coord_reward
            + 0.05 * general_msg_reward
            + 0.05 * switch_penalty
        )

        return max(min(reward, 1.0), -1.0)

    # ─────────────────────────────────────────
    # EV Coordination Reward (Event-Triggered)
    # ─────────────────────────────────────────

    def _compute_ev_coordination_reward(self) -> float:
        """
        Only fires during steps with active EV events.
        Returns 0.0 on steps with no EV events (~90% of steps).

        Three events:
          A) Received emergency_alert → did agent react?
          B) EV exited toward heuristic → did agent forward?
          C) EV passed through agent intersection → zero delay?
        """
        reward = 0.0

        # ── EVENT A: Received emergency_alert ──
        for msg in self._current_incoming_messages:
            if msg.message_type != MessageType.EMERGENCY_ALERT:
                continue
            target_id = msg.to_node
            if target_id not in self._task_config.agent_cluster:
                continue

            # What phase does the EV need?
            correct_phase = self._get_receiving_phase_from_message(msg)
            intersection = self._sim.get_intersection(target_id)
            if intersection is None:
                continue

            current_phase = intersection.current_phase
            ev_id = self._extract_ev_id_from_message(msg)

            if current_phase == correct_phase:
                reward += 0.15  # Proactive — already correct phase
                self._ledger.log_reaction(ev_id, target_id, "PROACTIVE", self._step_count)
            elif self._is_switching_to(target_id, correct_phase):
                reward += 0.10  # Reactive — switching now
                self._ledger.log_reaction(ev_id, target_id, "REACTIVE", self._step_count)
            # No penalty — emergency_penalty in Layer 1 handles failure

        # ── EVENT B: EV exited agent zone toward heuristic ──
        for ev_event in self._ev_exit_events:
            next_node = ev_event.get("next_node")
            if not next_node or next_node not in self._task_config.heuristic_cluster:
                continue

            ev_id = ev_event["ev_id"]
            sent_alert = any(
                m.to_node == next_node
                and m.message_type == MessageType.EMERGENCY_ALERT
                and m.urgency == Urgency.CRITICAL
                for m in (self._current_action.messages if self._current_action else [])
            )

            if sent_alert:
                reward += 0.15  # Chain maintained
                self._ledger.log_forward(ev_id, next_node, True, self._step_count)
            else:
                reward -= 0.10  # Chain broken
                self._ledger.log_forward(ev_id, next_node, False, self._step_count)

        # ── EVENT C: EV passed through agent intersection with zero delay ──
        for ev_event in self._ev_pass_events:
            i_id = ev_event.get("intersection_id")
            if i_id not in self._task_config.agent_cluster:
                continue
            delay = ev_event.get("delay", 0.0)
            if delay == 0:
                reward += 0.10  # Clean pass
            self._ledger.log_pass(
                ev_event["ev_id"], i_id, delay, self._step_count
            )

        return max(min(reward, 0.3), -0.2)

    # ─────────────────────────────────────────
    # General Message Reward (Soft, Non-EV)
    # ─────────────────────────────────────────

    def _compute_general_message_reward(self) -> float:
        """
        Soft scoring for non-EV messages (traffic alerts, congestion warnings).
        Returns 0.0 on steps with no relevant events.
        """
        if self._current_action is None:
            return 0.0

        obligations = self._get_non_ev_message_obligations()
        sent = [
            m
            for m in self._current_action.messages
            if m.message_type != MessageType.EMERGENCY_ALERT
        ]

        if not obligations and not sent:
            return 0.0

        hits = 0
        for ob in obligations:
            if any(self._msg_matches_obligation(m, ob) for m in sent):
                hits += 1
        spam = max(len(sent) - hits, 0)

        return max(min(hits * 0.05 - spam * 0.02, 0.1), -0.1)

    # ─────────────────────────────────────────
    # Observation Builder
    # ─────────────────────────────────────────

    def _build_observation(self, initial: bool = False) -> TrafficObservation:
        """Build the complete TrafficObservation from current sim state."""
        intersections: List[IntersectionObservation] = []

        for i_data in self._sim.get_all_intersections():
            approaches = []
            for d in [Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST]:
                a = i_data["approaches"].get(d.value, {})
                approaches.append(
                    ApproachState(
                        direction=d,
                        queue_length=a.get("queue_length", 0),
                        approaching_vehicles=a.get("approaching", 0),
                        wait_time_avg=a.get("wait_avg", 0.0),
                        wait_time_max=a.get("wait_max", 0.0),
                        throughput_last_phase=a.get("throughput_last", 0),
                    )
                )

            controlled_by = (
                "agent"
                if i_data["id"] in self._task_config.agent_cluster
                else "heuristic"
            )

            intersections.append(
                IntersectionObservation(
                    intersection_id=i_data["id"],
                    controlled_by=controlled_by,
                    current_phase=Phase(i_data["current_phase"]),
                    phase_time_elapsed=i_data.get("phase_elapsed", 0),
                    min_green_remaining=i_data.get("min_green_remaining", 0),
                    approaches=approaches,
                    total_vehicles_waiting=i_data.get("total_waiting", 0),
                    total_throughput=i_data.get("total_throughput", 0),
                )
            )

        ev_obs: List[EmergencyVehicleObs] = []
        for ev in self._sim.get_active_emergency_vehicles():
            ev_obs.append(
                EmergencyVehicleObs(
                    ev_id=ev["ev_id"],
                    current_road=ev["current_road"],
                    distance_to_next_intersection=ev["distance"],
                    target_intersection=ev["target_intersection"],
                    direction=Direction(ev["direction"]),
                    priority=ev["priority"],
                    waiting=ev.get("waiting", False),
                    total_delay=ev.get("total_delay", 0.0),
                )
            )

        net = self._sim.get_network_metrics()

        return TrafficObservation(
            step_number=self._step_count,
            time_seconds=self._time_seconds,
            intersections=intersections,
            emergency_vehicles=ev_obs,
            incoming_messages=self._current_incoming_messages if not initial else [],
            network_metrics=NetworkMetrics(
                total_vehicles_in_network=net.get("total_in_network", 0),
                total_vehicles_waiting=net.get("total_waiting", 0),
                total_vehicles_cleared=net.get("total_cleared", 0),
                avg_delay_per_vehicle=net.get("avg_delay", 0.0),
                max_delay_any_vehicle=net.get("max_delay", 0.0),
                total_emergency_delay=net.get("total_ev_delay", 0.0),
                network_pressure=net.get("network_pressure", 0.0),
            ),
            agent_cluster=list(self._task_config.agent_cluster),
            message=(
                f"Episode started: task={self._task_id}"
                if initial
                else f"Step {self._step_count}/{self._task_config.max_steps}"
            ),
        )

    # ─────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────

    def _get_receiving_phase_from_message(self, msg: CoordinationMessage) -> Phase:
        """Determine which phase an incoming EV needs at the target intersection."""
        content_lower = msg.content.lower()
        if "northbound" in content_lower or "southbound" in content_lower:
            return Phase.NS_GREEN
        if "eastbound" in content_lower or "westbound" in content_lower:
            return Phase.EW_GREEN
        # Fallback: parse suggested_action
        if msg.suggested_action:
            sa = msg.suggested_action.upper()
            if "NS_GREEN" in sa:
                return Phase.NS_GREEN
            if "EW_GREEN" in sa:
                return Phase.EW_GREEN
        return Phase.NS_GREEN  # Default

    def _extract_ev_id_from_message(self, msg: CoordinationMessage) -> str:
        """Extract EV ID from message content."""
        content = msg.content
        for token in content.split():
            if token.startswith(("AMB-", "FIRE-", "POL-", "EV-", "EVA-")):
                return token.rstrip(".,;:")
        # Fallback: generate from message fields
        return f"EV_from_{msg.from_node}"

    def _is_switching_to(self, intersection_id: str, target_phase: Phase) -> bool:
        """Check if the current action commands this intersection to switch."""
        if self._current_action is None:
            return False
        return any(
            cmd.intersection_id == intersection_id
            and cmd.target_phase == target_phase
            for cmd in self._current_action.commands
        )

    def _get_non_ev_message_obligations(self) -> List[Dict[str, Any]]:
        """
        Determine what non-EV messages the agent SHOULD be sending this step.
        Based on traffic flowing from agent zone to heuristic zone.
        """
        obligations: List[Dict[str, Any]] = []
        for a_id in self._task_config.agent_cluster:
            for link in self._sim.get_outgoing_links(a_id):
                if link["target_id"] not in self._task_config.heuristic_cluster:
                    continue
                if link["vehicles_in_transit"] >= TRAFFIC_ALERT_THRESHOLD:
                    obligations.append(
                        {
                            "to_node": link["target_id"],
                            "expected_type": MessageType.TRAFFIC_ALERT,
                            "expected_urgency": (
                                Urgency.URGENT
                                if link["vehicles_in_transit"] > 15
                                else Urgency.ELEVATED
                            ),
                        }
                    )
        return obligations

    def _msg_matches_obligation(
        self, msg: CoordinationMessage, obligation: Dict[str, Any]
    ) -> bool:
        """Check if a sent message matches an obligation."""
        return (
            msg.to_node == obligation["to_node"]
            and msg.message_type == obligation["expected_type"]
        )

    # ─────────────────────────────────────────
    # Public accessors for grader
    # ─────────────────────────────────────────

    @property
    def coordination_ledger(self) -> CoordinationLedger:
        return self._ledger

    @property
    def task_config(self) -> Optional[TaskConfig]:
        return self._task_config

    @property
    def simulation(self) -> Optional[TrafficSimulation]:
        return self._sim

    @property
    def episode_id(self) -> str:
        return self._episode_id
