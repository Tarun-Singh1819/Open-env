"""
Pydantic models for the Traffic Control OpenEnv environment.

Defines the typed Action, Observation, State, and CoordinationMessage
schemas. Inherits from openenv.core.env_server base types for spec compliance.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from openenv.core.env_server import (
    Action as OpenEnvAction,
    Observation as OpenEnvObservation,
    State as OpenEnvState,
)


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────


class Phase(str, Enum):
    """Traffic signal phases (simplified NEMA dual-ring)."""

    NS_GREEN = "NS_GREEN"
    NS_LEFT = "NS_LEFT"
    EW_GREEN = "EW_GREEN"
    EW_LEFT = "EW_LEFT"
    ALL_RED = "ALL_RED"  # Clearance interval (automatic, not selectable)
    YELLOW = "YELLOW"  # Transition (automatic, not selectable)


class Direction(str, Enum):
    """Cardinal approach directions."""

    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"


class MessageType(str, Enum):
    """Types of coordination messages between agents."""

    TRAFFIC_ALERT = "traffic_alert"
    EMERGENCY_ALERT = "emergency_alert"
    CONGESTION_WARNING = "congestion_warning"
    STATUS_UPDATE = "status_update"
    REQUEST = "request"


class Urgency(str, Enum):
    """Urgency levels for coordination messages."""

    ROUTINE = "routine"
    ELEVATED = "elevated"
    URGENT = "urgent"
    CRITICAL = "critical"


# ─────────────────────────────────────────────
# Coordination Message
# ─────────────────────────────────────────────


class CoordinationMessage(BaseModel):
    """
    Message between traffic control agents.
    Designed to be LLM-native: natural language content field
    instead of requiring precise numerical estimates.
    """

    from_node: str = Field(description="Sending intersection/agent ID")
    to_node: str = Field(description="Target intersection/agent ID")
    message_type: MessageType
    urgency: Urgency
    content: str = Field(
        description=(
            "Natural language description of the situation. "
            "E.g., 'Heavy northbound traffic heading to I3, "
            "~15 vehicles in next 30 seconds'"
        )
    )
    suggested_action: Optional[str] = Field(
        default=None,
        description="Optional suggested response. E.g., 'Prepare NS_GREEN at I3'",
    )


# ─────────────────────────────────────────────
# Observation Models
# ─────────────────────────────────────────────


class ApproachState(BaseModel):
    """State of one approach (direction) at an intersection."""

    direction: Direction
    queue_length: int = Field(ge=0, description="Vehicles stopped/waiting at red")
    approaching_vehicles: int = Field(
        ge=0, description="Vehicles moving toward intersection"
    )
    wait_time_avg: float = Field(
        ge=0.0, description="Avg wait time (seconds) for queued vehicles"
    )
    wait_time_max: float = Field(
        ge=0.0, description="Max wait time of any single queued vehicle"
    )
    throughput_last_phase: int = Field(
        ge=0, description="Vehicles cleared during the last green phase"
    )


class EmergencyVehicleObs(BaseModel):
    """An active emergency vehicle visible in the network."""

    ev_id: str
    current_road: str = Field(description="Road segment, e.g. 'I1_to_I2'")
    distance_to_next_intersection: int = Field(
        ge=0, description="Cells away from next intersection"
    )
    target_intersection: str = Field(description="Next intersection on route")
    direction: Direction = Field(description="Approach direction at target")
    priority: int = Field(
        ge=1, le=3, description="1=ambulance, 2=fire, 3=police"
    )
    waiting: bool = Field(description="True if stopped at a red light")
    total_delay: float = Field(
        ge=0.0, description="Accumulated delay this trip (seconds)"
    )


class IntersectionObservation(BaseModel):
    """Full observable state of one intersection."""

    intersection_id: str
    controlled_by: str = Field(description="'agent' or 'heuristic'")
    current_phase: Phase
    phase_time_elapsed: int = Field(ge=0, description="Steps in current phase")
    min_green_remaining: int = Field(
        ge=0, description="Steps of min green left (0 = can switch)"
    )
    approaches: List[ApproachState] = Field(description="4 directions")
    total_vehicles_waiting: int = 0
    total_throughput: int = Field(
        default=0, description="Cumulative vehicles cleared this episode"
    )


class NetworkMetrics(BaseModel):
    """Network-wide aggregated performance metrics."""

    total_vehicles_in_network: int = 0
    total_vehicles_waiting: int = 0
    total_vehicles_cleared: int = 0
    avg_delay_per_vehicle: float = 0.0
    max_delay_any_vehicle: float = 0.0
    total_emergency_delay: float = 0.0
    network_pressure: float = Field(
        default=0.0,
        description=(
            "Sum of |upstream - downstream| density across all links. "
            "Lower = better flow."
        ),
    )


class TrafficObservation(OpenEnvObservation):
    """
    Complete observation returned by step() and reset().

    Inherits from openenv.core.env_server.Observation which provides:
    - done: bool (whether episode has terminated)
    - reward: float | None (reward signal from the last action)
    - metadata: Dict[str, Any] (additional metadata)
    """

    model_config = ConfigDict(
        extra="allow",  # Override base 'forbid' to allow our custom fields
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    step_number: int = 0
    time_seconds: int = 0
    intersections: List[IntersectionObservation] = Field(default_factory=list)
    emergency_vehicles: List[EmergencyVehicleObs] = Field(default_factory=list)
    incoming_messages: List[CoordinationMessage] = Field(default_factory=list)
    network_metrics: NetworkMetrics = Field(default_factory=NetworkMetrics)
    agent_cluster: List[str] = Field(
        default_factory=list,
        description="IDs of intersections this agent controls",
    )
    message: str = ""
    last_action_error: Optional[str] = Field(
        default=None,
        description="Error message from the last action, or null if none",
    )


# ─────────────────────────────────────────────
# Action Models
# ─────────────────────────────────────────────


class PhaseCommand(BaseModel):
    """Signal phase command for one intersection."""

    intersection_id: str
    target_phase: Phase


class TrafficAction(OpenEnvAction):
    """
    Action submitted by the agent each step.

    Inherits from openenv.core.env_server.Action which provides:
    - metadata: Dict[str, Any]
    """

    model_config = ConfigDict(
        extra="allow",  # Override base 'forbid' to allow our custom fields
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    commands: List[PhaseCommand] = Field(
        default_factory=list,
        description=(
            "Phase commands for agent-controlled intersections. "
            "Intersections not listed keep their current phase."
        ),
    )
    messages: List[CoordinationMessage] = Field(
        default_factory=list,
        description="Coordination messages to send to neighboring agents.",
    )
    reasoning: str = Field(
        default="",
        description="Agent's reasoning (logged, helps with debugging)",
    )


# ─────────────────────────────────────────────
# State Model
# ─────────────────────────────────────────────


class TrafficState(OpenEnvState):
    """
    Episode state returned by the state() endpoint.

    Inherits from openenv.core.env_server.State which provides:
    - episode_id: Optional[str]
    - step_count: int (>= 0)
    """

    task_id: str = ""
    max_steps: int = 0
    time_seconds: int = 0
    is_done: bool = False
    cumulative_reward: float = 0.0
    # Traffic
    vehicles_cleared: int = 0
    vehicles_generated: int = 0
    # Emergency
    emergency_vehicles_cleared: int = 0
    emergency_vehicles_total: int = 0
    total_emergency_delay: float = 0.0
    # Comparison
    avg_delay: float = 0.0
    fixed_timing_avg_delay: float = 0.0
    # Messages
    messages_sent: int = 0
    messages_received: int = 0
    message_quality_score: float = 0.0


# ─────────────────────────────────────────────
# Coordination Ledger (used by reward + grader)
# ─────────────────────────────────────────────


class CoordinationEvent(BaseModel):
    """One link in an EV's coordination chain."""

    ev_id: str
    step: int
    event_type: str  # "REACTION" | "FORWARD" | "PASS"
    intersection_id: str
    success: bool
    detail: str  # "PROACTIVE" | "REACTIVE" | "MISSED" | "FORWARDED" | "BROKEN"
    delay_at_intersection: float = 0.0


class CoordinationLedger:
    """
    Tracks the full coordination chain for every EV across the episode.
    Used by the reward function (per-step) and grader (end-of-episode).
    """

    def __init__(self) -> None:
        self.events: List[CoordinationEvent] = []

    def reset(self) -> None:
        self.events.clear()

    # ── Logging helpers ──

    def log_reaction(
        self, ev_id: str, intersection_id: str, reaction_type: str, step: int
    ) -> None:
        self.events.append(
            CoordinationEvent(
                ev_id=ev_id,
                step=step,
                event_type="REACTION",
                intersection_id=intersection_id,
                success=True,
                detail=reaction_type,
            )
        )

    def log_forward(
        self, ev_id: str, target_node: str, was_sent: bool, step: int
    ) -> None:
        self.events.append(
            CoordinationEvent(
                ev_id=ev_id,
                step=step,
                event_type="FORWARD",
                intersection_id=target_node,
                success=was_sent,
                detail="FORWARDED" if was_sent else "BROKEN",
            )
        )

    def log_pass(
        self, ev_id: str, intersection_id: str, delay: float, step: int
    ) -> None:
        self.events.append(
            CoordinationEvent(
                ev_id=ev_id,
                step=step,
                event_type="PASS",
                intersection_id=intersection_id,
                success=(delay == 0),
                detail=f"delay={delay}s",
                delay_at_intersection=delay,
            )
        )

    # ── Query helpers ──

    def get_chain_for_ev(self, ev_id: str) -> List[CoordinationEvent]:
        return [e for e in self.events if e.ev_id == ev_id]

    def get_reaction(
        self, ev_id: str, intersection_id: str
    ) -> Optional[CoordinationEvent]:
        for e in self.events:
            if (
                e.ev_id == ev_id
                and e.intersection_id == intersection_id
                and e.event_type == "REACTION"
            ):
                return e
        return None

    def get_forward(
        self, ev_id: str, target_node: str
    ) -> Optional[CoordinationEvent]:
        for e in self.events:
            if (
                e.ev_id == ev_id
                and e.intersection_id == target_node
                and e.event_type == "FORWARD"
            ):
                return e
        return None
