"""
traffic_env — Autonomous Traffic Signal Control OpenEnv Environment.
"""

from .models import (
    CoordinationLedger,
    CoordinationMessage,
    Direction,
    EmergencyVehicleObs,
    MessageType,
    Phase,
    PhaseCommand,
    TrafficAction,
    TrafficObservation,
    TrafficState,
    Urgency,
)

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from typing import Any, Dict


class TrafficEnvClient(EnvClient[TrafficAction, TrafficObservation, TrafficState]):
    """OpenEnv client for connecting to the Traffic Control environment."""

    def _step_payload(self, action: TrafficAction) -> Dict[str, Any]:
        return {"action": action.model_dump()}

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[TrafficObservation]:
        obs_data = payload.get("observation", payload)
        obs = TrafficObservation(**obs_data)
        return StepResult(
            observation=obs,
            reward=payload.get("reward", obs.reward),
            done=payload.get("done", obs.done),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> TrafficState:
        return TrafficState(**payload)


__all__ = [
    "CoordinationLedger",
    "CoordinationMessage",
    "Direction",
    "EmergencyVehicleObs",
    "MessageType",
    "Phase",
    "PhaseCommand",
    "TrafficAction",
    "TrafficEnvClient",
    "TrafficObservation",
    "TrafficState",
    "Urgency",
]
