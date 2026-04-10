"""
Message Generator — Utility for generating realistic coordination messages.

The ACTUAL message generation logic lives in TrafficEnvironment._generate_incoming_messages().
This module provides standalone helpers for testing and for the baseline inference script.
"""

from __future__ import annotations

from typing import List

from ..models import CoordinationMessage, MessageType, Urgency


def create_traffic_alert(
    from_node: str,
    to_node: str,
    vehicle_count: int,
    direction: str,
    eta_seconds: int,
) -> CoordinationMessage:
    """Create a traffic alert message."""
    urgency = Urgency.URGENT if vehicle_count > 15 else Urgency.ELEVATED
    return CoordinationMessage(
        from_node=from_node,
        to_node=to_node,
        message_type=MessageType.TRAFFIC_ALERT,
        urgency=urgency,
        content=(
            f"Released {vehicle_count} vehicles heading {direction}bound "
            f"toward {to_node}. ETA approximately {eta_seconds} seconds."
        ),
        suggested_action=(
            f"Prepare green for {direction}bound traffic at {to_node}"
        ),
    )


def create_emergency_alert(
    from_node: str,
    to_node: str,
    ev_id: str,
    priority: int,
    direction: str,
    distance_cells: int,
    total_delay: float = 0.0,
) -> CoordinationMessage:
    """Create an emergency vehicle alert message."""
    phase = "NS_GREEN" if direction in ("north", "south") else "EW_GREEN"
    return CoordinationMessage(
        from_node=from_node,
        to_node=to_node,
        message_type=MessageType.EMERGENCY_ALERT,
        urgency=Urgency.CRITICAL,
        content=(
            f"Emergency vehicle {ev_id} (priority {priority}) heading "
            f"{direction}bound toward {to_node}. "
            f"Currently {distance_cells} cells away "
            f"(~{distance_cells * 5} seconds). "
            f"Total delay so far: {total_delay:.0f}s."
        ),
        suggested_action=f"Immediate {phase} at {to_node} to provide clear path",
    )


def create_congestion_warning(
    from_node: str,
    to_node: str,
    direction: str,
    occupancy_pct: float,
) -> CoordinationMessage:
    """Create a congestion warning message."""
    return CoordinationMessage(
        from_node=from_node,
        to_node=to_node,
        message_type=MessageType.CONGESTION_WARNING,
        urgency=Urgency.URGENT,
        content=(
            f"Link from {to_node} to {from_node} is near capacity "
            f"({occupancy_pct:.0f}% full). Risk of spillback. "
            f"Please reduce {direction}bound flow from {to_node}."
        ),
        suggested_action=(
            f"Reduce green time for {direction}bound at {to_node}"
        ),
    )


def create_status_update(
    from_node: str,
    to_node: str,
    status: str,
) -> CoordinationMessage:
    """Create a status update message."""
    return CoordinationMessage(
        from_node=from_node,
        to_node=to_node,
        message_type=MessageType.STATUS_UPDATE,
        urgency=Urgency.ROUTINE,
        content=status,
        suggested_action=None,
    )
