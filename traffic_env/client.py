"""
EnvClient — HTTP client for interacting with the traffic control environment.

Wraps the FastAPI endpoints into a clean Python interface.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from .models import TrafficAction, TrafficObservation, TrafficState


class TrafficEnvClient:
    """
    HTTP client for the traffic control OpenEnv environment.

    Usage:
        client = TrafficEnvClient("http://localhost:7860")
        obs = client.reset("single_intersection")
        while not done:
            result = client.step(action)
            obs, reward, done = result["observation"], result["reward"], result["done"]
        score = client.grade("single_intersection")
    """

    def __init__(self, base_url: str = "http://localhost:7860", timeout: float = 30.0):
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def health(self) -> Dict[str, Any]:
        """Check server health."""
        resp = self._client.get(f"{self._base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def tasks(self) -> List[Dict[str, Any]]:
        """List available tasks."""
        resp = self._client.get(f"{self._base_url}/tasks")
        resp.raise_for_status()
        return resp.json()

    def reset(self, task_id: str = "single_intersection") -> TrafficObservation:
        """Reset environment with a new task. Returns initial observation."""
        resp = self._client.post(
            f"{self._base_url}/reset",
            json={"task_id": task_id},
        )
        resp.raise_for_status()
        return TrafficObservation(**resp.json())

    def step(self, action: TrafficAction) -> Dict[str, Any]:
        """
        Execute one step. Returns dict with:
          observation: TrafficObservation
          reward: float
          done: bool
          info: dict
        """
        resp = self._client.post(
            f"{self._base_url}/step",
            json={"action": action.model_dump()},
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "observation": TrafficObservation(**data["observation"]),
            "reward": data["reward"],
            "done": data["done"],
            "info": data.get("info", {}),
        }

    def state(self) -> TrafficState:
        """Get current episode state."""
        resp = self._client.get(f"{self._base_url}/state")
        resp.raise_for_status()
        return TrafficState(**resp.json())

    def grade(self, task_id: str) -> Dict[str, Any]:
        """Grade the completed episode. Returns score and details."""
        resp = self._client.post(
            f"{self._base_url}/grader",
            json={"task_id": task_id},
        )
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        """Close HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
