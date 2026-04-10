"""
Grader — Deterministic episode grading with EV Coordination Chain scoring.

Compares agent performance against a fixed-timing baseline.
The EV Coordination Chain is the single largest grading component
in Tasks 2 and 3, testing whether the agent learned the
receive → react → forward protocol.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..models import CoordinationLedger, MessageType, Urgency


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class TrafficGrader:
    """
    Deterministic grader that scores an episode on [0.0, 1.0].

    Must be called AFTER the episode is complete. Uses:
    - Episode stats from the environment
    - The CoordinationLedger for EV chain scoring
    - Pre-computed fixed-timing baseline for comparison
    """

    def __init__(
        self,
        task_id: str,
        episode_stats: Dict[str, Any],
        coordination_ledger: CoordinationLedger,
        baseline_avg_delay: float,
        agent_cluster: List[str],
        heuristic_cluster: List[str],
    ) -> None:
        self._task_id = task_id
        self._stats = episode_stats
        self._ledger = coordination_ledger
        self._baseline_delay = max(baseline_avg_delay, 0.1)
        self._agent_cluster = agent_cluster
        self._heuristic_cluster = heuristic_cluster

    def grade(self) -> float:
        """Return final grade in (0, 1) — strictly between 0 and 1."""
        if self._task_id == "single_intersection":
            raw = self._grade_task1()
        elif self._task_id == "corridor_green_wave":
            raw = self._grade_task2()
        elif self._task_id == "grid_incident":
            raw = self._grade_task3()
        else:
            raw = 0.01
        # Strictly between 0 and 1 — never exactly 0.0 or 1.0
        return clamp(raw, 0.01, 0.99)

    # ─────────────────────────────────────────
    # Task 1: Pure delay comparison
    # ─────────────────────────────────────────

    def _grade_task1(self) -> float:
        """
        100% delay vs baseline.
        Matching baseline = 0.5, 50% better = 1.0, 50% worse = 0.0.
        """
        agent_delay = self._stats.get("avg_delay", 0.0)
        delay_ratio = agent_delay / self._baseline_delay
        score = clamp(1.0 - (delay_ratio - 0.5), 0.0, 1.0)
        return round(score, 4)

    # ─────────────────────────────────────────
    # Task 2: Traffic + EV Chain + Messages + Equity
    # ─────────────────────────────────────────

    def _grade_task2(self) -> float:
        """
        40% throughput | 35% EV chain | 15% general messages | 10% equity
        """
        throughput_score = self._compute_delay_score()
        ev_chain_score = self._grade_ev_coordination_chain()
        general_msg_score = self._grade_general_messages()
        equity_score = self._compute_equity_score()

        return round(
            0.40 * throughput_score
            + 0.35 * ev_chain_score
            + 0.15 * general_msg_score
            + 0.10 * equity_score,
            4,
        )

    # ─────────────────────────────────────────
    # Task 3: Traffic + EV Chain + Messages + Equity + Gridlock
    # ─────────────────────────────────────────

    def _grade_task3(self) -> float:
        """
        25% delay | 30% EV chain | 15% general messages | 15% equity | 15% gridlock
        """
        delay_score = self._compute_delay_score()
        ev_chain_score = self._grade_ev_coordination_chain()
        general_msg_score = self._grade_general_messages()
        equity_score = self._compute_equity_score()
        gridlock_score = self._compute_gridlock_score()

        return round(
            0.25 * delay_score
            + 0.30 * ev_chain_score
            + 0.15 * general_msg_score
            + 0.15 * equity_score
            + 0.15 * gridlock_score,
            4,
        )

    # ─────────────────────────────────────────
    # Delay score (vs baseline)
    # ─────────────────────────────────────────

    def _compute_delay_score(self) -> float:
        agent_delay = self._stats.get("avg_delay", 0.0)
        delay_ratio = agent_delay / self._baseline_delay
        return clamp(1.0 - (delay_ratio - 0.3), 0.0, 1.0)

    # ─────────────────────────────────────────
    # EV Coordination Chain — The Heart of Grading
    # ─────────────────────────────────────────

    def _grade_ev_coordination_chain(self) -> float:
        """
        Traces each EV's complete path. Scores every intersection:
          - PRE-CLEARING  (50%): correct phase on EV arrival?
          - ZERO-DELAY    (30%): EV passed without stopping?
          - FORWARD ALERT (20%): agent sent alert to next node?
        """
        ev_results = self._stats.get("emergency_results", [])
        if not ev_results:
            return 1.0  # No EVs in task

        ev_scores: List[float] = []

        for ev in ev_results:
            ev_id = ev["ev_id"]
            route = ev["route"]
            intersection_scores: List[float] = []

            for i, i_id in enumerate(route):
                if i_id in self._agent_cluster:
                    # ═══ Agent-controlled: full 3-link scoring ═══
                    score = self._score_agent_intersection(ev, ev_id, i_id, i, route)
                else:
                    # ═══ Heuristic-controlled: did agent's message help? ═══
                    score = self._score_heuristic_intersection(ev, ev_id, i_id)

                intersection_scores.append(score)

            if intersection_scores:
                ev_scores.append(
                    sum(intersection_scores) / len(intersection_scores)
                )

        if not ev_scores:
            return 1.0
        return round(sum(ev_scores) / len(ev_scores), 4)

    def _score_agent_intersection(
        self,
        ev: Dict[str, Any],
        ev_id: str,
        i_id: str,
        route_index: int,
        route: List[str],
    ) -> float:
        """Score an agent-controlled intersection in the EV's chain."""
        # Link 1: PRE-CLEARING (0.5 max)
        phase_correct = ev.get("phase_correct_on_arrival", {}).get(i_id, False)
        if phase_correct:
            pre_clear = 0.5
        else:
            reaction = self._ledger.get_reaction(ev_id, i_id)
            if reaction and reaction.detail == "REACTIVE":
                pre_clear = 0.3
            else:
                pre_clear = 0.0

        # Link 2: ZERO-DELAY PASS (0.3 max)
        delay = ev.get("delay_per_intersection", {}).get(i_id, 0.0)
        if delay == 0:
            pass_score = 0.3
        else:
            pass_score = max(0.3 - (delay / 30.0) * 0.3, 0.0)

        # Link 3: FORWARD ALERT (0.2 max)
        next_node = route[route_index + 1] if route_index + 1 < len(route) else None
        if next_node and next_node in self._heuristic_cluster:
            fwd = self._ledger.get_forward(ev_id, next_node)
            forward_score = 0.2 if (fwd and fwd.success) else 0.0
        elif next_node and next_node in self._agent_cluster:
            forward_score = 0.2  # Agent handles directly
        else:
            forward_score = 0.2  # EV exits network

        return pre_clear + pass_score + forward_score

    def _score_heuristic_intersection(
        self, ev: Dict[str, Any], ev_id: str, i_id: str
    ) -> float:
        """Score a heuristic-controlled intersection in the EV's chain."""
        delay = ev.get("delay_per_intersection", {}).get(i_id, 0.0)
        fwd = self._ledger.get_forward(ev_id, i_id)
        was_alerted = fwd and fwd.success

        if was_alerted:
            if delay == 0:
                return 1.0  # Alert + zero delay = perfect
            return max(0.7 - delay / 30.0, 0.3)
        else:
            if delay == 0:
                return 0.5  # Lucky
            return max(0.3 - delay / 30.0, 0.0)

    # ─────────────────────────────────────────
    # General message quality
    # ─────────────────────────────────────────

    def _grade_general_messages(self) -> float:
        """
        Score non-EV messages: traffic alerts, congestion warnings.
        40% completeness, 40% accuracy, 20% precision.
        """
        obligations = self._stats.get("message_obligations", [])
        sent = self._stats.get("non_ev_messages_sent", [])

        if not obligations:
            return 1.0 if not sent else 0.8

        hits = 0
        accuracy_sum = 0.0

        for ob in obligations:
            match = self._find_best_match(sent, ob)
            if match:
                hits += 1
                type_ok = 1.0 if match.get("type") == ob.get("expected_type") else 0.3
                urg_ok = 1.0 if match.get("urgency") == ob.get("expected_urgency") else 0.5
                tgt_ok = 1.0 if match.get("to_node") == ob.get("to_node") else 0.0
                accuracy_sum += (type_ok + urg_ok + tgt_ok) / 3.0

        completeness = hits / len(obligations)
        accuracy = accuracy_sum / max(hits, 1)
        precision = hits / max(len(sent), 1) if sent else 1.0

        return round(0.4 * completeness + 0.4 * accuracy + 0.2 * precision, 4)

    def _find_best_match(
        self, sent: List[Dict], obligation: Dict
    ) -> Optional[Dict]:
        """Find the best matching sent message for an obligation."""
        for msg in sent:
            if msg.get("to_node") == obligation.get("to_node"):
                return msg
        return None

    # ─────────────────────────────────────────
    # Equity & Gridlock
    # ─────────────────────────────────────────

    def _compute_equity_score(self) -> float:
        """Worst approach delay shouldn't be >> average."""
        avg = max(self._stats.get("avg_delay", 1.0), 0.1)
        worst = self._stats.get("max_approach_delay", avg)
        ratio = worst / avg
        return clamp(1.0 - (ratio - 1.0) / 3.0, 0.0, 1.0)

    def _compute_gridlock_score(self) -> float:
        """Penalize if any queue exceeds capacity."""
        max_queue = self._stats.get("max_queue", 0)
        capacity = 10  # Road length = queue capacity
        if max_queue < capacity:
            return 1.0
        return clamp(1.0 - (max_queue - capacity) / capacity, 0.0, 0.5)
