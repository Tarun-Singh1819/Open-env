"""
MessageEvaluator — Evaluates the quality of agent-sent messages.

Used by the grader to score non-EV coordination messages
(traffic alerts, congestion warnings, status updates).
Separate from the per-step reward — this is end-of-episode scoring.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..models import CoordinationMessage, MessageType, Urgency


class MessageObligation:
    """A message the agent SHOULD have sent at a particular step."""

    def __init__(
        self,
        step: int,
        to_node: str,
        expected_type: MessageType,
        expected_urgency: Urgency,
        context: str = "",
    ) -> None:
        self.step = step
        self.to_node = to_node
        self.expected_type = expected_type
        self.expected_urgency = expected_urgency
        self.context = context


class MessageEvaluator:
    """
    Post-episode evaluation of all agent-sent messages.

    Scoring breakdown:
    - Completeness (40%): Did the agent send messages when obligations existed?
    - Accuracy    (40%): Were message_type, urgency, and target correct?
    - Precision   (20%): Ratio of useful messages to total sent (penalizes spam).
    """

    def __init__(
        self,
        obligations: List[MessageObligation],
        sent_messages: List[CoordinationMessage],
    ) -> None:
        self._obligations = obligations
        self._sent = sent_messages

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate message quality. Returns:
        {
            "score": float in [0, 1],
            "completeness": float,
            "accuracy": float,
            "precision": float,
            "hits": int,
            "obligations": int,
            "total_sent": int,
        }
        """
        non_ev_sent = [
            m for m in self._sent
            if m.message_type != MessageType.EMERGENCY_ALERT
        ]

        if not self._obligations:
            # No obligations — score based on whether agent was spammy
            score = 1.0 if not non_ev_sent else 0.8
            return {
                "score": score,
                "completeness": 1.0,
                "accuracy": 1.0,
                "precision": score,
                "hits": 0,
                "obligations": 0,
                "total_sent": len(non_ev_sent),
            }

        hits = 0
        accuracy_sum = 0.0

        for ob in self._obligations:
            match = self._find_best_match(non_ev_sent, ob)
            if match:
                hits += 1
                type_ok = 1.0 if match.message_type == ob.expected_type else 0.3
                urgency_ok = 1.0 if match.urgency == ob.expected_urgency else 0.5
                target_ok = 1.0 if match.to_node == ob.to_node else 0.0
                accuracy_sum += (type_ok + urgency_ok + target_ok) / 3.0

        completeness = hits / len(self._obligations)
        accuracy = accuracy_sum / max(hits, 1)
        precision = hits / max(len(non_ev_sent), 1) if non_ev_sent else 1.0

        score = 0.4 * completeness + 0.4 * accuracy + 0.2 * precision

        return {
            "score": round(score, 4),
            "completeness": round(completeness, 4),
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "hits": hits,
            "obligations": len(self._obligations),
            "total_sent": len(non_ev_sent),
        }

    def _find_best_match(
        self,
        sent: List[CoordinationMessage],
        obligation: MessageObligation,
    ) -> Optional[CoordinationMessage]:
        """Find the best matching sent message for an obligation."""
        # Prioritize exact target + type match
        for msg in sent:
            if (
                msg.to_node == obligation.to_node
                and msg.message_type == obligation.expected_type
            ):
                return msg

        # Fallback: just target match
        for msg in sent:
            if msg.to_node == obligation.to_node:
                return msg

        return None
