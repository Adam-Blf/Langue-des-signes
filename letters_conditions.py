"""Rule-based heuristics that classify hand landmarks into letters.

The conditions are intentionally simple and work as a quick fallback when the
machine-learning model cannot provide a confident prediction.  The module keeps
its dependencies minimal so it can be unit-tested without mediapipe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence


@dataclass(frozen=True)
class Landmark:
    """Simple container used for typing and testing."""

    x: float
    y: float
    z: float = 0.0


LandmarkSequence = Sequence[Landmark]

# Mediapipe landmark indices
WRIST = 0
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20


def _ensure_landmarks(hand_landmarks) -> LandmarkSequence:
    """Return a sequence of landmark-like objects."""
    if isinstance(hand_landmarks, Sequence):
        return hand_landmarks  # Already a sequence of simple landmarks

    # Mediapipe hands landmark object exposes a `.landmark` attribute
    return hand_landmarks.landmark


def _distance_x(p1: Landmark, p2: Landmark) -> float:
    return abs(p1.x - p2.x)


def _distance_y(p1: Landmark, p2: Landmark) -> float:
    return abs(p1.y - p2.y)


def _are_extended(tips: Iterable[Landmark], joints: Iterable[Landmark]) -> bool:
    """Return True when each tip is above (y smaller than) its joint."""
    return all(tip.y < joint.y for tip, joint in zip(tips, joints))


def _are_folded(tips: Iterable[Landmark], joints: Iterable[Landmark]) -> bool:
    """Return True when each tip is below (y greater than) its joint."""
    return all(tip.y > joint.y for tip, joint in zip(tips, joints))


def _detect_a(landmarks: LandmarkSequence) -> Optional[str]:
    thumb_tip = landmarks[THUMB_TIP]
    thumb_ip = landmarks[THUMB_IP]
    index_tip = landmarks[INDEX_TIP]
    finger_mcp = [landmarks[i] for i in (INDEX_MCP, 9, 13, 17)]

    if not _are_folded([landmarks[i] for i in (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)], finger_mcp):
        return None

    if thumb_tip.y >= thumb_ip.y:
        return None

    if _distance_x(thumb_tip, index_tip) <= 0.05:
        return None

    return "A"


def _detect_b(landmarks: LandmarkSequence) -> Optional[str]:
    thumb_tip = landmarks[THUMB_TIP]
    index_mcp = landmarks[INDEX_MCP]
    tips = [landmarks[i] for i in (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)]
    dips = [landmarks[i] for i in (INDEX_DIP, MIDDLE_DIP, RING_DIP, PINKY_DIP)]
    if not _are_extended(tips, dips):
        return None

    if _distance_x(thumb_tip, index_mcp) >= 0.05:
        return None

    if thumb_tip.y <= index_mcp.y:
        return None

    return "B"


def _detect_c(landmarks: LandmarkSequence) -> Optional[str]:
    thumb_tip = landmarks[THUMB_TIP]
    index_tip = landmarks[INDEX_TIP]
    index_mcp = landmarks[INDEX_MCP]
    tips = [landmarks[i] for i in (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)]
    dips = [landmarks[i] for i in (INDEX_DIP, MIDDLE_DIP, RING_DIP, PINKY_DIP)]
    mcps = [landmarks[i] for i in (INDEX_MCP, 9, 13, 17)]

    tips_under_dips = all(tip.y > dip.y for tip, dip in zip(tips, dips))
    tips_forward = all(_distance_x(tip, mcp) > 0.02 for tip, mcp in zip(tips, mcps))
    thumb_index_gap = _distance_y(thumb_tip, index_tip)
    thumb_forward = _distance_x(thumb_tip, index_mcp) > 0.05

    if tips_under_dips and tips_forward and thumb_forward and 0.05 < thumb_index_gap < 0.4:
        return "C"

    return None


def _detect_d(landmarks: LandmarkSequence) -> Optional[str]:
    index_tip = landmarks[INDEX_TIP]
    index_dip = landmarks[INDEX_DIP]
    middle_tip = landmarks[MIDDLE_TIP]
    middle_pip = landmarks[MIDDLE_PIP]
    ring_tip = landmarks[RING_TIP]
    ring_pip = landmarks[RING_PIP]
    pinky_tip = landmarks[PINKY_TIP]
    pinky_pip = landmarks[PINKY_PIP]
    thumb_tip = landmarks[THUMB_TIP]

    index_extended = index_tip.y < index_dip.y
    others_folded = all(
        finger_tip.y > finger_pip.y
        for finger_tip, finger_pip in (
            (middle_tip, middle_pip),
            (ring_tip, ring_pip),
            (pinky_tip, pinky_pip),
        )
    )
    thumb_near_middle = _distance_x(thumb_tip, middle_tip) < 0.02 and _distance_y(thumb_tip, middle_tip) < 0.04

    if index_extended and others_folded and thumb_near_middle:
        return "D"

    return None


def _detect_f(landmarks: LandmarkSequence) -> Optional[str]:
    thumb_tip = landmarks[THUMB_TIP]
    index_tip = landmarks[INDEX_TIP]
    index_pip = landmarks[INDEX_PIP]
    ring_mcp = landmarks[13]
    pinky_mcp = landmarks[17]
    middle_mcp = landmarks[9]

    other_mcp = [middle_mcp, ring_mcp, pinky_mcp]
    other_tips = [landmarks[i] for i in (MIDDLE_TIP, RING_TIP, PINKY_TIP)]

    index_curled = index_tip.y > index_pip.y
    thumb_over_index = thumb_tip.y < index_tip.y
    thumb_index_close = _distance_x(thumb_tip, index_tip) < 0.05
    others_straight = _are_extended(other_tips, other_mcp)

    if index_curled and thumb_over_index and thumb_index_close and others_straight:
        return "F"

    return None


DETECTORS = (_detect_a, _detect_b, _detect_c, _detect_d, _detect_f)


def detect_letter(hand_landmarks) -> Optional[str]:
    """Return the detected letter or ``None`` when no rule matches."""
    landmarks = _ensure_landmarks(hand_landmarks)
    if len(landmarks) < 21:
        return None

    return next(
        (letter for detector in DETECTORS if (letter := detector(landmarks)) is not None),
        None,
    )
