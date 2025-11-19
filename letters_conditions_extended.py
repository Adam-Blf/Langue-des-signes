"""Extended rule-based heuristics for complete LSF alphabet (A-Z).

This module extends letters_conditions.py to support all 26 letters of the
French Sign Language alphabet with additional heuristic rules.
"""

from __future__ import annotations

from typing import Optional
from letters_conditions import (
    Landmark,
    LandmarkSequence,
    _ensure_landmarks,
    _distance_x,
    _distance_y,
    _are_extended,
    _are_folded,
    # Re-import landmark indices
    WRIST, THUMB_IP, THUMB_TIP, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP,
    MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP, RING_PIP, RING_DIP, RING_TIP,
    PINKY_PIP, PINKY_DIP, PINKY_TIP,
    # Import existing detectors
    _detect_a, _detect_b, _detect_c, _detect_d, _detect_f
)


def _detect_e(landmarks: LandmarkSequence) -> Optional[str]:
    """E: All fingers folded with thumb tip touching index middle finger."""
    thumb_tip = landmarks[THUMB_TIP]
    index_mid = landmarks[INDEX_PIP]
    tips = [landmarks[i] for i in (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)]
    mcps = [landmarks[i] for i in (INDEX_MCP, 9, 13, 17)]
    
    all_folded = all(tip.y > mcp.y for tip, mcp in zip(tips, mcps))
    thumb_touching = _distance_x(thumb_tip, index_mid) < 0.03 and _distance_y(thumb_tip, index_mid) < 0.03
    
    if all_folded and thumb_touching:
        return "E"
    return None


def _detect_g(landmarks: LandmarkSequence) -> Optional[str]:
    """G: Index extended horizontally, thumb out, others folded."""
    index_tip = landmarks[INDEX_TIP]
    index_mcp = landmarks[INDEX_MCP]
    thumb_tip = landmarks[THUMB_TIP]
    wrist = landmarks[WRIST]
    
    # Index horizontal (x distance > y distance from mcp to tip)
    index_horizontal = _distance_x(index_tip, index_mcp) > _distance_y(index_tip, index_mcp)
    thumb_extended = _distance_x(thumb_tip, wrist) > 0.08
    
    other_tips = [landmarks[i] for i in (MIDDLE_TIP, RING_TIP, PINKY_TIP)]
    other_mcps = [landmarks[i] for i in (9, 13, 17)]
    others_folded = all(tip.y > mcp.y for tip, mcp in zip(other_tips, other_mcps))
    
    if index_horizontal and thumb_extended and others_folded:
        return "G"
    return None


def _detect_h(landmarks: LandmarkSequence) -> Optional[str]:
    """H: Index and middle extended horizontally side by side."""
    index_tip = landmarks[INDEX_TIP]
    middle_tip = landmarks[MIDDLE_TIP]
    index_mcp = landmarks[INDEX_MCP]
    
    # Both extended
    tips = [index_tip, middle_tip]
    dips = [landmarks[INDEX_DIP], landmarks[MIDDLE_DIP]]
    both_extended = _are_extended(tips, dips)
    
    # Horizontal
    horizontal = _distance_x(index_tip, index_mcp) > _distance_y(index_tip, index_mcp)
    
    # Close together
    close_together = _distance_x(index_tip, middle_tip) < 0.03
    
    # Others folded
    other_tips = [landmarks[RING_TIP], landmarks[PINKY_TIP]]
    other_mcps = [landmarks[13], landmarks[17]]
    others_folded = all(tip.y > mcp.y for tip, mcp in zip(other_tips, other_mcps))
    
    if both_extended and horizontal and close_together and others_folded:
        return "H"
    return None


def _detect_i(landmarks: LandmarkSequence) -> Optional[str]:
    """I: Pinky extended, all others folded."""
    pinky_tip = landmarks[PINKY_TIP]
    pinky_dip = landmarks[PINKY_DIP]
    
    pinky_extended = pinky_tip.y < pinky_dip.y
    
    other_tips = [landmarks[i] for i in (INDEX_TIP, MIDDLE_TIP, RING_TIP)]
    other_pips = [landmarks[i] for i in (INDEX_PIP, MIDDLE_PIP, RING_PIP)]
    others_folded = _are_folded(other_tips, other_pips)
    
    if pinky_extended and others_folded:
        return "I"
    return None


def _detect_j(landmarks: LandmarkSequence) -> Optional[str]:
    """J: Pinky extended making J curve motion (detected by orientation)."""
    pinky_tip = landmarks[PINKY_TIP]
    pinky_dip = landmarks[PINKY_DIP]
    pinky_pip = landmarks[PINKY_PIP]
    
    # Pinky extended and curved
    pinky_extended = pinky_tip.y < pinky_pip.y
    curved = _distance_x(pinky_tip, pinky_dip) > 0.02
    
    if pinky_extended and curved:
        return "J"
    return None


def _detect_k(landmarks: LandmarkSequence) -> Optional[str]:
    """K: Index and middle extended in V shape, thumb between them."""
    index_tip = landmarks[INDEX_TIP]
    middle_tip = landmarks[MIDDLE_TIP]
    thumb_tip = landmarks[THUMB_TIP]
    index_dip = landmarks[INDEX_DIP]
    middle_dip = landmarks[MIDDLE_DIP]
    
    # Both extended
    both_extended = index_tip.y < index_dip.y and middle_tip.y < middle_dip.y
    
    # V shape (fingers apart)
    v_shape = _distance_x(index_tip, middle_tip) > 0.05
    
    # Thumb between them
    thumb_between = (
        min(index_tip.x, middle_tip.x) < thumb_tip.x < max(index_tip.x, middle_tip.x)
    )
    
    if both_extended and v_shape and thumb_between:
        return "K"
    return None


def _detect_l(landmarks: LandmarkSequence) -> Optional[str]:
    """L: Thumb and index extended at 90 degrees."""
    thumb_tip = landmarks[THUMB_TIP]
    index_tip = landmarks[INDEX_TIP]
    thumb_ip = landmarks[THUMB_IP]
    index_dip = landmarks[INDEX_DIP]
    
    # Both extended
    thumb_extended = thumb_tip.y < thumb_ip.y
    index_extended = index_tip.y < index_dip.y
    
    # 90 degree angle (x distance significant, y distance significant)
    angle_90 = _distance_x(thumb_tip, index_tip) > 0.08 and _distance_y(thumb_tip, index_tip) > 0.08
    
    # Others folded
    other_tips = [landmarks[i] for i in (MIDDLE_TIP, RING_TIP, PINKY_TIP)]
    other_pips = [landmarks[i] for i in (MIDDLE_PIP, RING_PIP, PINKY_PIP)]
    others_folded = _are_folded(other_tips, other_pips)
    
    if thumb_extended and index_extended and angle_90 and others_folded:
        return "L"
    return None


def _detect_m(landmarks: LandmarkSequence) -> Optional[str]:
    """M: Thumb under three folded fingers (index, middle, ring)."""
    thumb_tip = landmarks[THUMB_TIP]
    index_tip = landmarks[INDEX_TIP]
    middle_tip = landmarks[MIDDLE_TIP]
    ring_tip = landmarks[RING_TIP]
    
    # Three fingers folded
    tips = [index_tip, middle_tip, ring_tip]
    pips = [landmarks[INDEX_PIP], landmarks[MIDDLE_PIP], landmarks[RING_PIP]]
    three_folded = _are_folded(tips, pips)
    
    # Thumb under them
    thumb_under = thumb_tip.y > min(index_tip.y, middle_tip.y, ring_tip.y)
    
    if three_folded and thumb_under:
        return "M"
    return None


def _detect_n(landmarks: LandmarkSequence) -> Optional[str]:
    """N: Thumb under two folded fingers (index, middle)."""
    thumb_tip = landmarks[THUMB_TIP]
    index_tip = landmarks[INDEX_TIP]
    middle_tip = landmarks[MIDDLE_TIP]
    
    # Two fingers folded
    tips = [index_tip, middle_tip]
    pips = [landmarks[INDEX_PIP], landmarks[MIDDLE_PIP]]
    two_folded = _are_folded(tips, pips)
    
    # Thumb under them
    thumb_under = thumb_tip.y > min(index_tip.y, middle_tip.y)
    
    # Ring and pinky folded
    ring_folded = landmarks[RING_TIP].y > landmarks[RING_PIP].y
    pinky_folded = landmarks[PINKY_TIP].y > landmarks[PINKY_PIP].y
    
    if two_folded and thumb_under and ring_folded and pinky_folded:
        return "N"
    return None


def _detect_o(landmarks: LandmarkSequence) -> Optional[str]:
    """O: All fingertips touch forming a circle."""
    thumb_tip = landmarks[THUMB_TIP]
    index_tip = landmarks[INDEX_TIP]
    middle_tip = landmarks[MIDDLE_TIP]
    ring_tip = landmarks[RING_TIP]
    pinky_tip = landmarks[PINKY_TIP]
    
    # Calculate center point
    center_x = (thumb_tip.x + index_tip.x + middle_tip.x + ring_tip.x + pinky_tip.x) / 5
    center_y = (thumb_tip.y + index_tip.y + middle_tip.y + ring_tip.y + pinky_tip.y) / 5
    center = Landmark(center_x, center_y)
    
    # All tips should be close to each other (forming circle)
    all_close = all(
        _distance_x(tip, center) < 0.04 and _distance_y(tip, center) < 0.04
        for tip in [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
    )
    
    if all_close:
        return "O"
    return None


def _detect_p(landmarks: LandmarkSequence) -> Optional[str]:
    """P: Similar to K but pointing down."""
    index_tip = landmarks[INDEX_TIP]
    middle_tip = landmarks[MIDDLE_TIP]
    index_mcp = landmarks[INDEX_MCP]
    
    # Both extended downward
    both_down = index_tip.y > index_mcp.y and middle_tip.y > landmarks[9].y
    
    # V shape
    v_shape = _distance_x(index_tip, middle_tip) > 0.04
    
    if both_down and v_shape:
        return "P"
    return None


def _detect_q(landmarks: LandmarkSequence) -> Optional[str]:
    """Q: Similar to G but pointing down."""
    index_tip = landmarks[INDEX_TIP]
    thumb_tip = landmarks[THUMB_TIP]
    index_mcp = landmarks[INDEX_MCP]
    
    # Index pointing down
    index_down = index_tip.y > index_mcp.y
    
    # Thumb extended
    thumb_extended = _distance_x(thumb_tip, index_mcp) > 0.05
    
    if index_down and thumb_extended:
        return "Q"
    return None


def _detect_r(landmarks: LandmarkSequence) -> Optional[str]:
    """R: Index and middle crossed."""
    index_tip = landmarks[INDEX_TIP]
    middle_tip = landmarks[MIDDLE_TIP]
    index_dip = landmarks[INDEX_DIP]
    middle_dip = landmarks[MIDDLE_DIP]
    
    # Both extended
    both_extended = index_tip.y < index_dip.y and middle_tip.y < middle_dip.y
    
    # Crossed (x positions swapped relative to base)
    crossed = abs(index_tip.x - middle_tip.x) < 0.02
    
    if both_extended and crossed:
        return "R"
    return None


def _detect_s(landmarks: LandmarkSequence) -> Optional[str]:
    """S: Fist with thumb in front."""
    thumb_tip = landmarks[THUMB_TIP]
    index_mcp = landmarks[INDEX_MCP]
    
    # All fingers folded
    tips = [landmarks[i] for i in (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)]
    mcps = [landmarks[i] for i in (INDEX_MCP, 9, 13, 17)]
    all_folded = all(tip.y > mcp.y for tip, mcp in zip(tips, mcps))
    
    # Thumb in front (not tucked)
    thumb_out = thumb_tip.y < index_mcp.y and _distance_x(thumb_tip, index_mcp) < 0.05
    
    if all_folded and thumb_out:
        return "S"
    return None


def _detect_t(landmarks: LandmarkSequence) -> Optional[str]:
    """T: Thumb between index and middle (like M but thumb visible)."""
    thumb_tip = landmarks[THUMB_TIP]
    index_tip = landmarks[INDEX_TIP]
    middle_tip = landmarks[MIDDLE_TIP]
    
    # Thumb between fingers
    thumb_between = (
        min(index_tip.x, middle_tip.x) < thumb_tip.x < max(index_tip.x, middle_tip.x)
        and thumb_tip.y < index_tip.y
    )
    
    if thumb_between:
        return "T"
    return None


def _detect_u(landmarks: LandmarkSequence) -> Optional[str]:
    """U: Index and middle extended together pointing up."""
    index_tip = landmarks[INDEX_TIP]
    middle_tip = landmarks[MIDDLE_TIP]
    index_dip = landmarks[INDEX_DIP]
    middle_dip = landmarks[MIDDLE_DIP]
    
    # Both extended up
    both_up = index_tip.y < index_dip.y and middle_tip.y < middle_dip.y
    
    # Close together
    close = _distance_x(index_tip, middle_tip) < 0.02
    
    # Others folded
    other_tips = [landmarks[RING_TIP], landmarks[PINKY_TIP]]
    other_pips = [landmarks[RING_PIP], landmarks[PINKY_PIP]]
    others_folded = _are_folded(other_tips, other_pips)
    
    if both_up and close and others_folded:
        return "U"
    return None


def _detect_v(landmarks: LandmarkSequence) -> Optional[str]:
    """V: Index and middle extended in V shape."""
    index_tip = landmarks[INDEX_TIP]
    middle_tip = landmarks[MIDDLE_TIP]
    index_dip = landmarks[INDEX_DIP]
    middle_dip = landmarks[MIDDLE_DIP]
    
    # Both extended
    both_extended = index_tip.y < index_dip.y and middle_tip.y < middle_dip.y
    
    # V shape (apart)
    v_shape = _distance_x(index_tip, middle_tip) > 0.04
    
    # Others folded
    other_tips = [landmarks[RING_TIP], landmarks[PINKY_TIP]]
    other_pips = [landmarks[RING_PIP], landmarks[PINKY_PIP]]
    others_folded = _are_folded(other_tips, other_pips)
    
    if both_extended and v_shape and others_folded:
        return "V"
    return None


def _detect_w(landmarks: LandmarkSequence) -> Optional[str]:
    """W: Index, middle, and ring extended in W shape."""
    tips = [landmarks[i] for i in (INDEX_TIP, MIDDLE_TIP, RING_TIP)]
    dips = [landmarks[i] for i in (INDEX_DIP, MIDDLE_DIP, RING_DIP)]
    
    # Three extended
    three_extended = _are_extended(tips, dips)
    
    # Spread apart
    spread = all(
        _distance_x(tips[i], tips[i+1]) > 0.03
        for i in range(len(tips)-1)
    )
    
    # Pinky folded
    pinky_folded = landmarks[PINKY_TIP].y > landmarks[PINKY_PIP].y
    
    if three_extended and spread and pinky_folded:
        return "W"
    return None


def _detect_x(landmarks: LandmarkSequence) -> Optional[str]:
    """X: Index hooked."""
    index_tip = landmarks[INDEX_TIP]
    index_dip = landmarks[INDEX_DIP]
    index_pip = landmarks[INDEX_PIP]
    
    # Index hooked (tip below dip, but above pip)
    hooked = index_pip.y < index_tip.y < index_dip.y
    
    # Others folded
    other_tips = [landmarks[i] for i in (MIDDLE_TIP, RING_TIP, PINKY_TIP)]
    other_pips = [landmarks[i] for i in (MIDDLE_PIP, RING_PIP, PINKY_PIP)]
    others_folded = _are_folded(other_tips, other_pips)
    
    if hooked and others_folded:
        return "X"
    return None


def _detect_y(landmarks: LandmarkSequence) -> Optional[str]:
    """Y: Thumb and pinky extended (hang loose gesture)."""
    thumb_tip = landmarks[THUMB_TIP]
    pinky_tip = landmarks[PINKY_TIP]
    thumb_ip = landmarks[THUMB_IP]
    pinky_dip = landmarks[PINKY_DIP]
    
    # Both extended
    thumb_extended = thumb_tip.y < thumb_ip.y or _distance_x(thumb_tip, landmarks[WRIST]) > 0.08
    pinky_extended = pinky_tip.y < pinky_dip.y
    
    # Others folded
    other_tips = [landmarks[i] for i in (INDEX_TIP, MIDDLE_TIP, RING_TIP)]
    other_pips = [landmarks[i] for i in (INDEX_PIP, MIDDLE_PIP, RING_PIP)]
    others_folded = _are_folded(other_tips, other_pips)
    
    if thumb_extended and pinky_extended and others_folded:
        return "Y"
    return None


def _detect_z(landmarks: LandmarkSequence) -> Optional[str]:
    """Z: Index extended making Z motion (detected by angle)."""
    index_tip = landmarks[INDEX_TIP]
    index_dip = landmarks[INDEX_DIP]
    index_mcp = landmarks[INDEX_MCP]
    
    # Index extended
    index_extended = index_tip.y < index_dip.y
    
    # Diagonal orientation (Z shape approximation)
    diagonal = abs(_distance_x(index_tip, index_mcp) - _distance_y(index_tip, index_mcp)) < 0.03
    
    if index_extended and diagonal:
        return "Z"
    return None


# Complete detector list for all 26 letters
EXTENDED_DETECTORS = (
    _detect_a, _detect_b, _detect_c, _detect_d, _detect_e, _detect_f,
    _detect_g, _detect_h, _detect_i, _detect_j, _detect_k, _detect_l,
    _detect_m, _detect_n, _detect_o, _detect_p, _detect_q, _detect_r,
    _detect_s, _detect_t, _detect_u, _detect_v, _detect_w, _detect_x,
    _detect_y, _detect_z
)


def detect_letter_extended(hand_landmarks) -> Optional[str]:
    """Return the detected letter from full A-Z alphabet or None."""
    landmarks = _ensure_landmarks(hand_landmarks)
    if len(landmarks) < 21:
        return None

    return next(
        (letter for detector in EXTENDED_DETECTORS if (letter := detector(landmarks)) is not None),
        None,
    )
