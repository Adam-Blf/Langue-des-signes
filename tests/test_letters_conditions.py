from letters_conditions import Landmark, detect_letter


def _base_hand():
    hand = [Landmark(0.0, 0.0, 0.0) for _ in range(21)]

    # Thumb
    hand[1] = Landmark(-0.05, 0.15, 0.0)
    hand[2] = Landmark(-0.1, 0.18, 0.0)
    hand[3] = Landmark(-0.12, 0.22, 0.0)
    hand[4] = Landmark(-0.08, 0.24, 0.0)

    # Index
    hand[5] = Landmark(0.1, 0.2, 0.0)
    hand[6] = Landmark(0.1, 0.25, 0.0)
    hand[7] = Landmark(0.1, 0.3, 0.0)
    hand[8] = Landmark(0.1, 0.35, 0.0)

    # Middle
    hand[9] = Landmark(0.2, 0.2, 0.0)
    hand[10] = Landmark(0.2, 0.25, 0.0)
    hand[11] = Landmark(0.2, 0.3, 0.0)
    hand[12] = Landmark(0.2, 0.35, 0.0)

    # Ring
    hand[13] = Landmark(0.3, 0.2, 0.0)
    hand[14] = Landmark(0.3, 0.25, 0.0)
    hand[15] = Landmark(0.3, 0.3, 0.0)
    hand[16] = Landmark(0.3, 0.35, 0.0)

    # Pinky
    hand[17] = Landmark(0.4, 0.2, 0.0)
    hand[18] = Landmark(0.4, 0.25, 0.0)
    hand[19] = Landmark(0.4, 0.3, 0.0)
    hand[20] = Landmark(0.4, 0.35, 0.0)

    return hand


def _hand_letter_a():
    hand = _base_hand()
    # Fold the fingers by moving tips below MCP
    hand[8] = Landmark(0.1, 0.55, 0.0)
    hand[12] = Landmark(0.2, 0.55, 0.0)
    hand[16] = Landmark(0.3, 0.55, 0.0)
    hand[20] = Landmark(0.4, 0.55, 0.0)
    # Lift the thumb and move it away from the index
    hand[3] = Landmark(-0.15, 0.2, 0.0)
    hand[4] = Landmark(-0.2, 0.1, 0.0)
    return hand


def _hand_letter_b():
    hand = _base_hand()
    # Extend fingers by moving tips above DIP joints
    hand[8] = Landmark(0.1, 0.15, 0.0)
    hand[12] = Landmark(0.2, 0.15, 0.0)
    hand[16] = Landmark(0.3, 0.15, 0.0)
    hand[20] = Landmark(0.4, 0.15, 0.0)
    # Lower the thumb alongside the palm
    hand[4] = Landmark(0.12, 0.3, 0.0)
    return hand


def test_detect_letter_a():
    assert detect_letter(_hand_letter_a()) == "A"


def test_detect_letter_b():
    assert detect_letter(_hand_letter_b()) == "B"


def test_detect_letter_none_for_neutral_pose():
    assert detect_letter(_base_hand()) is None
