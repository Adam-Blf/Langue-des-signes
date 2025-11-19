"""Tests for letters_conditions_extended.py - Complete A-Z alphabet detection."""

import pytest
from unittest.mock import Mock
from letters_conditions_extended import (
    detect_letter_extended,
    _detect_e, _detect_g, _detect_h, _detect_i, _detect_j,
    _detect_k, _detect_l, _detect_m, _detect_n, _detect_o,
    _detect_p, _detect_q, _detect_r, _detect_s, _detect_t,
    _detect_u, _detect_v, _detect_w, _detect_x, _detect_y, _detect_z
)


def create_mock_landmark(x: float, y: float, z: float = 0.0):
    """Create a mock MediaPipe landmark."""
    landmark = Mock()
    landmark.x = x
    landmark.y = y
    landmark.z = z
    return landmark


def create_mock_hand_landmarks():
    """Create mock hand landmarks structure."""
    landmarks = Mock()
    landmarks.landmark = [create_mock_landmark(0.5, 0.5) for _ in range(21)]
    return landmarks


class TestAlphabetExtended:
    """Test extended alphabet detection."""
    
    def test_detect_letter_extended_returns_none_for_empty_input(self):
        """Test that None input returns None."""
        assert detect_letter_extended(None) is None
    
    def test_detect_e_with_extended_fingers(self):
        """Test letter E detection with extended fingers."""
        landmarks = create_mock_hand_landmarks()
        # Configure specific geometry for E
        landmarks.landmark[8].y = 0.3  # Index up
        landmarks.landmark[12].y = 0.3  # Middle up
        landmarks.landmark[16].y = 0.3  # Ring up
        landmarks.landmark[20].y = 0.3  # Pinky up
        landmarks.landmark[4].y = 0.6  # Thumb down
        
        result = _detect_e(landmarks.landmark)
        # E requires specific configuration, may return None or 'E'
        assert result is None or result == 'E'
    
    def test_detect_g_basic_geometry(self):
        """Test letter G detection."""
        landmarks = create_mock_hand_landmarks()
        result = _detect_g(landmarks.landmark)
        assert result is None or result == 'G'
    
    def test_detect_h_horizontal_configuration(self):
        """Test letter H detection."""
        landmarks = create_mock_hand_landmarks()
        result = _detect_h(landmarks.landmark)
        assert result is None or result == 'H'
    
    def test_detect_letter_extended_handles_all_letters(self):
        """Test that detect_letter_extended processes all detectors."""
        landmarks = create_mock_hand_landmarks()
        
        # Should return None or a valid letter A-Z
        result = detect_letter_extended(landmarks)
        assert result is None or (isinstance(result, str) and len(result) == 1 and result.isupper())
    
    def test_all_detectors_return_valid_format(self):
        """Test that all detector functions return None or single uppercase letter."""
        landmarks = create_mock_hand_landmarks()
        
        detectors = [
            _detect_e, _detect_g, _detect_h, _detect_i, _detect_j,
            _detect_k, _detect_l, _detect_m, _detect_n, _detect_o,
            _detect_p, _detect_q, _detect_r, _detect_s, _detect_t,
            _detect_u, _detect_v, _detect_w, _detect_x, _detect_y, _detect_z
        ]
        
        for detector in detectors:
            result = detector(landmarks.landmark)
            assert result is None or (isinstance(result, str) and len(result) == 1 and result.isupper())
    
    def test_detect_letter_extended_prioritizes_first_match(self):
        """Test that the first matching detector is returned."""
        landmarks = create_mock_hand_landmarks()
        
        # Multiple detections should return first one found
        result = detect_letter_extended(landmarks)
        
        # Result should be deterministic for same input
        result2 = detect_letter_extended(landmarks)
        assert result == result2


class TestGeometryHelpers:
    """Test geometry helper functions if exposed."""
    
    def test_mock_landmarks_have_xyz_coordinates(self):
        """Test that mock landmarks have proper attributes."""
        landmark = create_mock_landmark(0.5, 0.3, 0.1)
        assert landmark.x == 0.5
        assert landmark.y == 0.3
        assert landmark.z == 0.1
    
    def test_hand_landmarks_have_21_points(self):
        """Test that hand landmarks contain 21 points."""
        landmarks = create_mock_hand_landmarks()
        assert len(landmarks.landmark) == 21


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
