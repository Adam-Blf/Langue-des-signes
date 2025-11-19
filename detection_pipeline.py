from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Optional, Sequence, Tuple

from letters_conditions import detect_letter
from predict_ml import predict_ml_with_confidence

WRIST_INDEX = 0


@dataclass(frozen=True)
class RawDetection:
    letter: str
    source: str
    confidence: float


@dataclass(frozen=True)
class DetectionResult:
    letter: Optional[str]
    source: Optional[str]
    confidence: Optional[float]
    raw_letter: Optional[str]
    raw_source: Optional[str]
    raw_confidence: Optional[float]


class SignDetectionPipeline:
    """Combine rule-based and ML predictions with simple temporal smoothing."""

    def __init__(
        self,
        *,
        history_size: int = 5,
        min_consensus: int = 2,
        max_misses: int = 5,
        ml_threshold: float = 0.83,
        enable_rules: bool = True,
        enable_ml: bool = True,
    ) -> None:
        self.history: Deque[RawDetection] = deque(maxlen=history_size)
        self.miss_count = 0
        self.min_consensus = min_consensus
        self.max_misses = max_misses
        self.ml_threshold = ml_threshold
        self.enable_rules = enable_rules
        self.enable_ml = enable_ml

    def process(self, hand_landmarks) -> DetectionResult:
        """Process a new frame and return both raw and stabilised detections."""
        raw_detection = self._detect(hand_landmarks)
        if raw_detection:
            self.history.append(raw_detection)
            self.miss_count = 0
        else:
            self.miss_count += 1
            if self.miss_count >= self.max_misses:
                self.history.clear()

        stable_detection = self._resolve_history()
        return DetectionResult(
            letter=stable_detection[0] if stable_detection else None,
            source=stable_detection[1] if stable_detection else None,
            confidence=stable_detection[2] if stable_detection else None,
            raw_letter=raw_detection.letter if raw_detection else None,
            raw_source=raw_detection.source if raw_detection else None,
            raw_confidence=raw_detection.confidence if raw_detection else None,
        )

    def reset(self) -> None:
        """Clear the temporal history."""
        self.history.clear()
        self.miss_count = 0

    def _detect(self, hand_landmarks) -> Optional[RawDetection]:
        """Run the rule-based and ML detectors on a single frame."""
        if hand_landmarks is None:
            return None

        if self.enable_rules:
            letter = detect_letter(hand_landmarks)
            if letter:
                return RawDetection(letter=letter, source="rules", confidence=1.0)

        if self.enable_ml:
            flat = extract_flattened_landmarks(hand_landmarks)
            if flat is None:
                return None
            prediction = predict_ml_with_confidence(flat, self.ml_threshold)
            if prediction:
                return RawDetection(
                    letter=prediction.label,
                    source="ml",
                    confidence=prediction.confidence,
                )

        return None

    def _resolve_history(self) -> Optional[Tuple[str, str, float]]:
        """Return the stabilised detection once enough votes are collected."""
        if not self.history:
            return None

        counts = Counter(item.letter for item in self.history)
        letter, votes = counts.most_common(1)[0]
        if votes < self.min_consensus:
            return None

        confidence = sum(item.confidence for item in self.history if item.letter == letter) / votes

        # Pick the most recent source that produced the winning letter
        for item in reversed(self.history):
            if item.letter == letter:
                return letter, item.source, confidence

        return None


def extract_flattened_landmarks(hand_landmarks) -> Optional[Sequence[float]]:
    """Return a flattened list of relative landmark coordinates."""
    try:
        landmarks: Sequence = getattr(hand_landmarks, "landmark", hand_landmarks)  # type: ignore[assignment]
        if not isinstance(landmarks, Sequence) or len(landmarks) < 21:
            return None

        wrist = landmarks[WRIST_INDEX]
        if not hasattr(wrist, "x"):
            return None

        flattened = []
        for landmark in landmarks:
            try:
                flattened.extend(
                    (
                        float(landmark.x) - float(wrist.x),
                        float(landmark.y) - float(wrist.y),
                        float(getattr(landmark, "z", 0.0)) - float(getattr(wrist, "z", 0.0)),
                    )
                )
            except (AttributeError, ValueError, TypeError):
                # Skip malformed landmarks
                continue
        
        # Verify we have the expected number of coordinates (21 landmarks * 3 coords)
        if len(flattened) != 63:
            return None

        return flattened
    except Exception:
        return None
